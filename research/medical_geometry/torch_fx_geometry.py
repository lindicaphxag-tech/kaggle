"""torch.fx patient-space semantic type checker for 3-D medical models.

The checker propagates feature lattices through a traced PyTorch graph and
certifies branch merges. Tensor-shape compatibility is not sufficient: two
features may have identical shapes while sampling different patient-space
locations because of asymmetric padding or different operator geometry.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node, symbolic_trace

from .clingeom import FeatureLattice, feature_affine, physical_corners
from .torch_geometry import UnsupportedGeometryOp, _apply_supported_geometry


class GeometryMergeError(RuntimeError):
    """Raised when tensor-compatible branches disagree in patient space."""


@dataclass(frozen=True)
class MergeCertificate:
    node_name: str
    operation: str
    valid: bool
    shape_equal: bool
    max_center_displacement_mm: float
    reason: str


@dataclass(frozen=True)
class GraphGeometryResult:
    graph_module: GraphModule
    lattices: dict[str, FeatureLattice]
    merges: tuple[MergeCertificate, ...]
    output_lattice: FeatureLattice


_PAD_MODULES = (nn.ConstantPad3d, nn.ReflectionPad3d, nn.ReplicationPad3d)


def _six_padding(padding: Any) -> tuple[int, int, int, int, int, int]:
    if isinstance(padding, int):
        return (padding,) * 6
    values = tuple(int(x) for x in padding)
    if len(values) != 6:
        raise UnsupportedGeometryOp("3-D asymmetric padding must contain six values")
    if any(x < 0 for x in values):
        raise UnsupportedGeometryOp("negative padding/cropping is not certified by FX v1")
    return values


def _apply_padding(module: nn.Module, lattice: FeatureLattice) -> FeatureLattice:
    left, right, top, bottom, front, back = _six_padding(module.padding)
    # PyTorch pad order for 5-D tensors is (Wl, Wr, Hl, Hr, Dl, Dr).
    before = np.asarray([front, top, left], dtype=np.float64)
    after = np.asarray([back, bottom, right], dtype=np.float64)
    shape = np.asarray(lattice.shape, dtype=np.int64)
    out_shape = shape + before.astype(np.int64) + after.astype(np.int64)
    return FeatureLattice(
        tuple(int(x) for x in out_shape),
        lattice.jump.copy(),
        lattice.start - before * lattice.jump,
        lattice.receptive_field.copy(),
    )


def _corner_center_displacement_mm(
    a: FeatureLattice,
    b: FeatureLattice,
    input_affine: np.ndarray,
) -> tuple[bool, float]:
    shape_equal = a.shape == b.shape
    if not shape_equal:
        return False, float("inf")
    aa = feature_affine(input_affine, a)
    bb = feature_affine(input_affine, b)
    corners_a = physical_corners(a.shape, aa)
    corners_b = physical_corners(b.shape, bb)
    return True, float(np.max(np.linalg.norm(corners_a - corners_b, axis=1)))


def certify_merge(
    node_name: str,
    operation: str,
    lattices: list[FeatureLattice],
    input_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
) -> MergeCertificate:
    if len(lattices) < 2:
        raise ValueError("merge certification requires at least two feature lattices")
    ref = lattices[0]
    shape_equal = True
    max_disp = 0.0
    for other in lattices[1:]:
        same_shape, disp = _corner_center_displacement_mm(ref, other, input_affine)
        shape_equal = shape_equal and same_shape
        max_disp = max(max_disp, disp)
    valid = bool(shape_equal and max_disp <= tolerance_mm)
    if not shape_equal:
        reason = "merge inputs have different spatial tensor shapes"
    elif not valid:
        reason = "tensor-compatible merge inputs sample different patient-space locations"
    else:
        reason = "merge inputs share the same patient-space sampling lattice"
    return MergeCertificate(
        node_name, operation, valid, shape_equal, float(max_disp), reason
    )


def _node_lattice(value: Any, env: dict[Node, FeatureLattice]) -> FeatureLattice:
    if isinstance(value, Node) and value in env:
        return env[value]
    raise UnsupportedGeometryOp("expected a tensor-valued FX node with certified geometry")


def _merge_lattices_from_arg(value: Any, env: dict[Node, FeatureLattice]) -> list[FeatureLattice]:
    if isinstance(value, Node):
        return [_node_lattice(value, env)]
    if isinstance(value, (tuple, list)):
        out: list[FeatureLattice] = []
        for item in value:
            out.extend(_merge_lattices_from_arg(item, env))
        return out
    return []


def _resolve_upsample_shape(
    module: nn.Upsample,
    input_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if module.size is not None:
        if isinstance(module.size, int):
            return (module.size, module.size, module.size)
        size = tuple(int(x) for x in module.size)
        if len(size) != 3:
            raise UnsupportedGeometryOp("Upsample size must have three spatial values")
        return size

    scale = module.scale_factor
    if scale is None:
        raise UnsupportedGeometryOp("Upsample must define size or scale_factor")
    if isinstance(scale, (int, float)):
        factors = (float(scale),) * 3
    else:
        factors = tuple(float(x) for x in scale)
        if len(factors) != 3:
            raise UnsupportedGeometryOp("Upsample scale_factor must have three spatial values")
    out = tuple(int(np.floor(n * s)) for n, s in zip(input_shape, factors, strict=True))
    if any(n <= 0 for n in out):
        raise UnsupportedGeometryOp("Upsample produces an empty spatial dimension")
    return out


def _apply_upsample(module: nn.Upsample, lattice: FeatureLattice) -> FeatureLattice:
    if module.mode != "trilinear":
        raise UnsupportedGeometryOp(
            "FX v1 certifies nn.Upsample only for trilinear mode; nearest sampling "
            "does not define one affine source-sampling lattice"
        )
    if module.align_corners is None:
        raise UnsupportedGeometryOp("trilinear Upsample must declare align_corners")

    out_shape = _resolve_upsample_shape(module, lattice.shape)
    in_shape = np.asarray(lattice.shape, dtype=np.float64)
    out = np.asarray(out_shape, dtype=np.float64)

    if module.align_corners:
        if np.any(in_shape <= 1) or np.any(out <= 1):
            raise UnsupportedGeometryOp(
                "align_corners=True with singleton spatial dimensions is not certified by v1"
            )
        ratio = (in_shape - 1.0) / (out - 1.0)
        new_start = lattice.start.copy()
        new_jump = lattice.jump * ratio
    else:
        ratio = in_shape / out
        new_start = lattice.start + (0.5 * ratio - 0.5) * lattice.jump
        new_jump = lattice.jump * ratio

    # Interpolation mixes neighboring coarse features; one constant receptive-field
    # extent is generally phase-dependent, so v1 marks RF size unknown while still
    # certifying the affine grid of interpolation sample coordinates.
    new_rf = np.full(3, np.nan, dtype=np.float64)
    return FeatureLattice(
        out_shape,
        np.asarray(new_jump, dtype=np.float64),
        np.asarray(new_start, dtype=np.float64),
        new_rf,
    )

def _module_lattice(module: nn.Module, lattice: FeatureLattice) -> FeatureLattice:
    if isinstance(module, _PAD_MODULES):
        return _apply_padding(module, lattice)
    if isinstance(module, nn.Upsample):
        return _apply_upsample(module, lattice)
    return _apply_supported_geometry(module, lattice)


def compile_fx_geometry(
    module: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    input_affine: np.ndarray,
    merge_tolerance_mm: float = 1e-6,
) -> GraphGeometryResult:
    """Trace a model and type-check patient-space geometry at graph merges."""

    affine = np.asarray(input_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("input_affine must be a finite 4x4 matrix")
    if merge_tolerance_mm < 0:
        raise ValueError("merge_tolerance_mm must be nonnegative")

    gm = symbolic_trace(module)
    modules = dict(gm.named_modules())
    env: dict[Node, FeatureLattice] = {}
    merges: list[MergeCertificate] = []
    input_lattice = FeatureLattice.input_grid(input_shape)
    output_lattice: FeatureLattice | None = None

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if env:
                raise UnsupportedGeometryOp("FX v1 supports exactly one spatial tensor input")
            env[node] = input_lattice
            continue

        if node.op == "get_attr":
            continue

        if node.op == "call_module":
            source = _node_lattice(node.args[0], env)
            env[node] = _module_lattice(modules[str(node.target)], source)
            continue

        if node.op == "call_function":
            if node.target in (operator.add, torch.add):
                inputs = [_node_lattice(node.args[0], env), _node_lattice(node.args[1], env)]
                cert = certify_merge(
                    node.name, "add", inputs, affine, tolerance_mm=merge_tolerance_mm
                )
                merges.append(cert)
                if not cert.valid:
                    raise GeometryMergeError(
                        f"{node.name}: {cert.reason}; max displacement="
                        f"{cert.max_center_displacement_mm:.6g} mm"
                    )
                env[node] = inputs[0]
                continue

            if node.target is torch.cat:
                inputs = _merge_lattices_from_arg(node.args[0], env)
                cert = certify_merge(
                    node.name, "cat", inputs, affine, tolerance_mm=merge_tolerance_mm
                )
                merges.append(cert)
                if not cert.valid:
                    raise GeometryMergeError(
                        f"{node.name}: {cert.reason}; max displacement="
                        f"{cert.max_center_displacement_mm:.6g} mm"
                    )
                env[node] = inputs[0]
                continue

            if node.target in (torch.relu, F.relu):
                env[node] = _node_lattice(node.args[0], env)
                continue

            raise UnsupportedGeometryOp(
                f"FX function {node.target!r} has no certified patient-space geometry rule"
            )

        if node.op == "call_method":
            if node.target in ("add", "relu"):
                if node.target == "relu":
                    env[node] = _node_lattice(node.args[0], env)
                    continue
                inputs = [_node_lattice(node.args[0], env), _node_lattice(node.args[1], env)]
                cert = certify_merge(
                    node.name, "add", inputs, affine, tolerance_mm=merge_tolerance_mm
                )
                merges.append(cert)
                if not cert.valid:
                    raise GeometryMergeError(
                        f"{node.name}: {cert.reason}; max displacement="
                        f"{cert.max_center_displacement_mm:.6g} mm"
                    )
                env[node] = inputs[0]
                continue
            raise UnsupportedGeometryOp(
                f"FX method {node.target!r} has no certified patient-space geometry rule"
            )

        if node.op == "output":
            output_lattice = _node_lattice(node.args[0], env)
            continue

        raise UnsupportedGeometryOp(f"unsupported FX node operation: {node.op}")

    if output_lattice is None:
        raise UnsupportedGeometryOp("traced graph did not produce a certified tensor output")

    return GraphGeometryResult(
        graph_module=gm,
        lattices={node.name: lattice for node, lattice in env.items()},
        merges=tuple(merges),
        output_lattice=output_lattice,
    )

@dataclass(frozen=True)
class MergeRepairPlan:
    """Constructive repair class for two tensor-compatible feature lattices."""

    kind: str
    exact_on_retained_domain: bool
    shift_cells: tuple[float, float, float]
    reference_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    candidate_slices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None
    overlap_shape: tuple[int, int, int] | None
    max_center_displacement_mm: float
    reason: str


def _integer_overlap_slices(
    shape: tuple[int, int, int],
    shift: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    tuple[int, int, int],
]:
    ref_slices: list[tuple[int, int]] = []
    cand_slices: list[tuple[int, int]] = []
    overlap: list[int] = []
    for n, s in zip(shape, shift.astype(int), strict=True):
        ref_start = max(0, s)
        ref_end = min(n, n + s)
        cand_start = max(0, -s)
        cand_end = min(n, n - s)
        length = ref_end - ref_start
        if length <= 0 or length != cand_end - cand_start:
            raise GeometryMergeError("integer lattice shift has no valid common overlap")
        ref_slices.append((int(ref_start), int(ref_end)))
        cand_slices.append((int(cand_start), int(cand_end)))
        overlap.append(int(length))
    return tuple(ref_slices), tuple(cand_slices), tuple(overlap)


def plan_pairwise_merge_repair(
    reference: FeatureLattice,
    candidate: FeatureLattice,
    input_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
    integer_tolerance: float = 1e-8,
) -> MergeRepairPlan:
    """Classify an exact-overlap crop repair versus interpolation-required repair."""

    same_shape, max_disp = _corner_center_displacement_mm(
        reference, candidate, input_affine
    )
    if same_shape and max_disp <= tolerance_mm:
        return MergeRepairPlan(
            "aligned",
            True,
            (0.0, 0.0, 0.0),
            tuple((0, int(n)) for n in reference.shape),
            tuple((0, int(n)) for n in candidate.shape),
            reference.shape,
            max_disp,
            "no repair required",
        )

    if reference.shape != candidate.shape:
        return MergeRepairPlan(
            "resample_or_resize",
            False,
            (float("nan"),) * 3,
            None,
            None,
            None,
            max_disp,
            "spatial shapes differ; exact cellwise fusion is undefined",
        )

    if not np.allclose(reference.jump, candidate.jump, atol=integer_tolerance, rtol=0):
        return MergeRepairPlan(
            "resample",
            False,
            (float("nan"),) * 3,
            None,
            None,
            None,
            max_disp,
            "feature lattice spacing differs; interpolation is required",
        )

    shift = (candidate.start - reference.start) / reference.jump
    nearest = np.rint(shift)
    if np.allclose(shift, nearest, atol=integer_tolerance, rtol=0):
        ref_slices, cand_slices, overlap = _integer_overlap_slices(
            reference.shape, nearest.astype(int)
        )
        return MergeRepairPlan(
            "crop_to_common_lattice",
            True,
            tuple(float(x) for x in nearest),
            ref_slices,
            cand_slices,
            overlap,
            max_disp,
            "integer-cell offset admits exact no-interpolation fusion on the common physical domain",
        )

    return MergeRepairPlan(
        "resample",
        False,
        tuple(float(x) for x in shift),
        None,
        None,
        None,
        max_disp,
        "fractional-cell offset cannot be repaired by reindexing/cropping alone",
    )
