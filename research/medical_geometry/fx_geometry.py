"""torch.fx patient-space type checking for branched medical networks.

The compiler traces a PyTorch model, propagates ClinGeom feature lattices through
supported spatial operators, and checks add/cat merge points. Tensor-shape
compatibility is not sufficient: merge inputs must denote the same patient-space
sampling lattice, or the compiler emits an atomic repair obligation.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn

from .clingeom import FeatureLattice, feature_affine
from .repair_synthesis import MergeRepairBundle, synthesize_merge_repair
from .torch_geometry import UnsupportedGeometryOp, _apply_supported_geometry


@dataclass(frozen=True)
class FXNodeGeometry:
    node_name: str
    op: str
    target: str
    lattice: FeatureLattice
    patient_affine: np.ndarray


@dataclass(frozen=True)
class MergeRepairObligation:
    node_name: str
    merge_kind: str
    left_node: str
    right_node: str
    bundle: MergeRepairBundle
    reason: str


@dataclass(frozen=True)
class FXGeometryReport:
    graph_module: fx.GraphModule
    nodes: tuple[FXNodeGeometry, ...]
    repair_obligations: tuple[MergeRepairObligation, ...]
    output_lattice: FeatureLattice


class PatientSpaceMergeError(RuntimeError):
    """Raised when strict type checking encounters a merge requiring repair."""


def _same_merge_lattice(
    a: FeatureLattice,
    b: FeatureLattice,
    *,
    atol: float = 1e-9,
) -> bool:
    return (
        tuple(a.shape) == tuple(b.shape)
        and np.allclose(a.jump, b.jump, atol=atol, rtol=0)
        and np.allclose(a.start, b.start, atol=atol, rtol=0)
    )


def _node_lattice(env: dict[fx.Node, FeatureLattice], value: Any) -> FeatureLattice:
    if not isinstance(value, fx.Node) or value not in env:
        raise UnsupportedGeometryOp("geometry input is not a previously typed tensor node")
    return env[value]


def _merge_pair(
    *,
    node: fx.Node,
    kind: str,
    left_node: fx.Node,
    right_node: fx.Node,
    env: dict[fx.Node, FeatureLattice],
    obligations: list[MergeRepairObligation],
    strict: bool,
    role: str,
) -> FeatureLattice:
    left = _node_lattice(env, left_node)
    right = _node_lattice(env, right_node)

    if tuple(left.shape) != tuple(right.shape):
        raise UnsupportedGeometryOp(
            f"{kind} at {node.name!r} has ordinary spatial shape mismatch: "
            f"{left.shape} vs {right.shape}"
        )

    if _same_merge_lattice(left, right):
        return FeatureLattice(
            left.shape,
            np.asarray(left.jump, dtype=np.float64).copy(),
            np.asarray(left.start, dtype=np.float64).copy(),
            np.maximum(
                np.asarray(left.receptive_field, dtype=np.float64),
                np.asarray(right.receptive_field, dtype=np.float64),
            ),
        )

    bundle = synthesize_merge_repair(left, right, role=role)
    obligations.append(
        MergeRepairObligation(
            node_name=node.name,
            merge_kind=kind,
            left_node=left_node.name,
            right_node=right_node.name,
            bundle=bundle,
            reason=(
                "tensor shapes agree but patient-space feature-center lattices differ; "
                "fusion requires the certified atomic repair bundle"
            ),
        )
    )

    if strict:
        raise PatientSpaceMergeError(
            f"{kind} at {node.name!r} is tensor-valid but patient-space invalid; "
            f"repair bundle {bundle.bundle_digest}"
        )

    if "refuse" in {bundle.left.kind, bundle.right.kind}:
        raise UnsupportedGeometryOp(
            f"{kind} at {node.name!r} has no certified repair in ClinGeom v1"
        )

    return bundle.target


def compile_fx_geometry(
    module: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    input_affine: np.ndarray,
    role: str = "feature",
    strict: bool = False,
    require_channel_aligned_cat: bool = False,
) -> FXGeometryReport:
    """Trace a PyTorch DAG under an optional single-lattice fusion contract.

    Pointwise addition always requires matching patient-space sampling lattices.
    Channel concatenation may intentionally preserve different sampling phases.
    By default, mismatched channel cat belongs in the LatticeBundle compiler.
    Set require_channel_aligned_cat=True only when the application contract
    explicitly requires aligned channel centers before concatenation.
    """

    affine = np.asarray(input_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("input_affine must be a finite 4x4 matrix")

    gm = fx.symbolic_trace(module)
    env: dict[fx.Node, FeatureLattice] = {}
    typed_nodes: list[FXNodeGeometry] = []
    obligations: list[MergeRepairObligation] = []
    output_lattice: FeatureLattice | None = None
    placeholder_count = 0

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_count += 1
            if placeholder_count > 1:
                raise UnsupportedGeometryOp("ClinGeom fx v1 supports one tensor input")
            lattice = FeatureLattice.input_grid(input_shape)
            env[node] = lattice

        elif node.op == "call_module":
            layer = gm.get_submodule(str(node.target))
            if not node.args:
                raise UnsupportedGeometryOp(f"module {node.target!r} has no tensor input")
            source = _node_lattice(env, node.args[0])
            lattice = _apply_supported_geometry(layer, source)
            env[node] = lattice

        elif node.op == "call_function" and node.target in {operator.add, torch.add}:
            if len(node.args) < 2:
                raise UnsupportedGeometryOp("add node must have two tensor inputs")
            lattice = _merge_pair(
                node=node,
                kind="add",
                left_node=node.args[0],
                right_node=node.args[1],
                env=env,
                obligations=obligations,
                strict=strict,
                role=role,
            )
            env[node] = lattice

        elif node.op == "call_function" and node.target is torch.cat:
            tensors = node.args[0]
            if not isinstance(tensors, (tuple, list)) or len(tensors) < 2:
                raise UnsupportedGeometryOp("torch.cat must receive at least two typed tensors")
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
            if dim not in {1, -4}:
                raise UnsupportedGeometryOp(
                    "ClinGeom fx v1 only certifies channel-wise cat for NCDHW tensors"
                )
            current_node = tensors[0]
            current = _node_lattice(env, current_node)
            for other_node in tensors[1:]:
                other = _node_lattice(env, other_node)
                if tuple(current.shape) != tuple(other.shape):
                    raise UnsupportedGeometryOp(
                        f"cat at {node.name!r} has ordinary spatial shape mismatch"
                    )
                if _same_merge_lattice(current, other):
                    current = FeatureLattice(
                        current.shape,
                        np.asarray(current.jump).copy(),
                        np.asarray(current.start).copy(),
                        np.maximum(
                            np.asarray(current.receptive_field),
                            np.asarray(other.receptive_field),
                        ),
                    )
                    current_node = other_node
                    continue
                if not require_channel_aligned_cat:
                    raise UnsupportedGeometryOp(
                        f"cat at {node.name!r} contains distinct channel sampling lattices; "
                        "generic channel concatenation should preserve them as a LatticeBundle. "
                        "Use compile_fx_bundle_geometry, or set "
                        "require_channel_aligned_cat=True only for an aligned-fusion contract."
                    )

                bundle = synthesize_merge_repair(current, other, role=role)
                obligations.append(
                    MergeRepairObligation(
                        node_name=node.name,
                        merge_kind="cat",
                        left_node=getattr(current_node, "name", "<merged>"),
                        right_node=other_node.name,
                        bundle=bundle,
                        reason=("application declared channel-aligned fusion, but tensor-valid ""channel groups have different feature-center lattices"),
                    )
                )
                if strict:
                    raise PatientSpaceMergeError(
                        f"cat at {node.name!r} requires repair bundle {bundle.bundle_digest}"
                    )
                if "refuse" in {bundle.left.kind, bundle.right.kind}:
                    raise UnsupportedGeometryOp(
                        f"cat at {node.name!r} has no certified repair in ClinGeom v1"
                    )
                current = bundle.target
                current_node = node
            lattice = current
            env[node] = lattice

        elif node.op == "call_function" and node.target is torch.relu:
            lattice = _node_lattice(env, node.args[0])
            env[node] = lattice

        elif node.op == "output":
            raw = node.args[0]
            if isinstance(raw, fx.Node):
                output_lattice = _node_lattice(env, raw)
            elif isinstance(raw, (tuple, list)) and len(raw) == 1 and isinstance(raw[0], fx.Node):
                output_lattice = _node_lattice(env, raw[0])
            else:
                raise UnsupportedGeometryOp("ClinGeom fx v1 requires one tensor output")
            continue

        elif node.op == "get_attr":
            continue

        else:
            raise UnsupportedGeometryOp(
                f"FX node {node.name!r}: op={node.op!r}, target={node.target!r} "
                "has no certified patient-space rule"
            )

        typed_nodes.append(
            FXNodeGeometry(
                node_name=node.name,
                op=node.op,
                target=str(node.target),
                lattice=lattice,
                patient_affine=feature_affine(affine, lattice),
            )
        )

    if placeholder_count != 1 or output_lattice is None:
        raise UnsupportedGeometryOp("failed to establish one input and one output lattice")

    return FXGeometryReport(
        graph_module=gm,
        nodes=tuple(typed_nodes),
        repair_obligations=tuple(obligations),
        output_lattice=output_lattice,
    )
