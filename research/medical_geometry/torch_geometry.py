"""PyTorch model-graph compiler for ClinGeom patient-space lattices.

The first implementation intentionally supports only sequential 3-D geometry
whose receptive-field lattice can be derived exactly. Unsupported spatial
operators fail closed instead of receiving shape-only metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch.nn as nn

from .clingeom import FeatureLattice, feature_affine, propagate_conv_lattice


class UnsupportedGeometryOp(RuntimeError):
    """Raised when patient-space feature geometry cannot be certified."""


@dataclass(frozen=True)
class LayerGeometry:
    name: str
    module_type: str
    lattice: FeatureLattice
    patient_affine: np.ndarray


_PASS_THROUGH = (
    nn.Identity,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.ELU,
    nn.PReLU,
    nn.Dropout,
    nn.Dropout3d,
)


def _spatial_tuple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, str):
        raise UnsupportedGeometryOp(
            f"{name}={value!r} is dynamic/string-valued and is not certified by v1"
        )
    if isinstance(value, int):
        return (value, value, value)
    seq = tuple(int(x) for x in value)
    if len(seq) != 3:
        raise UnsupportedGeometryOp(f"{name} must resolve to exactly three spatial values")
    return seq


def _pool_stride(module: nn.MaxPool3d | nn.AvgPool3d) -> tuple[int, int, int]:
    stride = module.stride if module.stride is not None else module.kernel_size
    return _spatial_tuple(stride, "stride")


def _apply_supported_geometry(module: nn.Module, lattice: FeatureLattice) -> FeatureLattice:
    # Geometric center maps and dependency support are DIFFERENT contracts.
    # GroupNorm always uses current-volume spatial statistics; BatchNorm3d
    # and InstanceNorm3d use them in training (and in eval if running stats
    # are disabled). Such an operation makes an output voxel depend on distant
    # voxels. Reusing the unchanged local receptive_field would be unsound.
    if isinstance(module, nn.GroupNorm):
        raise UnsupportedGeometryOp(
            "GroupNorm globally couples spatial voxels; local receptive support "
            "cannot be passed through without a global-dependency type"
        )
    if isinstance(module, (nn.InstanceNorm3d, nn.BatchNorm3d)):
        if module.training or not module.track_running_stats:
            raise UnsupportedGeometryOp(
                f"{type(module).__name__} uses input-dependent spatial statistics; "
                "a local receptive-support certificate is not sound"
            )
        # In eval mode with tracked running statistics the affine normalization
        # is pointwise in space, so the sampling/support lattice is unchanged.
        return lattice
    if isinstance(module, nn.Conv3d):
        return propagate_conv_lattice(
            lattice,
            kernel_size=_spatial_tuple(module.kernel_size, "kernel_size"),
            stride=_spatial_tuple(module.stride, "stride"),
            padding=_spatial_tuple(module.padding, "padding"),
            dilation=_spatial_tuple(module.dilation, "dilation"),
        )

    if isinstance(module, nn.MaxPool3d):
        if module.ceil_mode:
            raise UnsupportedGeometryOp("MaxPool3d ceil_mode=True is not certified by v1")
        return propagate_conv_lattice(
            lattice,
            kernel_size=_spatial_tuple(module.kernel_size, "kernel_size"),
            stride=_pool_stride(module),
            padding=_spatial_tuple(module.padding, "padding"),
            dilation=_spatial_tuple(module.dilation, "dilation"),
        )

    if isinstance(module, nn.AvgPool3d):
        if module.ceil_mode:
            raise UnsupportedGeometryOp("AvgPool3d ceil_mode=True is not certified by v1")
        return propagate_conv_lattice(
            lattice,
            kernel_size=_spatial_tuple(module.kernel_size, "kernel_size"),
            stride=_pool_stride(module),
            padding=_spatial_tuple(module.padding, "padding"),
            dilation=1,
        )

    if isinstance(module, _PASS_THROUGH):
        return lattice

    raise UnsupportedGeometryOp(
        f"{type(module).__name__} has no certified patient-space geometry rule"
    )


def _leaf_modules(module: nn.Module, prefix: str = "") -> Iterable[tuple[str, nn.Module]]:
    children = list(module.named_children())
    if not children:
        yield prefix or "<root>", module
        return

    # v1 only treats containers as serial composition. Arbitrary forward graphs
    # with children can branch/merge in ways named_children cannot recover.
    if not isinstance(module, nn.Sequential):
        raise UnsupportedGeometryOp(
            f"{type(module).__name__} is not an nn.Sequential container; "
            "v1 refuses to infer branch/merge semantics from module registration order"
        )

    for name, child in children:
        path = f"{prefix}.{name}" if prefix else name
        yield from _leaf_modules(child, path)


def compile_sequential_geometry(
    module: nn.Module,
    *,
    input_shape: tuple[int, int, int],
    input_affine: np.ndarray,
) -> list[LayerGeometry]:
    """Compile exact patient-space lattices through a supported sequential model."""

    affine = np.asarray(input_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("input_affine must be a finite 4x4 matrix")

    lattice = FeatureLattice.input_grid(input_shape)
    trace: list[LayerGeometry] = [
        LayerGeometry("<input>", "Input", lattice, feature_affine(affine, lattice))
    ]

    for name, layer in _leaf_modules(module):
        lattice = _apply_supported_geometry(layer, lattice)
        trace.append(
            LayerGeometry(name, type(layer).__name__, lattice, feature_affine(affine, lattice))
        )

    return trace
