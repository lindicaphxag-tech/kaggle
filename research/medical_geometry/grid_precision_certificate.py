"""Patient-mm certificate for mixed-precision grid-coordinate quantization.

This certifies ONE numerical component: the source-coordinate drift caused by
storing normalized grid_sample coordinates in float16/bfloat16/float32/float64.
It does NOT certify arithmetic inside grid_sample, input/value rounding,
floating-point convolution, or the C2 interpolation residual.

For a separable 3-D affine index map each axis's quantization error depends
only on its own target index. The Euclidean norm under an arbitrarily sheared
patient affine is convex in those three errors. Its maximum over the finite
Cartesian product is attained at one of the 2^3 combinations of each axis's
min/max error. Thus an exact coordinate-storage maximum (modulo host float64
arithmetic) needs only O(D+H+W) work, not O(DHW).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from .clingeom import FeatureLattice
from .repair_synthesis import target_to_source_index_map


@dataclass(frozen=True)
class GridCoordinatePrecisionCertificate:
    input_dtype: str
    max_world_coordinate_displacement_mm: float
    max_absolute_source_index_displacement: tuple[float, float, float]
    source_index_error_ranges: tuple[tuple[float, float], ...]
    target_shape: tuple[int, int, int]
    resampling_executed: bool
    assumption: str


def certify_grid_coordinate_precision(
    source: FeatureLattice,
    target: FeatureLattice,
    *,
    input_voxel_to_patient_mm: np.ndarray,
    input_dtype: torch.dtype,
    resampling_executed: bool = True,
) -> GridCoordinatePrecisionCertificate:
    """Exact 8-corner physical displacement bound for stored grid coordinates.

    LatticeRepair3d._grid calculates normalized coordinates in float64 and
    casts them to x.dtype before passing them to torch.grid_sample.
    Here we reconstruct the source-index drift of those *stored* grid values.

    All candidate target centers must be within the source-grid extent. This
    guard makes this certificate compatible with ClinGeom's no-extrapolation
    transport policy. Under a non-resampling exact crop/identity, no normalized
    grid is emitted and the displacement is exactly zero.

    This is NOT an end-to-end floating-point value-error certificate.
    """
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if input_dtype not in allowed:
        raise ValueError("unsupported floating dtype for grid precision certificate")
    affine = np.asarray(input_voxel_to_patient_mm, dtype=np.float64)
    if (affine.shape != (4, 4)
            or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0],
                               atol=1e-12, rtol=0)
            or np.linalg.matrix_rank(affine[:3, :3]) != 3):
        raise ValueError("patient affine must be finite, invertible homogeneous 4x4")
    for name, lattice in (("source", source), ("target", target)):
        if (len(lattice.shape) != 3 or any(int(n) < 1 for n in lattice.shape)
                or not np.isfinite(lattice.start).all()
                or not np.isfinite(lattice.jump).all()
                or np.any(np.asarray(lattice.jump) <= 0)):
            raise ValueError(f"{name} grid must have finite positive sampling geometry")
    if not isinstance(resampling_executed, bool):
        raise ValueError("resampling_executed must be boolean")

    ranges: list[tuple[float, float]] = []
    if not resampling_executed:
        ranges = [(0.0, 0.0)] * 3
    else:
        m = target_to_source_index_map(source, target)
        if (m.shape != (4, 4) or not np.isfinite(m).all()
                or not np.allclose(m[:3, :3],
                                   np.diag(np.diag(m[:3, :3])),
                                   atol=1e-12, rtol=0)):
            raise ValueError("precision certificate requires separable source index map")

        for axis in range(3):
            size = int(source.shape[axis])
            count = int(target.shape[axis])
            index = torch.arange(count, dtype=torch.float64)
            source_index = index * float(m[axis, axis]) + float(m[axis, 3])
            if (torch.any(source_index < -1e-9).item()
                    or torch.any(source_index > size - 1 + 1e-9).item()):
                raise ValueError("target centers outside source support")
            if size == 1:
                if not torch.all(torch.abs(source_index) < 1e-9).item():
                    raise ValueError("single-node source cannot encode off-node targets")
                delta = torch.zeros_like(source_index)
            else:
                normalized = (2.0 * source_index / float(size - 1)) - 1.0
                stored = normalized.to(input_dtype).to(torch.float64)
                decoded_index = 0.5 * (stored + 1.0) * float(size - 1)
                delta = decoded_index - source_index
            ranges.append((float(delta.min()), float(delta.max())))

    # Source-index drift -> input index -> physical patient coordinates.
    B = affine[:3, :3] @ np.diag(np.asarray(source.jump, dtype=np.float64))
    longest = 0.0
    for corner in product(*((lo, hi) for lo, hi in ranges)):
        displacement = float(np.linalg.norm(B @ np.asarray(corner, dtype=np.float64)))
        longest = max(longest, displacement)
    return GridCoordinatePrecisionCertificate(
        input_dtype=str(input_dtype),
        max_world_coordinate_displacement_mm=longest,
        max_absolute_source_index_displacement=tuple(
            max(abs(lo), abs(hi)) for lo, hi in ranges
        ),
        source_index_error_ranges=tuple(ranges),
        target_shape=tuple(int(n) for n in target.shape),
        resampling_executed=resampling_executed,
        assumption=(
            "Coordinate-storage quantization only: float64 normalized "
            "grid values cast to input dtype before grid_sample; "
            "arithmetic/feature errors are outside this certificate"
        ),
    )
