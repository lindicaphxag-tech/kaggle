"""Channel-fibered patient-space semantics for medical feature tensors.

A conventional feature tensor is usually treated as if every channel shares one
spatial sampling lattice. That assumption is false for architectures that
concatenate different sampling phases along channels (for example DARTS-style
factorized reduction used by MONAI DiNTS).

This module represents spatial semantics as channel fibers, each carrying one or
more affine sampling lattices. It can certify when a whole tensor admits one
global feature affine and track when channel mixing destroys that property.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .clingeom import FeatureLattice, feature_affine, physical_corners, propagate_conv_lattice


@dataclass(frozen=True)
class ChannelFiber:
    """A contiguous channel range sharing the same set of spatial supports."""

    channel_start: int
    channel_stop: int
    supports: tuple[FeatureLattice, ...]
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.channel_start < 0 or self.channel_stop <= self.channel_start:
            raise ValueError("channel range must be non-empty and nonnegative")
        if not self.supports:
            raise ValueError("a channel fiber must carry at least one spatial support")
        shape = self.supports[0].shape
        if any(s.shape != shape for s in self.supports):
            raise ValueError("all supports in one fiber must share tensor spatial shape")

    @property
    def channels(self) -> int:
        return self.channel_stop - self.channel_start


@dataclass(frozen=True)
class LatticeBundle:
    """Patient-space spatial type of a multi-channel feature tensor."""

    shape: tuple[int, int, int]
    channels: int
    fibers: tuple[ChannelFiber, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(n) <= 0 for n in self.shape):
            raise ValueError("shape must contain three positive spatial dimensions")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if not self.fibers:
            raise ValueError("bundle must contain at least one channel fiber")
        ordered = sorted(self.fibers, key=lambda f: f.channel_start)
        cursor = 0
        for fiber in ordered:
            if fiber.channel_start != cursor:
                raise ValueError("channel fibers must exactly partition the channel axis")
            if any(s.shape != self.shape for s in fiber.supports):
                raise ValueError("fiber support shape must match bundle shape")
            cursor = fiber.channel_stop
        if cursor != self.channels:
            raise ValueError("channel fibers must cover all bundle channels")


@dataclass(frozen=True)
class SingleLatticeCertificate:
    """Whether one global affine lattice is valid for the full tensor."""

    valid: bool
    unique_center_lattices: int
    max_intra_tensor_displacement_mm: float
    single_affine_minimax_lower_bound_mm: float
    representative: FeatureLattice | None
    reason: str

@dataclass(frozen=True)
class PointAffinePartitionCertificate:
    """Minimal exact channel partition into point-valued patient-space lattices."""

    point_affine_representable: bool
    distinct_patient_lattice_types: int | None
    minimum_contiguous_fibers: int | None
    globally_single_affine: bool
    fiber_type_ids: tuple[int, ...]
    reason: str

@dataclass(frozen=True)
class ParallelLatticeCollapseCertificate:
    """Exact minimax collapse certificate for parallel patient-space lattices."""

    applicable: bool
    num_unique_lattices: int
    common_linear: np.ndarray | None
    optimal_translation_mm: np.ndarray | None
    exact_minimax_radius_mm: float | None
    diameter_lower_bound_mm: float | None
    optimal_affine: np.ndarray | None
    safe_under_tolerance: bool | None
    reason: str


def bundle_from_lattice(
    channels: int,
    lattice: FeatureLattice,
    *,
    provenance: str = "homogeneous",
) -> LatticeBundle:
    if channels <= 0:
        raise ValueError("channels must be positive")
    fiber = ChannelFiber(0, channels, (lattice,), provenance)
    return LatticeBundle(lattice.shape, channels, (fiber,))


def _center_signature(lattice: FeatureLattice, decimals: int = 12) -> tuple:
    return (
        lattice.shape,
        tuple(np.round(lattice.jump, decimals=decimals)),
        tuple(np.round(lattice.start, decimals=decimals)),
    )


def _deduplicate_supports(supports: Iterable[FeatureLattice]) -> tuple[FeatureLattice, ...]:
    seen: dict[tuple, FeatureLattice] = {}
    for support in supports:
        seen.setdefault(_center_signature(support), support)
    return tuple(seen.values())


def _max_pairwise_center_displacement_mm(
    supports: tuple[FeatureLattice, ...],
    input_affine: np.ndarray,
) -> float:
    if len(supports) <= 1:
        return 0.0
    maximum = 0.0
    for i, a in enumerate(supports):
        A = feature_affine(input_affine, a)
        corners_a = physical_corners(a.shape, A)
        for b in supports[i + 1 :]:
            if a.shape != b.shape:
                return float("inf")
            B = feature_affine(input_affine, b)
            corners_b = physical_corners(b.shape, B)
            maximum = max(
                maximum,
                float(np.max(np.linalg.norm(corners_a - corners_b, axis=1))),
            )
    return maximum


def all_bundle_supports(bundle: LatticeBundle) -> tuple[FeatureLattice, ...]:
    return _deduplicate_supports(
        support for fiber in bundle.fibers for support in fiber.supports
    )


def certify_single_lattice(
    bundle: LatticeBundle,
    input_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
) -> SingleLatticeCertificate:
    """Certify whether every channel can share one patient-space feature affine."""

    supports = all_bundle_supports(bundle)
    max_disp = _max_pairwise_center_displacement_mm(supports, input_affine)
    valid = bool(len(supports) == 1 or max_disp <= tolerance_mm)
    if valid:
        return SingleLatticeCertificate(
            True,
            len(supports),
            max_disp,
            0.0,
            supports[0],
            "all channel fibers share one patient-space center lattice",
        )
    return SingleLatticeCertificate(
        False,
        len(supports),
        max_disp,
        max_disp / 2.0,
        None,
        "one global feature affine would collapse distinct channel sampling lattices",
    )


def certify_minimal_point_affine_partition(
    bundle: LatticeBundle,
    input_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
) -> PointAffinePartitionCertificate:
    """Certify the minimal exact point-affine channel representation.

    This strengthens certify_single_lattice. If any channel fiber structurally
    depends on multiple distinct lattices, no partition into point-valued
    affine fibers is exact. Otherwise every fiber has one patient-space affine
    center map; the certificate counts both distinct map types and the minimum
    channel-contiguous segments required by the current channel order.
    """
    ordered = tuple(sorted(bundle.fibers, key=lambda fiber: fiber.channel_start))
    patient_affines: list[np.ndarray] = []

    for fiber in ordered:
        supports = _deduplicate_supports(fiber.supports)
        if len(supports) != 1:
            return PointAffinePartitionCertificate(
                point_affine_representable=False,
                distinct_patient_lattice_types=None,
                minimum_contiguous_fibers=None,
                globally_single_affine=False,
                fiber_type_ids=(),
                reason=(
                    'at least one channel fiber has multiple structural patient-space '
                    'supports, so no exact point-affine channel partition exists'
                ),
            )
        patient_affines.append(feature_affine(input_affine, supports[0]))

    representatives: list[np.ndarray] = []
    type_ids: list[int] = []

    for affine in patient_affines:
        corners = physical_corners(bundle.shape, affine)
        assigned: int | None = None
        for class_id, representative in enumerate(representatives):
            representative_corners = physical_corners(bundle.shape, representative)
            error = float(
                np.max(np.linalg.norm(corners - representative_corners, axis=1))
            )
            if error <= tolerance_mm:
                assigned = class_id
                break
        if assigned is None:
            representatives.append(affine)
            assigned = len(representatives) - 1
        type_ids.append(assigned)

    distinct_types = len(representatives)
    contiguous = 1 + sum(
        left != right for left, right in zip(type_ids, type_ids[1:])
    )
    globally_single = distinct_types == 1

    return PointAffinePartitionCertificate(
        point_affine_representable=True,
        distinct_patient_lattice_types=distinct_types,
        minimum_contiguous_fibers=contiguous,
        globally_single_affine=globally_single,
        fiber_type_ids=tuple(type_ids),
        reason=(
            'all channel fibers share one patient-space affine'
            if globally_single
            else (
                'multiple patient-space affine types are required; the reported '
                'contiguous-fiber count is minimal for the current channel order'
            )
        ),
    )

def _sphere_from_boundary(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return the unique minimum-radius sphere through 1..4 boundary points."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0 or n > 4 or pts.shape != (n, 3):
        raise ValueError('boundary must contain 1..4 three-dimensional points')
    if n == 1:
        return pts[0].copy(), 0.0
    if n == 2:
        center = 0.5 * (pts[0] + pts[1])
        return center, float(np.linalg.norm(pts[0] - center))
    if n == 3:
        a, b, d = pts
        v1 = b - a
        v2 = d - a
        gram = np.array(
            [[np.dot(v1, v1), np.dot(v1, v2)],
             [np.dot(v1, v2), np.dot(v2, v2)]],
            dtype=np.float64,
        )
        rhs = 0.5 * np.array([np.dot(v1, v1), np.dot(v2, v2)])
        if np.linalg.matrix_rank(gram, tol=1e-12) < 2:
            return None
        coeff = np.linalg.solve(gram, rhs)
        center = a + coeff[0] * v1 + coeff[1] * v2
        return center, float(np.linalg.norm(center - a))
    a = pts[0]
    matrix = 2.0 * (pts[1:] - a)
    rhs = np.sum(pts[1:] ** 2, axis=1) - np.sum(a ** 2)
    if np.linalg.matrix_rank(matrix, tol=1e-12) < 3:
        return None
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(center - a))


def _smallest_enclosing_ball(points: np.ndarray, *, atol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Exact finite 3-D minimum enclosing ball by boundary enumeration.

    In R^3 an enclosing ball is determined by at most four boundary points.
    This exhaustive implementation is intended for the small number of spatial
    support lattices carried by one feature tensor, not large point clouds.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError('points must have shape [N, 3] with N > 0')
    unique = np.unique(np.round(pts, decimals=12), axis=0)
    best_center = unique[0].copy()
    best_radius = float('inf')
    for size in range(1, min(4, len(unique)) + 1):
        for ids in combinations(range(len(unique)), size):
            candidate = _sphere_from_boundary(unique[list(ids)])
            if candidate is None:
                continue
            center, radius = candidate
            distances = np.linalg.norm(unique - center, axis=1)
            cover_tol = atol * max(1.0, radius, float(distances.max(initial=0.0)))
            if np.all(distances <= radius + cover_tol):
                if radius < best_radius - cover_tol:
                    best_center = center
                    best_radius = radius
    if not np.isfinite(best_radius):
        raise RuntimeError('failed to construct a finite enclosing ball')
    return best_center, float(best_radius)


def certify_parallel_lattice_collapse(
    bundle: LatticeBundle,
    input_affine: np.ndarray,
    *,
    tolerance_mm: float = 1e-6,
    linear_tolerance: float = 1e-10,
) -> ParallelLatticeCollapseCertificate:
    """Compute the exact best single-affine approximation for parallel lattices.

    When all unique patient-space support affines share one linear part and
    differ only by translation, the worst-case error of collapsing them to one
    affine is independent of feature index. Minimizing that error is exactly
    the Euclidean minimum-enclosing-ball problem over their translation vectors.
    """
    supports = all_bundle_supports(bundle)
    affines = [feature_affine(input_affine, support) for support in supports]
    common = affines[0][:3, :3].copy()
    if any(
        not np.allclose(a[:3, :3], common, atol=linear_tolerance, rtol=linear_tolerance)
        for a in affines[1:]
    ):
        return ParallelLatticeCollapseCertificate(
            applicable=False,
            num_unique_lattices=len(affines),
            common_linear=None,
            optimal_translation_mm=None,
            exact_minimax_radius_mm=None,
            diameter_lower_bound_mm=None,
            optimal_affine=None,
            safe_under_tolerance=None,
            reason=(
                'support lattices do not share one patient-space linear part; '
                'parallel-lattice collapse theorem does not apply'
            ),
        )

    translations = np.stack([a[:3, 3] for a in affines], axis=0)
    center, radius = _smallest_enclosing_ball(translations)
    diameter = 0.0
    for i, left in enumerate(translations):
        for right in translations[i + 1:]:
            diameter = max(diameter, float(np.linalg.norm(left - right)))

    optimal = np.eye(4, dtype=np.float64)
    optimal[:3, :3] = common
    optimal[:3, 3] = center
    return ParallelLatticeCollapseCertificate(
        applicable=True,
        num_unique_lattices=len(affines),
        common_linear=common,
        optimal_translation_mm=center,
        exact_minimax_radius_mm=radius,
        diameter_lower_bound_mm=diameter / 2.0,
        optimal_affine=optimal,
        safe_under_tolerance=bool(radius <= tolerance_mm),
        reason=(
            'exact minimax radius is the minimum-enclosing-ball radius of '
            'patient-space lattice translations'
        ),
    )

def shift_lattice_origin(
    lattice: FeatureLattice,
    offset_cells: tuple[int, int, int],
    *,
    new_shape: tuple[int, int, int] | None = None,
) -> FeatureLattice:
    offset = np.asarray(offset_cells, dtype=np.float64)
    shape = lattice.shape if new_shape is None else new_shape
    return FeatureLattice(
        shape,
        lattice.jump.copy(),
        lattice.start + offset * lattice.jump,
        lattice.receptive_field.copy(),
    )


def factorized_reduce_bundle(
    bundle: LatticeBundle,
    *,
    out_channels: int,
) -> LatticeBundle:
    """Propagate MONAI/DARTS-style shifted factorized stride-2 reduction.

    The first output channel group samples phase (0,0,0); the second first
    slices the input by one cell on every spatial axis and samples phase
    (1,1,1). MONAI requires spatial lengths to be multiples of two.
    """

    if out_channels < 2:
        raise ValueError("out_channels must be at least 2")
    if any(n % 2 for n in bundle.shape):
        raise ValueError("factorized reduction requires even spatial dimensions")

    first_channels = out_channels // 2
    second_channels = out_channels - first_channels
    branch_a: list[FeatureLattice] = []
    branch_b: list[FeatureLattice] = []

    for support in all_bundle_supports(bundle):
        branch_a.append(
            propagate_conv_lattice(support, kernel_size=1, stride=2, padding=0)
        )

        sliced_shape = tuple(int(n - 1) for n in support.shape)
        shifted = shift_lattice_origin(
            support, (1, 1, 1), new_shape=sliced_shape
        )
        branch_b.append(
            propagate_conv_lattice(shifted, kernel_size=1, stride=2, padding=0)
        )

    supports_a = _deduplicate_supports(branch_a)
    supports_b = _deduplicate_supports(branch_b)
    out_shape = supports_a[0].shape
    if any(s.shape != out_shape for s in supports_b):
        raise ValueError("factorized branches do not have equal output shapes")

    fibers = (
        ChannelFiber(
            0,
            first_channels,
            supports_a,
            "factorized_reduce phase=(0,0,0)",
        ),
        ChannelFiber(
            first_channels,
            first_channels + second_channels,
            supports_b,
            "factorized_reduce phase=(1,1,1)",
        ),
    )
    return LatticeBundle(out_shape, out_channels, fibers)


def dense_conv_mix_bundle(
    bundle: LatticeBundle,
    *,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
) -> LatticeBundle:
    """Propagate a groups=1 convolution that mixes every input channel.

    Every output channel depends on every incoming channel fiber, so the output
    carries the union of all transformed support lattices. A multi-support
    output intentionally has no single feature affine certificate.
    """

    if out_channels <= 0:
        raise ValueError("out_channels must be positive")
    transformed = [
        propagate_conv_lattice(
            support,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        for support in all_bundle_supports(bundle)
    ]
    supports = _deduplicate_supports(transformed)
    shape = supports[0].shape
    if any(s.shape != shape for s in supports):
        raise ValueError("mixed supports produce incompatible output shapes")
    return LatticeBundle(
        shape,
        out_channels,
        (ChannelFiber(0, out_channels, supports, "dense channel-mixing convolution"),),
    )


def bundle_summary(bundle: LatticeBundle) -> dict:
    """JSON-friendly structural summary for evidence artifacts."""

    return {
        "shape": list(bundle.shape),
        "channels": bundle.channels,
        "fibers": [
            {
                "channel_range": [fiber.channel_start, fiber.channel_stop],
                "provenance": fiber.provenance,
                "supports": [
                    {
                        "jump": support.jump.tolist(),
                        "start": support.start.tolist(),
                        "receptive_field": support.receptive_field.tolist(),
                    }
                    for support in fiber.supports
                ],
            }
            for fiber in bundle.fibers
        ],
    }

def map_bundle_supports(
    bundle: LatticeBundle,
    transform,
    *,
    provenance_suffix: str = "",
) -> LatticeBundle:
    """Apply one spatial map independently to every channel fiber support."""

    fibers: list[ChannelFiber] = []
    out_shape: tuple[int, int, int] | None = None
    for fiber in bundle.fibers:
        supports = _deduplicate_supports(transform(s) for s in fiber.supports)
        if out_shape is None:
            out_shape = supports[0].shape
        if any(s.shape != out_shape for s in supports):
            raise ValueError("support transform produced inconsistent output shapes")
        provenance = fiber.provenance
        if provenance_suffix:
            provenance = f"{provenance} -> {provenance_suffix}" if provenance else provenance_suffix
        fibers.append(
            ChannelFiber(
                fiber.channel_start,
                fiber.channel_stop,
                supports,
                provenance,
            )
        )
    assert out_shape is not None
    return LatticeBundle(out_shape, bundle.channels, tuple(fibers))


def concat_channel_bundles(
    bundles: Iterable[LatticeBundle],
    *,
    provenance: str = "channel concatenation",
) -> LatticeBundle:
    """Concatenate tensors on channels without falsely requiring one shared lattice."""

    seq = tuple(bundles)
    if len(seq) < 2:
        raise ValueError("channel concatenation requires at least two bundles")
    shape = seq[0].shape
    if any(bundle.shape != shape for bundle in seq):
        raise ValueError("channel concatenation requires equal spatial tensor shapes")

    fibers: list[ChannelFiber] = []
    cursor = 0
    for bundle in seq:
        for fiber in bundle.fibers:
            width = fiber.channels
            source = fiber.provenance
            label = f"{provenance}: {source}" if source else provenance
            fibers.append(
                ChannelFiber(cursor, cursor + width, fiber.supports, label)
            )
            cursor += width
    return LatticeBundle(shape, cursor, tuple(fibers))


def slice_bundle_spatial(
    bundle: LatticeBundle,
    starts: tuple[int, int, int],
    stops: tuple[int | None, int | None, int | None],
) -> LatticeBundle:
    """Apply a positive-step spatial slice while preserving patient-space coordinates."""

    new_shape: list[int] = []
    for n, start, stop in zip(bundle.shape, starts, stops, strict=True):
        if start < 0:
            raise ValueError("negative spatial slice starts are not certified")
        end = n if stop is None else int(stop)
        if end < 0:
            end += n
        end = min(n, end)
        if start >= end:
            raise ValueError("spatial slice would be empty")
        new_shape.append(end - start)
    shape = tuple(new_shape)

    def transform(support: FeatureLattice) -> FeatureLattice:
        return shift_lattice_origin(support, starts, new_shape=shape)

    return map_bundle_supports(
        bundle, transform, provenance_suffix=f"slice starts={starts}"
    )
