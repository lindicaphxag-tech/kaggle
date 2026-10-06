from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PromptPrimitive(str, Enum):
    AABB = 'axis_aligned_box'
    OBB = 'oriented_box'
    PARALLELEPIPED = 'parallelepiped'
    MASK = 'mask'


class Exactness(str, Enum):
    CONTINUUM = 'continuum_exact'
    TARGET_GRID = 'target_grid_exact'
    REJECTED = 'rejected'


@dataclass(frozen=True)
class BoxRepresentationCertificate:
    closed_as_aabb: bool
    exact_region_volume: float
    tight_aabb_volume: float
    overcoverage_ratio: float
    tight_aabb_iou: float
    target_widths: np.ndarray


@dataclass(frozen=True)
class LiftDecision:
    target: PromptPrimitive | None
    exactness: Exactness
    authorized: bool
    complexity_level: int | None
    reason: str


@dataclass(frozen=True)
class ApproximationDebtCertificate:
    total_linear_map: np.ndarray
    direct_target_widths: np.ndarray
    eager_target_widths: np.ndarray
    exact_final_volume: float
    direct_tight_aabb_volume: float
    eager_stagewise_aabb_volume: float
    direct_prompt_iou: float
    eager_prompt_iou: float
    path_overcoverage_ratio: float
    log_volume_debt: float
    fidelity_retention: float
    avoidable_prompt_iou_loss: float
    should_defer_projection: bool

@dataclass(frozen=True)
class SampledFidelity:
    analytic_iou: float
    sampled_iou: float
    absolute_error: float
    samples_per_axis: int


def _linear(linear_map: np.ndarray) -> np.ndarray:
    linear = np.asarray(linear_map, dtype=np.float64)
    if linear.shape != (3, 3) or not np.all(np.isfinite(linear)):
        raise ValueError('linear_map must be one finite 3x3 matrix')
    return linear


def is_monomial(linear_map: np.ndarray, *, atol: float = 1e-8) -> bool:
    linear = _linear(linear_map)
    nz = np.abs(linear) > atol
    return bool(np.all(nz.sum(axis=0) == 1) and np.all(nz.sum(axis=1) == 1))


def has_orthogonal_columns(
    linear_map: np.ndarray, *, atol: float = 1e-8
) -> bool:
    linear = _linear(linear_map)
    gram = linear.T @ linear
    scale = max(1.0, float(np.max(np.abs(np.diag(gram)))))
    off = gram - np.diag(np.diag(gram))
    return bool(np.max(np.abs(off)) <= atol * scale)


def is_similarity(linear_map: np.ndarray, *, atol: float = 1e-8) -> bool:
    """Whether the entire OBB family is closed under the linear map."""
    linear = _linear(linear_map)
    gram = linear.T @ linear
    scale2 = float(np.trace(gram) / 3.0)
    if scale2 <= atol:
        return False
    return bool(
        np.allclose(
            gram,
            np.eye(3) * scale2,
            atol=atol * max(1.0, scale2),
            rtol=atol,
        )
    )


def certify_aabb_representation(
    linear_map: np.ndarray,
    source_extents: np.ndarray,
    *,
    atol: float = 1e-8,
) -> BoxRepresentationCertificate:
    """Quantify the tight-AABB representation loss after affine transport.

    For source side lengths d and source-to-target linear map A:

        V_exact = |det(A)| prod(d)
        width_i = sum_j |A_ij| d_j
        V_AABB = prod(width)
        IoU_prompt = V_exact / V_AABB.

    The quantity is a representation-fidelity certificate, not a bound on
    the final segmentation accuracy of a model.
    """
    linear = _linear(linear_map)
    extents = np.asarray(source_extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError('source_extents must be one finite length-3 vector')
    if np.any(extents <= 0):
        raise ValueError('source_extents must be strictly positive')
    determinant = float(np.linalg.det(linear))
    if abs(determinant) <= atol:
        raise ValueError('linear map must be nonsingular')

    widths = np.abs(linear) @ extents
    exact_volume = float(abs(determinant) * np.prod(extents))
    aabb_volume = float(np.prod(widths))
    ratio = max(1.0, aabb_volume / exact_volume)
    return BoxRepresentationCertificate(
        closed_as_aabb=is_monomial(linear, atol=atol),
        exact_region_volume=exact_volume,
        tight_aabb_volume=aabb_volume,
        overcoverage_ratio=ratio - 1.0,
        tight_aabb_iou=1.0 / ratio,
        target_widths=widths,
    )


def minimal_exact_lift_for_aabb(
    linear_map: np.ndarray,
    *,
    supported: set[PromptPrimitive] | frozenset[PromptPrimitive],
    atol: float = 1e-8,
) -> LiftDecision:
    """Select AABB -> OBB -> parallelepiped -> target-grid mask."""
    linear = _linear(linear_map)
    if abs(float(np.linalg.det(linear))) <= atol:
        return LiftDecision(
            None, Exactness.REJECTED, False, None,
            'singular chart maps do not preserve a 3-D box region',
        )
    supported = set(supported)
    if PromptPrimitive.AABB in supported and is_monomial(linear, atol=atol):
        return LiftDecision(
            PromptPrimitive.AABB, Exactness.CONTINUUM, True, 0,
            'monomial map preserves target-axis alignment',
        )
    if PromptPrimitive.OBB in supported and has_orthogonal_columns(linear, atol=atol):
        return LiftDecision(
            PromptPrimitive.OBB, Exactness.CONTINUUM, True, 1,
            'transported source edges remain mutually orthogonal',
        )
    if PromptPrimitive.PARALLELEPIPED in supported:
        return LiftDecision(
            PromptPrimitive.PARALLELEPIPED, Exactness.CONTINUUM, True, 2,
            'every invertible affine image of a box is a parallelepiped',
        )
    if PromptPrimitive.MASK in supported:
        return LiftDecision(
            PromptPrimitive.MASK, Exactness.TARGET_GRID, True, 3,
            'rasterize physical membership on the declared target grid',
        )
    return LiftDecision(
        None, Exactness.REJECTED, False, None,
        'no supported representation preserves the transported region',
    )


def sample_tight_aabb_iou(
    linear_map: np.ndarray,
    source_extents: np.ndarray,
    *,
    samples_per_axis: int = 65,
) -> SampledFidelity:
    """Verify the analytic certificate by deterministic target-box sampling."""
    if samples_per_axis < 3:
        raise ValueError('samples_per_axis must be at least 3')
    linear = _linear(linear_map)
    extents = np.asarray(source_extents, dtype=np.float64)
    cert = certify_aabb_representation(linear, extents)
    corners = np.array(
        [
            [x, y, z]
            for x in (0.0, extents[0])
            for y in (0.0, extents[1])
            for z in (0.0, extents[2])
        ],
        dtype=np.float64,
    )
    transformed = corners @ linear.T
    low = transformed.min(axis=0)
    high = transformed.max(axis=0)
    axes = [
        low[i]
        + (np.arange(samples_per_axis, dtype=np.float64) + 0.5)
        * (high[i] - low[i])
        / samples_per_axis
        for i in range(3)
    ]
    inverse = np.linalg.inv(linear)
    yy, zz = np.meshgrid(axes[1], axes[2], indexing='ij')
    yz = np.stack([yy.ravel(), zz.ravel()], axis=1)
    inside_count = 0
    total = 0
    for x in axes[0]:
        target = np.column_stack([np.full(len(yz), x), yz])
        source = target @ inverse.T
        inside = np.all(
            (source >= -1e-12) & (source <= extents + 1e-12), axis=1
        )
        inside_count += int(inside.sum())
        total += int(len(inside))
    sampled = inside_count / total
    return SampledFidelity(
        analytic_iou=cert.tight_aabb_iou,
        sampled_iou=float(sampled),
        absolute_error=float(abs(sampled - cert.tight_aabb_iou)),
        samples_per_axis=samples_per_axis,
    )


def certify_aabb_approximation_debt(
    linear_maps: tuple[np.ndarray, ...] | list[np.ndarray],
    source_extents: np.ndarray,
    *,
    atol: float = 1e-8,
) -> ApproximationDebtCertificate:
    """Certify loss caused by projecting to AABB after every affine stage.

    Deferred projection uses |Ak...A1| d. Eager stagewise projection
    uses |Ak|...|A1| d. The elementwise inequality |AB| <= |A||B|
    guarantees that eager projection cannot beat deferred projection.
    """
    if not linear_maps:
        raise ValueError('at least one linear map is required')
    extents = np.asarray(source_extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        raise ValueError('source_extents must be one finite length-3 vector')
    if np.any(extents <= 0):
        raise ValueError('source_extents must be strictly positive')

    total = np.eye(3, dtype=np.float64)
    eager_widths = extents.copy()
    for linear_map in linear_maps:
        linear = _linear(linear_map)
        if abs(float(np.linalg.det(linear))) <= atol:
            raise ValueError('every linear map must be nonsingular')
        total = linear @ total
        eager_widths = np.abs(linear) @ eager_widths

    direct_widths = np.abs(total) @ extents
    exact_volume = float(abs(np.linalg.det(total)) * np.prod(extents))
    direct_volume = float(np.prod(direct_widths))
    eager_volume = float(np.prod(eager_widths))
    if eager_volume + atol < direct_volume:
        raise RuntimeError('eager AABB volume violated the absolute-matrix bound')
    direct_iou = min(1.0, exact_volume / direct_volume)
    eager_iou = min(1.0, exact_volume / eager_volume)
    volume_ratio = max(1.0, eager_volume / direct_volume)
    log_volume_debt = float(np.log(volume_ratio))
    fidelity_retention = float(eager_iou / direct_iou) if direct_iou > 0 else 0.0
    return ApproximationDebtCertificate(
        total_linear_map=total,
        direct_target_widths=direct_widths,
        eager_target_widths=eager_widths,
        exact_final_volume=exact_volume,
        direct_tight_aabb_volume=direct_volume,
        eager_stagewise_aabb_volume=eager_volume,
        direct_prompt_iou=direct_iou,
        eager_prompt_iou=eager_iou,
        path_overcoverage_ratio=volume_ratio - 1.0,
        log_volume_debt=log_volume_debt,
        fidelity_retention=fidelity_retention,
        avoidable_prompt_iou_loss=max(0.0, direct_iou - eager_iou),
        should_defer_projection=bool(eager_iou + atol < direct_iou),
    )
