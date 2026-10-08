"""Conditional, physically calibrated interpolation-error certificates.

Geometric alignment is not feature-value recovery. For a scalar C² field f
defined in patient space, tensor-product linear interpolation obeys

  |f(x) - I f(x)| <= 1/2 sum_j alpha_j (1-alpha_j) M_j,

where alpha_j is the fractional source-grid coordinate and M_j is a *valid
uniform upper bound* on the absolute second directional derivative along one
source grid cell. A global patient-space Hessian operator-norm bound H gives
M_j = H ||v_j||² for physical cell vectors v_j.

This claim is conditional on an externally justified Hessian bound. It does
not infer smoothness from a few samples or certify neural-feature equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .clingeom import FeatureLattice
from .repair_synthesis import LatticeRepairPlan, RepairRole, target_to_source_index_map, verify_repair_plan


@dataclass(frozen=True)
class InterpolationErrorCertificate:
    authorized: bool
    plan_kind: str
    interpolation: str | None
    max_absolute_error_bound: float | None
    maximum_allowed_error: float
    physical_cell_lengths_mm: tuple[float, float, float]
    directional_curvature_bounds: tuple[float, float, float] | None
    max_fractional_weights: tuple[float, float, float]
    assumption: str
    reason: str


def _max_fractional_weight(source_coord: np.ndarray) -> float:
    # For a grid node exactly at an integer, alpha = 0; at non-integers the
    # interpolation shape weights are alpha and 1-alpha.
    fractional = np.mod(source_coord, 1.0)
    fractional = np.where(
        np.isclose(fractional, 1.0, atol=1e-12, rtol=0), 0.0, fractional
    )
    return float(np.max(fractional * (1.0 - fractional)))


def certify_interpolation_error(
    source: FeatureLattice,
    target: FeatureLattice,
    plan: LatticeRepairPlan,
    *,
    input_voxel_to_patient_mm: np.ndarray,
    maximum_allowed_error: float,
    world_hessian_operator_norm_bound: float | None,
    role: RepairRole = "feature",
) -> InterpolationErrorCertificate:
    """Conservatively authorize trilinear repair under an explicit C² bound.

    The Hessian norm is an external uniform bound over all source cells touched
    by the repair, in output-value units / mm². It is NOT estimated or proved
    by this function. For class labels/discontinuous signals no smoothness
    guarantee is fabricated. Exact identity and crop need no curvature.
    """
    affine = np.asarray(input_voxel_to_patient_mm, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("patient affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0):
        raise ValueError("patient affine must have standard homogeneous row")
    if np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("patient affine linear part must be invertible")
    if not np.isfinite(maximum_allowed_error) or maximum_allowed_error < 0:
        raise ValueError("maximum_allowed_error must be finite and nonnegative")
    vectors = affine[:3, :3] @ np.diag(np.asarray(source.jump, dtype=float))
    lengths = tuple(float(np.linalg.norm(vectors[:, k])) for k in range(3))

    def result(allowed, bound, curvature, weights, reason, assumption):
        return InterpolationErrorCertificate(
            authorized=bool(allowed), plan_kind=plan.kind,
            interpolation=plan.interpolation,
            max_absolute_error_bound=bound,
            maximum_allowed_error=float(maximum_allowed_error),
            physical_cell_lengths_mm=lengths,
            directional_curvature_bounds=curvature,
            max_fractional_weights=weights,
            assumption=assumption, reason=reason,
        )

    if not verify_repair_plan(source, target, plan, role=role):
        return result(False, None, None, (0.0, 0.0, 0.0),
                      "underlying geometry/semantic repair certificate is invalid",
                      "none")
    if plan.kind in {"identity", "exact_crop"}:
        return result(True, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                      "exact point-preserving reindexing/crop: no interpolation error",
                      "no smoothness assumption needed")
    if plan.kind != "resample" or plan.interpolation != "trilinear":
        return result(False, None, None, (0.0, 0.0, 0.0),
                      "nearest/discontinuous-value transport has no C2 bound",
                      "none")
    if world_hessian_operator_norm_bound is None:
        return result(False, None, None, (0.0, 0.0, 0.0),
                      "no externally justified uniform Hessian bound provided",
                      "unknown curvature; fail closed")
    hessian_bound = float(world_hessian_operator_norm_bound)
    if not np.isfinite(hessian_bound) or hessian_bound < 0:
        raise ValueError("Hessian operator-norm bound must be finite and nonnegative")

    mapping = target_to_source_index_map(source, target)
    if not np.allclose(mapping[:3, :3], np.diag(np.diag(mapping[:3, :3])), atol=1e-10, rtol=0):
        return result(False, None, None, (0.0, 0.0, 0.0),
                      "cross-axis source mapping is outside separable-grid v1 proof",
                      "unavailable")
    weights = []
    for axis, n in enumerate(target.shape):
        # Product-grid maximum factors into three independent 1D maxima.
        coords = mapping[axis, axis] * np.arange(n, dtype=float) + mapping[axis, 3]
        weights.append(_max_fractional_weight(coords))
    weights = tuple(weights)
    M = tuple(hessian_bound * length**2 for length in lengths)
    bound = 0.5 * float(sum(w * m for w, m in zip(weights, M)))
    accepted = bound <= maximum_allowed_error + 1e-12
    return result(
        accepted, bound, M, weights,
        "curvature-conditional bound within budget" if accepted else
        "curvature-conditional bound exceeds requested error budget",
        "user-supplied UNIFORM world-space Hessian operator-norm bound over sampled cells",
    )
