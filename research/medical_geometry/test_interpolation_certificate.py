"""Falsification tests for conditional patient-space interpolation bounds."""

import numpy as np
import pytest

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.interpolation_certificate import certify_interpolation_error
from research.medical_geometry.repair_synthesis import synthesize_lattice_repair


def grid(shape, jump, start):
    return FeatureLattice(
        tuple(shape), np.asarray(jump, dtype=float),
        np.asarray(start, dtype=float), np.ones(3, dtype=float),
    )


def affine(*spacings):
    return np.diag([*spacings, 1.0]).astype(float)


def cert(source, target, plan, *, affine_mm=None, budget=1.0, hessian=None, role='feature'):
    if affine_mm is None:
        affine_mm = affine(2.0, 2.0, 2.0)
    return certify_interpolation_error(
        source, target, plan, input_voxel_to_patient_mm=affine_mm,
        maximum_allowed_error=budget,
        world_hessian_operator_norm_bound=hessian, role=role,
    )


def test_quadratic_trilinear_error_attains_exact_bound():
    source = grid((5, 5, 5), (1, 1, 1), (0, 0, 0))
    target = grid((4, 4, 4), (1, 1, 1), (0.5, 0.5, 0.5))
    plan = synthesize_lattice_repair(source, target)
    H = 0.2  # uniform patient-space Hessian operator norm, value/mm²
    proof = cert(source, target, plan, hessian=H, budget=0.3)
    assert proof.authorized
    assert proof.max_absolute_error_bound == pytest.approx(0.3)
    assert proof.max_fractional_weights == pytest.approx((0.25, 0.25, 0.25))
    assert proof.physical_cell_lengths_mm == pytest.approx((2.0, 2.0, 2.0))

    # f(p) = 1/2 H ||p||² is the sharp equality witness for this bound.
    source_samples = np.indices(source.shape, dtype=float).reshape(3, -1).T
    source_world = source_samples * 2.0
    f = (0.5 * H * np.sum(source_world**2, axis=1)).reshape(source.shape)
    errors = []
    for i in np.ndindex(target.shape):
        a = np.asarray(i, dtype=int)
        corners = [f[a[0]+u,a[1]+v,a[2]+w]
                   for u in (0,1) for v in (0,1) for w in (0,1)]
        interpolated = np.mean(corners)
        phys = 2.0 * (np.asarray(i, dtype=float) + 0.5)
        exact = 0.5 * H * float(np.dot(phys, phys))
        errors.append(abs(interpolated-exact))
    assert max(errors) == pytest.approx(proof.max_absolute_error_bound, abs=1e-12)


def test_missing_curvature_witness_fails_closed_even_if_geometry_is_valid():
    source = grid((5,5,5), (1,1,1), (0,0,0))
    target = grid((4,4,4), (1,1,1), (0.5,0.5,0.5))
    plan = synthesize_lattice_repair(source, target)
    result = cert(source,target,plan,hessian=None,budget=100)
    assert not result.authorized
    assert result.max_absolute_error_bound is None


def test_oversized_curvature_budget_fails_closed():
    source = grid((5,5,5),(1,1,1),(0,0,0))
    target = grid((4,4,4),(1,1,1),(0.5,0.5,0.5))
    plan = synthesize_lattice_repair(source,target)
    result = cert(source,target,plan,hessian=0.2,budget=0.299)
    assert not result.authorized
    assert result.max_absolute_error_bound == pytest.approx(0.3)


def test_patient_mm_anisotropy_changes_the_same_voxel_shift_budget():
    source = grid((5,5,5),(1,1,1),(0,0,0))
    target = grid((4,4,4),(1,1,1),(0.5,0.5,0.5))
    plan = synthesize_lattice_repair(source,target)
    isotropic = cert(source,target,plan,hessian=0.1,budget=1,affine_mm=affine(1,1,1))
    thick = cert(source,target,plan,hessian=0.1,budget=1,affine_mm=affine(0.7,0.7,5))
    assert thick.max_absolute_error_bound > 9 * isotropic.max_absolute_error_bound
    assert thick.physical_cell_lengths_mm == pytest.approx((0.7,0.7,5.0))


def test_exact_crop_needs_no_smoothness_or_interpolation_budget():
    source=grid((8,8,8),(2,2,2),(0,0,0))
    target=grid((6,6,6),(2,2,2),(2,2,2))
    plan=synthesize_lattice_repair(source,target)
    result=cert(source,target,plan,hessian=None,budget=0)
    assert result.authorized
    assert result.max_absolute_error_bound == 0


def test_label_nearest_cannot_claim_a_c2_feature_bound():
    source=grid((5,5,5),(1,1,1),(0,0,0))
    target=grid((4,4,4),(1,1,1),(0.5,0.5,0.5))
    plan=synthesize_lattice_repair(source,target,role='label')
    result=cert(source,target,plan,hessian=0.1,budget=100,role='label')
    assert not result.authorized


def test_bad_role_and_forged_geometry_are_not_authorized():
    source=grid((5,5,5),(1,1,1),(0,0,0))
    target=grid((4,4,4),(1,1,1),(0.5,0.5,0.5))
    plan=synthesize_lattice_repair(source,target,role='feature')
    result=cert(source,target,plan,hessian=0.1,role='label')
    assert not result.authorized


def test_arbitrary_smooth_functions_can_share_samples_but_disagree_between_them():
    # Sampling the integer lattice cannot identify an arbitrary smooth signal.
    grid_nodes = np.arange(5,dtype=float)
    zero = np.zeros_like(grid_nodes)
    unresolved = np.sin(np.pi*grid_nodes)
    np.testing.assert_allclose(zero,unresolved,atol=1e-12)
    assert abs(float(np.sin(np.pi*0.5))) == pytest.approx(1)
    # A uniformly valid curvature bound, not the samples alone, is essential.
