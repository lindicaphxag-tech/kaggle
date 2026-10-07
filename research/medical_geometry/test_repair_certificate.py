"""Adversarial certificates: a valid-looking index map must not authorize extrapolation."""

from dataclasses import replace

import numpy as np

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import (
    _make_plan,
    synthesize_lattice_repair,
    synthesize_merge_repair,
    verify_merge_bundle,
    target_to_source_index_map,
    verify_repair_plan,
)


def lattice(shape, jump, start):
    return FeatureLattice(
        tuple(shape), np.asarray(jump, dtype=np.float64),
        np.asarray(start, dtype=np.float64), np.ones(3, dtype=np.float64),
    )


def test_signed_resample_certificate_outside_source_extent_is_rejected():
    source = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    target = lattice((16, 16, 16), (4, 4, 4), (1.5, 0, 0))
    assert synthesize_lattice_repair(source, target).kind == 'refuse'
    # This is not merely hash tampering: sign the forged plan with the same
    # internal canonical certificate routine as a legitimate compiler output.
    signed = _make_plan(
        kind='resample', source=source, target=target,
        mapping=target_to_source_index_map(source, target),
        exact=False, requires_interpolation=True,
        interpolation='trilinear', crop_start=None, crop_stop=None,
        reason='claim full-grid resampling without checking physical support',
    )
    assert not verify_repair_plan(source, target, signed)


def test_certificate_digest_binds_interpolation_and_authorization_flags():
    source = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    target = lattice((15, 15, 15), (4, 4, 4), (1.5, 1.5, 1.5))
    valid = synthesize_lattice_repair(source, target, role='feature')
    assert valid.kind == 'resample'
    assert verify_repair_plan(source, target, valid, role='feature')
    assert not verify_repair_plan(source, target, replace(valid, interpolation='nearest'))
    assert not verify_repair_plan(source, target, replace(valid, requires_interpolation=False))
    assert not verify_repair_plan(source, target, replace(valid, reason='different evidence'))


def test_verifier_rejects_wrong_semantic_role_even_with_valid_digest():
    source = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    target = lattice((15, 16, 16), (4, 4, 4), (1.5, 0, 0))
    labels = synthesize_lattice_repair(source, target, role='label')
    assert verify_repair_plan(source, target, labels, role='label')
    assert not verify_repair_plan(source, target, labels, role='feature')
    assert not verify_repair_plan(source, target, labels, role='landmark')


def test_certificate_cannot_be_rebound_to_a_different_source_grid():
    source = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    target = lattice((15, 15, 15), (4, 4, 4), (1.5, 1.5, 1.5))
    plan = synthesize_lattice_repair(source, target)
    # Shape rebinding must be rejected, even if the computed affine index map
    # is unchanged by spatial extent alone.
    other_source = lattice((18, 16, 16), (4, 4, 4), (0, 0, 0))
    assert not verify_repair_plan(other_source, target, plan)


def test_valid_integer_crop_is_still_accepted():
    source = lattice((20, 20, 20), (2, 2, 2), (0, 0, 0))
    target = lattice((16, 16, 16), (2, 2, 2), (4, 2, 6))
    plan = synthesize_lattice_repair(source, target)
    assert plan.kind == 'exact_crop'
    assert verify_repair_plan(source, target, plan)

def test_merge_bundle_cannot_lie_about_repair_cost():
    left = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    right = lattice((32, 32, 32), (2, 2, 2), (0.5, 0.5, 0.5))
    bundle = synthesize_merge_repair(left, right)
    assert verify_merge_bundle(left, right, bundle)
    assert bundle.interpolation_count == 1
    assert not verify_merge_bundle(
        left, right, replace(bundle, interpolation_count=0)
    )
    assert not verify_merge_bundle(
        left, right, replace(bundle, retained_fraction_left=1.0)
    )
    assert not verify_merge_bundle(
        left, right, replace(bundle, retained_fraction_right=1.0)
    )


def test_merge_bundle_role_mismatch_fails_closed():
    left = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    right = lattice((32, 32, 32), (2, 2, 2), (0.5, 0.5, 0.5))
    bundle = synthesize_merge_repair(left, right, role='feature')
    assert verify_merge_bundle(left, right, bundle, role='feature')
    assert not verify_merge_bundle(left, right, bundle, role='label')
