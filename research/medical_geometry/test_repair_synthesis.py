import numpy as np

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.repair_synthesis import (
    synthesize_lattice_repair,
    synthesize_merge_repair,
    verify_merge_bundle,
    target_to_source_index_map,
    verify_repair_plan,
)


def lattice(shape, jump, start):
    return FeatureLattice(
        tuple(shape),
        np.asarray(jump, dtype=np.float64),
        np.asarray(start, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )


def test_identity_repair_is_exact_and_verified():
    src = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    plan = synthesize_lattice_repair(src, src)
    assert plan.kind == "identity"
    assert plan.exact
    assert not plan.requires_interpolation
    assert verify_repair_plan(src, src, plan)


def test_integer_subset_uses_zero_interpolation_crop():
    src = lattice((20, 20, 20), (2, 2, 2), (0, 0, 0))
    dst = lattice((16, 16, 16), (2, 2, 2), (4, 2, 6))
    plan = synthesize_lattice_repair(src, dst)
    assert plan.kind == "exact_crop"
    assert plan.crop_start == (2, 1, 3)
    assert plan.crop_stop == (18, 17, 19)
    assert plan.exact
    assert plan.interpolation is None
    assert verify_repair_plan(src, dst, plan)


def test_half_voxel_offset_requires_resampling():
    src = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    dst = lattice((16, 16, 16), (4, 4, 4), (1.5, 1.5, 1.5))
    plan = synthesize_lattice_repair(src, dst, role="feature")
    assert plan.kind == "resample"
    assert plan.interpolation == "trilinear"
    assert plan.requires_interpolation
    assert not plan.exact
    assert verify_repair_plan(src, dst, plan)


def test_label_resampling_is_nearest_not_trilinear():
    src = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    dst = lattice((16, 16, 16), (4, 4, 4), (1.5, 0, 0))
    plan = synthesize_lattice_repair(src, dst, role="label")
    assert plan.kind == "resample"
    assert plan.interpolation == "nearest"
    assert verify_repair_plan(src, dst, plan)


def test_landmark_interpolation_fails_closed():
    src = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    dst = lattice((16, 16, 16), (4, 4, 4), (0.5, 0, 0))
    plan = synthesize_lattice_repair(src, dst, role="landmark")
    assert plan.kind == "refuse"
    assert plan.interpolation is None
    assert verify_repair_plan(src, dst, plan)


def test_different_sampling_rates_produce_fractional_index_map():
    src = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    dst = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    M = target_to_source_index_map(src, dst)
    assert np.allclose(np.diag(M)[:3], [2, 2, 2])
    plan = synthesize_lattice_repair(src, dst)
    assert plan.kind == "resample"
    assert verify_repair_plan(src, dst, plan)


def test_certificate_digest_changes_with_semantically_relevant_target():
    src = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    a = lattice((16, 16, 16), (4, 4, 4), (1.5, 0, 0))
    b = lattice((16, 16, 16), (4, 4, 4), (1.0, 0, 0))
    pa = synthesize_lattice_repair(src, a)
    pb = synthesize_lattice_repair(src, b)
    assert pa.certificate_digest != pb.certificate_digest


def test_same_shape_different_phase_gets_atomic_common_lattice():
    # Conv3d(k=3,s=2,p=1): centers 0,2,...,62
    conv = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    # Pool3d(k=2,s=2,p=0): centers 0.5,2.5,...,62.5
    pool = lattice((32, 32, 32), (2, 2, 2), (0.5, 0.5, 0.5))

    bundle = synthesize_merge_repair(conv, pool)
    assert bundle.target.shape == (31, 31, 31)
    assert np.allclose(bundle.target.start, [2.0, 2.0, 2.0])
    assert bundle.left.kind == "exact_crop"
    assert bundle.left.crop_start == (1, 1, 1)
    assert bundle.right.kind == "resample"
    assert bundle.right.interpolation == "trilinear"
    assert bundle.interpolation_count == 1
    assert verify_merge_bundle(conv, pool, bundle)


def test_merge_bundle_digest_is_order_sensitive_and_geometry_bound():
    a = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    b = lattice((32, 32, 32), (2, 2, 2), (0.5, 0.5, 0.5))
    ab = synthesize_merge_repair(a, b)
    ba = synthesize_merge_repair(b, a)
    assert ab.bundle_digest != ba.bundle_digest
    assert verify_merge_bundle(a, b, ab)
    assert verify_merge_bundle(b, a, ba)


def test_unequal_sampling_rates_fail_closed_in_merge_v1():
    fine = lattice((32, 32, 32), (2, 2, 2), (0, 0, 0))
    coarse = lattice((16, 16, 16), (4, 4, 4), (0, 0, 0))
    bundle = synthesize_merge_repair(fine, coarse)
    assert bundle.right.kind == "refuse"
    assert "unequal" in bundle.reason
