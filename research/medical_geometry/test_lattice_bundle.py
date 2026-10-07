import numpy as np
import pytest

from research.medical_geometry.clingeom import FeatureLattice
from research.medical_geometry.lattice_bundle import (
    ChannelFiber,
    LatticeBundle,
    all_bundle_supports,
    bundle_from_lattice,
    bundle_summary,
    certify_minimal_point_affine_partition,
    certify_single_lattice,
    dense_conv_mix_bundle,
    factorized_reduce_bundle,
    shift_lattice_origin,
)


def anisotropic_affine():
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.diag([0.7, 0.7, 5.0])
    return affine


def test_homogeneous_feature_tensor_has_one_global_lattice():
    source = bundle_from_lattice(32, FeatureLattice.input_grid((32, 32, 32)))
    cert = certify_single_lattice(source, anisotropic_affine())
    assert cert.valid
    assert cert.unique_center_lattices == 1
    assert cert.max_intra_tensor_displacement_mm == 0.0
    assert cert.single_affine_minimax_lower_bound_mm == 0.0
    assert cert.representative is not None


def test_factorized_reduce_creates_two_channel_sampling_phases():
    source = bundle_from_lattice(32, FeatureLattice.input_grid((32, 32, 32)))
    reduced = factorized_reduce_bundle(source, out_channels=64)

    assert reduced.shape == (16, 16, 16)
    assert reduced.channels == 64
    assert len(reduced.fibers) == 2
    assert (reduced.fibers[0].channel_start, reduced.fibers[0].channel_stop) == (0, 32)
    assert (reduced.fibers[1].channel_start, reduced.fibers[1].channel_stop) == (32, 64)

    phase0 = reduced.fibers[0].supports[0]
    phase1 = reduced.fibers[1].supports[0]
    assert np.allclose(phase0.jump, [2.0, 2.0, 2.0])
    assert np.allclose(phase1.jump, [2.0, 2.0, 2.0])
    assert np.allclose(phase0.start, [0.0, 0.0, 0.0])
    assert np.allclose(phase1.start, [1.0, 1.0, 1.0])


def test_dints_factorized_reduce_has_no_single_patient_space_affine():
    source = bundle_from_lattice(32, FeatureLattice.input_grid((32, 32, 32)))
    reduced = factorized_reduce_bundle(source, out_channels=64)
    cert = certify_single_lattice(reduced, anisotropic_affine())

    expected_mm = float(np.linalg.norm([0.7, 0.7, 5.0]))
    assert cert.valid is False
    assert cert.unique_center_lattices == 2
    assert cert.max_intra_tensor_displacement_mm == pytest.approx(expected_mm)
    assert cert.max_intra_tensor_displacement_mm == pytest.approx(5.0970579749498635)
    assert cert.single_affine_minimax_lower_bound_mm == pytest.approx(2.5485289874749318)
    assert cert.representative is None


def test_dense_conv_mixes_the_two_phases_into_every_output_channel():
    source = bundle_from_lattice(32, FeatureLattice.input_grid((32, 32, 32)))
    reduced = factorized_reduce_bundle(source, out_channels=64)
    mixed = dense_conv_mix_bundle(
        reduced, out_channels=64, kernel_size=3, stride=1, padding=1
    )

    assert mixed.shape == (16, 16, 16)
    assert len(mixed.fibers) == 1
    assert mixed.fibers[0].channel_start == 0
    assert mixed.fibers[0].channel_stop == 64
    assert len(mixed.fibers[0].supports) == 2

    supports = all_bundle_supports(mixed)
    assert len(supports) == 2
    assert np.allclose(supports[0].start, [0.0, 0.0, 0.0])
    assert np.allclose(supports[1].start, [1.0, 1.0, 1.0])
    cert = certify_single_lattice(mixed, anisotropic_affine())
    assert cert.valid is False
    assert cert.max_intra_tensor_displacement_mm == pytest.approx(
        np.linalg.norm([0.7, 0.7, 5.0])
    )


def test_pointwise_channel_mix_does_not_erase_spatial_phase_history():
    source = bundle_from_lattice(16, FeatureLattice.input_grid((24, 24, 24)))
    reduced = factorized_reduce_bundle(source, out_channels=32)
    pointwise = dense_conv_mix_bundle(
        reduced, out_channels=10, kernel_size=1, stride=1, padding=0
    )
    assert len(pointwise.fibers) == 1
    assert len(pointwise.fibers[0].supports) == 2
    assert certify_single_lattice(pointwise, np.eye(4)).valid is False


def test_factorized_reduce_rejects_odd_spatial_sizes_like_monai_contract():
    source = bundle_from_lattice(8, FeatureLattice.input_grid((31, 32, 32)))
    with pytest.raises(ValueError, match="even spatial dimensions"):
        factorized_reduce_bundle(source, out_channels=16)


def test_bundle_summary_exposes_channel_fiber_provenance():
    source = bundle_from_lattice(8, FeatureLattice.input_grid((16, 16, 16)))
    reduced = factorized_reduce_bundle(source, out_channels=12)
    summary = bundle_summary(reduced)
    assert summary["shape"] == [8, 8, 8]
    assert summary["channels"] == 12
    assert summary["fibers"][0]["channel_range"] == [0, 6]
    assert summary["fibers"][1]["channel_range"] == [6, 12]
    assert "phase=(0,0,0)" in summary["fibers"][0]["provenance"]
    assert "phase=(1,1,1)" in summary["fibers"][1]["provenance"]


def test_minimal_point_affine_partition_is_exact_for_homogeneous_tensor():
    source = bundle_from_lattice(16, FeatureLattice.input_grid((16, 16, 16)))
    cert = certify_minimal_point_affine_partition(source, anisotropic_affine())
    assert cert.point_affine_representable
    assert cert.distinct_patient_lattice_types == 1
    assert cert.minimum_contiguous_fibers == 1
    assert cert.globally_single_affine
    assert cert.fiber_type_ids == (0,)


def test_factorized_reduce_requires_exactly_two_patient_lattice_types():
    source = bundle_from_lattice(16, FeatureLattice.input_grid((24, 24, 24)))
    reduced = factorized_reduce_bundle(source, out_channels=32)
    cert = certify_minimal_point_affine_partition(reduced, anisotropic_affine())
    assert cert.point_affine_representable
    assert cert.distinct_patient_lattice_types == 2
    assert cert.minimum_contiguous_fibers == 2
    assert not cert.globally_single_affine
    assert cert.fiber_type_ids == (0, 1)


def test_distinct_lattice_type_count_differs_from_contiguous_fiber_minimum():
    a = FeatureLattice.input_grid((12, 12, 12))
    b = shift_lattice_origin(a, (1, 0, 0))
    bundle = LatticeBundle(
        a.shape,
        6,
        (
            ChannelFiber(0, 2, (a,), 'A-left'),
            ChannelFiber(2, 4, (b,), 'B-middle'),
            ChannelFiber(4, 6, (a,), 'A-right'),
        ),
    )
    cert = certify_minimal_point_affine_partition(bundle, anisotropic_affine())
    assert cert.point_affine_representable
    assert cert.distinct_patient_lattice_types == 2
    assert cert.minimum_contiguous_fibers == 3
    assert cert.fiber_type_ids == (0, 1, 0)


def test_dense_channel_mix_has_no_exact_point_affine_partition():
    source = bundle_from_lattice(16, FeatureLattice.input_grid((24, 24, 24)))
    reduced = factorized_reduce_bundle(source, out_channels=32)
    mixed = dense_conv_mix_bundle(
        reduced, out_channels=8, kernel_size=1, stride=1, padding=0
    )
    cert = certify_minimal_point_affine_partition(mixed, anisotropic_affine())
    assert not cert.point_affine_representable
    assert cert.distinct_patient_lattice_types is None
    assert cert.minimum_contiguous_fibers is None
    assert not cert.globally_single_affine
    assert cert.fiber_type_ids == ()
