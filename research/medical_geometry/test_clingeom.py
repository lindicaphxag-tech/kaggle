import itertools

import numpy as np

from research.medical_geometry.clingeom import (
    PatientSpaceSpec,
    all_signed_axis_reindexings,
    apply_reindex,
    FeatureLattice,
    certify_laterality_transform,
    center_matched_affine,
    certify_feature_geometry,
    certify_metadata_substitution,
    certify_reindex,
    feature_affine,
    propagate_conv_lattice,
    convert_affine_frame,
    expected_reindexed_affine,
    signed_reindex_map,
    swap_label_pairs,
    voxel_dice,
    voxel_to_world,
)


def source_spec():
    affine = np.array(
        [
            [0.0, -0.8, 0.0, 20.0],
            [1.2, 0.0, 0.0, -30.0],
            [0.0, 0.0, 2.5, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return PatientSpaceSpec((5, 6, 7), affine, frame="RAS", role="label")


def test_wrong_joint_flip_can_keep_dice_one_while_patient_space_is_wrong():
    src = PatientSpaceSpec((5, 5, 5), np.eye(4), role="label")
    gt = np.zeros(src.shape, dtype=np.uint8)
    gt[1, 2, 2] = 1
    pred = gt.copy()

    gt_wrong = apply_reindex(gt, (0, 1, 2), (True, False, False))
    pred_wrong = apply_reindex(pred, (0, 1, 2), (True, False, False))

    assert voxel_dice(gt_wrong, pred_wrong) == 1.0

    original_idx = np.argwhere(gt)[0]
    wrong_idx = np.argwhere(gt_wrong)[0]
    original_world = voxel_to_world(src.affine, original_idx)
    wrong_world = voxel_to_world(src.affine, wrong_idx)
    assert np.linalg.norm(original_world - wrong_world) == 2.0

    cert = certify_reindex(src, src.affine, (0, 1, 2), (True, False, False))
    assert cert.valid is False
    assert cert.reason == "array reindex and affine disagree"


def test_certified_flip_updates_affine_and_preserves_patient_space():
    src = source_spec()
    permutation = (2, 0, 1)
    flips = (True, False, True)
    out_affine, out_shape = expected_reindexed_affine(
        src.affine, src.shape, permutation, flips
    )
    cert = certify_reindex(src, out_affine, permutation, flips)
    assert cert.valid
    assert cert.output_shape == out_shape
    assert cert.max_matrix_error_mm <= 1e-12
    assert cert.requires_interpolation is False

    H, _ = signed_reindex_map(src.shape, permutation, flips)
    corners = itertools.product(*[(0, n - 1) for n in out_shape])
    for out_idx in corners:
        out_h = np.r_[np.asarray(out_idx, dtype=float), 1.0]
        src_idx = (H @ out_h)[:3]
        assert np.allclose(
            voxel_to_world(out_affine, out_idx),
            voxel_to_world(src.affine, src_idx),
            atol=1e-10,
        )


def test_all_48_signed_axis_reindexings_have_exact_patient_space_certificate():
    src = source_spec()
    count = 0
    for permutation, flips in all_signed_axis_reindexings():
        out_affine, out_shape = expected_reindexed_affine(
            src.affine, src.shape, permutation, flips
        )
        cert = certify_reindex(src, out_affine, permutation, flips)
        assert cert.valid
        assert cert.output_shape == out_shape
        assert cert.max_matrix_error_mm <= 1e-12
        count += 1
    assert count == 48


def test_apply_reindex_matches_homogeneous_index_map():
    src = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    permutation = (1, 2, 0)
    flips = (True, False, True)
    out = apply_reindex(src, permutation, flips)
    H, out_shape = signed_reindex_map(src.shape, permutation, flips)
    assert out.shape == out_shape

    probes = [(0, 0, 0), tuple(n - 1 for n in out_shape), (1, 2, 1)]
    for out_idx in probes:
        src_idx = np.rint((H @ np.r_[out_idx, 1.0])[:3]).astype(int)
        assert out[out_idx] == src[tuple(src_idx)]


def test_missing_output_affine_fails_closed():
    src = source_spec()
    cert = certify_reindex(src, None, (0, 1, 2), (False, False, False))
    assert cert.valid is False
    assert np.isinf(cert.max_matrix_error_mm)
    assert "cannot be established" in cert.reason


def test_ras_lps_affine_conversion_is_involutive():
    src = source_spec()
    lps = convert_affine_frame(src.affine, "RAS", "LPS")
    ras = convert_affine_frame(lps, "LPS", "RAS")
    assert np.allclose(ras, src.affine)
    p_ras = voxel_to_world(src.affine, (2, 3, 4))
    p_lps = voxel_to_world(lps, (2, 3, 4))
    assert np.allclose(p_lps, np.array([-p_ras[0], -p_ras[1], p_ras[2]]))

def test_metadata_only_small_rounding_correction_can_be_certified():
    affine = np.eye(4, dtype=np.float64)
    corrected = affine.copy()
    corrected[0, 3] += 2e-4
    cert = certify_metadata_substitution(
        (32, 32, 32), affine, corrected, tolerance_mm=1e-3
    )
    assert cert.valid
    assert cert.max_corner_displacement_mm < 1e-3
    assert cert.handedness_changed is False


def test_metadata_only_large_translation_fails_closed():
    affine = np.eye(4, dtype=np.float64)
    forged = affine.copy()
    forged[0, 3] = 99.0
    cert = certify_metadata_substitution(
        (32, 32, 32), affine, forged, tolerance_mm=1e-3
    )
    assert cert.valid is False
    assert cert.max_corner_displacement_mm == 99.0
    assert "moves voxel centers" in cert.reason


def test_metadata_only_reflection_fails_even_if_origin_matches():
    affine = np.eye(4, dtype=np.float64)
    reflected = affine.copy()
    reflected[0, 0] = -1.0
    reflected[0, 3] = 31.0
    cert = certify_metadata_substitution(
        (32, 32, 32), affine, reflected, tolerance_mm=100.0
    )
    assert cert.valid is False
    assert cert.handedness_changed is True
    assert "handedness" in cert.reason

def test_conv_feature_lattice_tracks_true_receptive_field_centers():
    lattice = FeatureLattice.input_grid((64, 64, 64))
    out = propagate_conv_lattice(
        lattice, kernel_size=3, stride=4, padding=1, dilation=1
    )
    assert out.shape == (16, 16, 16)
    assert np.allclose(out.jump, [4.0, 4.0, 4.0])
    assert np.allclose(out.start, [0.0, 0.0, 0.0])
    assert np.allclose(out.receptive_field, [3.0, 3.0, 3.0])

    A = feature_affine(np.eye(4), out)
    assert np.allclose(A[:3, :3], np.diag([4.0, 4.0, 4.0]))
    assert np.allclose(A[:3, 3], [0.0, 0.0, 0.0])


def test_shape_only_reference_affine_can_shift_conv_features_by_1p5mm():
    lattice = FeatureLattice.input_grid((64, 64, 64))
    out = propagate_conv_lattice(
        lattice, kernel_size=3, stride=4, padding=1, dilation=1
    )
    naive = center_matched_affine(np.eye(4), (64, 64, 64), out.shape)
    exact = feature_affine(np.eye(4), out)

    # Shape-only FOV matching centers a 16^3 grid at 1.5, 5.5, ... mm,
    # while this convolution samples receptive-field centers at 0, 4, ... mm.
    assert np.allclose(naive[:3, :3], np.diag([4.0, 4.0, 4.0]))
    assert np.allclose(naive[:3, 3], [1.5, 1.5, 1.5])
    assert np.allclose(exact[:3, 3], [0.0, 0.0, 0.0])

    cert = certify_feature_geometry(np.eye(4), out, naive)
    assert cert.valid is False
    assert cert.max_matrix_error_mm == 1.5


def test_shape_only_reference_affine_is_correct_for_nonoverlapping_pooling():
    lattice = FeatureLattice.input_grid((64, 64, 64))
    out = propagate_conv_lattice(
        lattice, kernel_size=4, stride=4, padding=0, dilation=1
    )
    naive = center_matched_affine(np.eye(4), (64, 64, 64), out.shape)
    exact = feature_affine(np.eye(4), out)
    assert np.allclose(out.start, [1.5, 1.5, 1.5])
    assert np.allclose(out.jump, [4.0, 4.0, 4.0])
    assert np.allclose(naive, exact)
    assert certify_feature_geometry(np.eye(4), out, naive).valid


def test_two_conv_layers_compose_geometry_not_just_shape():
    lattice = FeatureLattice.input_grid((64, 64, 64))
    one = propagate_conv_lattice(lattice, kernel_size=3, stride=2, padding=1)
    two = propagate_conv_lattice(one, kernel_size=3, stride=2, padding=1)
    assert one.shape == (32, 32, 32)
    assert two.shape == (16, 16, 16)
    assert np.allclose(two.jump, [4.0, 4.0, 4.0])
    assert np.allclose(two.start, [0.0, 0.0, 0.0])
    assert np.allclose(two.receptive_field, [7.0, 7.0, 7.0])


def test_feature_geometry_missing_affine_fails_closed():
    lattice = propagate_conv_lattice(
        FeatureLattice.input_grid((32, 32, 32)),
        kernel_size=3,
        stride=2,
        padding=1,
    )
    cert = certify_feature_geometry(np.eye(4), lattice, None)
    assert cert.valid is False
    assert np.isinf(cert.max_matrix_error_mm)
    assert "missing" in cert.reason

def test_representation_flip_does_not_change_clinical_laterality():
    cert = certify_laterality_transform(representation_only=True)
    assert cert.valid
    assert cert.physical_reflection is False
    assert cert.requires_label_swap is False


def test_physical_left_right_reflection_requires_atomic_label_swap():
    reflection = np.eye(4, dtype=np.float64)
    reflection[0, 0] = -1.0
    cert = certify_laterality_transform(
        representation_only=False, transform_world=reflection, left_right_axis=0
    )
    assert cert.valid
    assert cert.physical_reflection
    assert cert.requires_label_swap

    labels = np.array([[[0, 1, 2, 1, 2]]], dtype=np.int16)
    swapped = swap_label_pairs(labels, [(1, 2)])
    assert np.array_equal(swapped, np.array([[[0, 2, 1, 2, 1]]]))


def test_oblique_reflection_fails_closed_for_laterality():
    reflection = np.eye(4, dtype=np.float64)
    # Reflection that exchanges x/y semantics is not a pure declared LR reflection.
    reflection[:2, :2] = np.array([[0.0, -1.0], [-1.0, 0.0]])
    cert = certify_laterality_transform(
        representation_only=False, transform_world=reflection, left_right_axis=0
    )
    assert cert.valid is False
    assert cert.physical_reflection
    assert cert.requires_label_swap is False
    assert "ambiguous" in cert.reason


def test_laterality_label_pairs_must_be_disjoint():
    labels = np.array([1, 2, 3], dtype=np.int16)
    try:
        swap_label_pairs(labels, [(1, 2), (2, 3)])
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("overlapping label pairs must fail closed")
