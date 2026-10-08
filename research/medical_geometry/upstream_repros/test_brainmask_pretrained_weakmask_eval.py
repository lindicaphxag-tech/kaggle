"""Unit-level guards for the weak-reference metric implementation."""
import numpy as np
import pytest

from research.medical_geometry.upstream_repros.brainmask_pretrained_weakmask_eval import (
    centroid_distance_mm,
    dice_coefficient,
    hd95_mm,
    score,
)


def test_exact_agreement_in_oblique_patient_space():
    a = np.zeros((7, 8, 9), dtype=bool)
    a[1:4, 2:5, 3:7] = True
    affine = np.eye(4)
    affine[:3, :3] = [[2.0, 0.5, 0.0], [0.0, 3.0, 0.3], [0.0, 0.0, 4.5]]
    metrics = score(a, a.copy(), affine)
    assert metrics["dice_against_provided_nonexpert_mask"] == 1
    assert metrics["hd95_mm_against_provided_nonexpert_mask"] == pytest.approx(0)
    assert metrics["centroid_distance_mm_against_provided_nonexpert_mask"] == pytest.approx(0)


def test_hd95_uses_real_patient_mm_not_array_spacing():
    a = np.zeros((7, 8, 9), dtype=bool)
    b = np.zeros_like(a)
    a[2, 2, 2] = True
    b[2, 3, 2] = True
    affine = np.eye(4)
    affine[:3, :3] = [[2.0, 0.5, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.5]]
    expected = np.sqrt(0.5**2 + 3**2)
    assert hd95_mm(a, b, affine) == pytest.approx(expected)
    assert centroid_distance_mm(a, b, affine) == pytest.approx(expected)
    assert dice_coefficient(a, b) == 0


def test_undefined_surface_distance_is_explicit_null():
    a = np.zeros((7, 8, 9), dtype=bool)
    b = np.zeros_like(a)
    b[3, 3, 3] = True
    metrics = score(a, b, np.eye(4))
    assert metrics["hd95_mm_against_provided_nonexpert_mask"] is None
    assert metrics["centroid_distance_mm_against_provided_nonexpert_mask"] is None
    assert metrics["dice_against_provided_nonexpert_mask"] == 0


def test_shape_mismatch_never_silently_broadcasts():
    a = np.zeros((7, 8, 9), dtype=bool)
    b = np.zeros((7, 8, 10), dtype=bool)
    with pytest.raises(ValueError, match="common voxel grid"):
        dice_coefficient(a, b)
