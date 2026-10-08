"""Adversarial plane provenance tests; no neural inference or gold-standard claims."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from research.medical_geometry.upstream_repros.brainmask_pretrained_plane_fusion import (
    PLANE_TO_ACQUISITION_AXES,
    audit_exports,
    fuse_and_score,
)


def _fixture(root: Path):
    shape = (6, 6, 6)  # Equal dimensions deliberately conceal axis errors.
    grid = np.indices(shape).astype(np.float32)
    affine = np.eye(4)
    affine[:3, :3] = [[1.7, 0.2, 0.0], [0.0, 2.3, 0.4], [0.1, 0.0, 3.1]]
    affine[:3, 3] = [17., -4., 8.]
    native = {}
    for index, plane in enumerate(PLANE_TO_ACQUISITION_AXES):
        values = np.where(
            (grid[0] + grid[1] * 2 + index) % 6 < 3, 0.9, 0.1
        ).astype(np.float32)
        native[plane] = values
        axes = PLANE_TO_ACQUISITION_AXES[plane]
        inverse_axes = tuple(int(k) for k in np.argsort(axes))
        original = np.transpose(values, inverse_axes)
        for variant, array in (("original", original), ("patched", values)):
            dest = root / "exports" / plane / variant
            dest.mkdir(parents=True)
            nib.save(nib.Nifti1Image(array, affine), str(dest / "case.nii.gz"))
    return affine, native


def test_real_orientation_map_reproduces_exact_native_patch_and_fusion(tmp_path):
    affine, native = _fixture(tmp_path)
    planes = audit_exports(tmp_path, affine, (6, 6, 6))
    for p in PLANE_TO_ACQUISITION_AXES:
        np.testing.assert_array_equal(planes[p]["inverse_orientation"], native[p])
    reference = native["ax"] > .5
    report = fuse_and_score(planes, reference, affine)
    assert report["native_patched_fusion_exactly_matches_inverse_orientation"]
    assert report["raw_fusion_vs_repaired_fusion_dice"] < 1
    assert len(report["plane_prediction_pairwise_dice_after_reorientation"]) == 3
    assert "NOT segmentation-accuracy improvement" in report["note"]
    # Scores may improve or degrade depending on learned predictions.
    assert -1 <= report["paired_delta_dice_repaired_minus_raw"] <= 1


def test_swapped_plane_contract_rejected_even_if_shapes_and_affines_match(tmp_path):
    affine, _ = _fixture(tmp_path)
    bad = tmp_path / "exports" / "cor" / "patched" / "case.nii.gz"
    volume = nib.load(str(bad))
    data = np.asarray(volume.dataobj).copy()
    nib.save(nib.Nifti1Image(data.swapaxes(0, 1), affine), str(bad))
    with pytest.raises(ValueError, match="inverse source orientation"):
        audit_exports(tmp_path, affine, (6, 6, 6))


def test_affine_mutation_and_missing_export_fail_closed(tmp_path):
    affine, _ = _fixture(tmp_path)
    original = tmp_path / "exports" / "sag" / "original" / "case.nii.gz"
    volume = nib.load(str(original))
    shifted = affine.copy()
    shifted[0, 3] += 9
    nib.save(nib.Nifti1Image(np.asarray(volume.dataobj), shifted), str(original))
    with pytest.raises(ValueError, match="acquisition grid"):
        audit_exports(tmp_path, affine, (6, 6, 6))
    original.unlink()
    with pytest.raises(ValueError, match="one exported pretrained NIfTI"):
        audit_exports(tmp_path, affine, (6, 6, 6))


def test_injected_nonfinite_probability_refused(tmp_path):
    affine, _ = _fixture(tmp_path)
    file = tmp_path / "exports" / "ax" / "original" / "case.nii.gz"
    old = nib.load(str(file))
    data = np.asarray(old.dataobj).copy()
    data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, affine), str(file))
    with pytest.raises(ValueError, match="nonfinite"):
        audit_exports(tmp_path, affine, (6, 6, 6))


def test_missing_frozen_model_provenance_refused(tmp_path):
    affine, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="model execution provenance"):
        audit_exports(tmp_path, affine, (6, 6, 6), manifest={
            "upstream_source_commit": "wrong", "true_pretrained_model_inference": True
        })


def test_reference_grid_mismatch_refused(tmp_path):
    affine, _ = _fixture(tmp_path)
    planes = audit_exports(tmp_path, affine, (6, 6, 6))
    with pytest.raises(ValueError, match="reference mask"):
        fuse_and_score(planes, np.zeros((4, 4, 4), dtype=bool), affine)
