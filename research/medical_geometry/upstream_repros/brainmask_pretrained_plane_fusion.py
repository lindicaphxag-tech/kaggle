"""Actual frozen U-Net cross-plane prediction fusion with spatial provenance.

This is an EXPORT/ENSEMBLE experiment, not training, an architecture-level
pre-mix feature repair, or clinical segmentation validation. We fuse outputs
from three separately oriented runs of a single publicly released model.
The experiment rejects wrong / ambiguous correspondence before aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .brainmask_pretrained_weakmask_eval import (
    SOURCE_IMAGE_BLOB, SOURCE_MASK_BLOB, dice_coefficient, git_blob_sha1, score,
)

UPSTREAM_SHA = "ca789fb26b70e359b61a8d52968813dd772fb4b1"
PLANE_TO_ACQUISITION_AXES = {
    "ax": (0, 1, 2),
    "cor": (0, 2, 1),
    "sag": (2, 0, 1),
}


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def audit_exports(
    predictions_root: Path,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    manifest: dict | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Read all six actual exports and enforce the declared plane contract.

    An affine by itself CANNOT establish correct spatial identity when the
    exporter attaches a stale affine after transposing voxels. The plane map
    is a declared, externally known upstream operation, not inferred from MRI.
    """
    target_affine = np.asarray(target_affine, dtype=np.float64)
    if (target_affine.shape != (4, 4)
            or not np.isfinite(target_affine).all()
            or not np.allclose(target_affine[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or abs(np.linalg.det(target_affine[:3, :3])) < 1e-12):
        raise ValueError("missing or invalid acquisition affine")
    if len(target_shape) != 3 or any(s <= 0 for s in target_shape):
        raise ValueError("invalid 3D target shape")
    if manifest is not None:
        if (manifest.get("upstream_source_commit") != UPSTREAM_SHA
                or manifest.get("true_pretrained_model_inference") is not True
                or manifest.get("actual_upstream_patch_extract_and_reconstruct") is not True
                or manifest.get("checkpoint_changed_parameter_variables", 0) <= 0
                or len(str(manifest.get("loaded_parameter_digest", ""))) != 64):
            raise ValueError("model execution provenance is absent or unverified")
    planes = {}
    for plane, axes in PLANE_TO_ACQUISITION_AXES.items():
        exports = {}
        for variant in ("original", "patched"):
            directory = predictions_root / "exports" / plane / variant
            paths = list(directory.glob("*.nii.gz"))
            if len(paths) != 1:
                raise ValueError(f"expected one exported pretrained NIfTI: {directory}")
            volume = nib.load(str(paths[0]))
            if (tuple(volume.shape) != target_shape
                    or not np.allclose(volume.affine, target_affine, atol=1e-5, rtol=0)):
                raise ValueError(f"{plane}/{variant} does not claim the acquisition grid")
            arr = np.asarray(volume.dataobj, dtype=np.float32)
            if not np.isfinite(arr).all():
                raise ValueError(f"{plane}/{variant} contains nonfinite predictions")
            if manifest is not None:
                case = manifest["cases"][plane]
                expected_digest = case["export_evidence"][variant]["saved_probability_sha256"]
                if _sha256_array(arr) != expected_digest:
                    raise ValueError(f"{plane}/{variant} no longer matches trained-model report")
                if case["reconstruction_exact_to_real_model_output"] is not True:
                    raise ValueError("native TF reconstruction was not authenticated")
            exports[variant] = arr
        restored = np.transpose(exports["original"], axes)
        if not np.array_equal(restored, exports["patched"]):
            raise ValueError(
                f"{plane}: declared inverse source orientation disagrees with native patch"
            )
        planes[plane] = {**exports, "inverse_orientation": restored}
    return planes


def fuse_and_score(
    planes: dict[str, dict[str, np.ndarray]], reference: np.ndarray,
    affine: np.ndarray,
) -> dict:
    """Hold threshold/model fixed; compare raw-index vs valid-grid fusion."""
    raw_stack = np.stack([planes[p]["original"] for p in PLANE_TO_ACQUISITION_AXES])
    repaired_stack = np.stack([
        planes[p]["inverse_orientation"] for p in PLANE_TO_ACQUISITION_AXES
    ])
    native_stack = np.stack([planes[p]["patched"] for p in PLANE_TO_ACQUISITION_AXES])
    # Float64 aggregation is identical in both paths: only spatial semantics differ.
    raw_fusion = raw_stack.astype(np.float64).mean(axis=0)
    valid_fusion = repaired_stack.astype(np.float64).mean(axis=0)
    native_fusion = native_stack.astype(np.float64).mean(axis=0)
    if not np.array_equal(valid_fusion, native_fusion):
        raise ValueError("native patched fusion and inverse-orientation fusion disagree")
    if reference.shape != raw_fusion.shape:
        raise ValueError("reference mask must share the physical acquisition grid")
    if not np.isfinite(reference).all():
        raise ValueError("reference mask is not finite")
    wrong_mask, valid_mask = raw_fusion >= .5, valid_fusion >= .5
    plane_masks = {
        p: planes[p]["patched"] >= .5 for p in PLANE_TO_ACQUISITION_AXES
    }
    plane_pairs = {
        f"{a}:{b}": float(dice_coefficient(plane_masks[a], plane_masks[b]))
        for a, b in (("ax", "cor"), ("ax", "sag"), ("cor", "sag"))
    }
    return {
        "predeclared_fusion": "unweighted mean of three authentic independently oriented outputs; no tuning",
        "raw_index_fusion_vs_provided_mask": score(wrong_mask, reference, affine),
        "provenance_repaired_fusion_vs_provided_mask": score(valid_mask, reference, affine),
        "paired_delta_dice_repaired_minus_raw": float(
            dice_coefficient(valid_mask, reference)
            - dice_coefficient(wrong_mask, reference)
        ),
        "raw_fusion_vs_repaired_fusion_dice": float(dice_coefficient(wrong_mask, valid_mask)),
        "native_patched_fusion_exactly_matches_inverse_orientation": True,
        "plane_prediction_pairwise_dice_after_reorientation": plane_pairs,
        "raw_and_repaired_probability_sha256": {
            "raw": hashlib.sha256(np.ascontiguousarray(raw_fusion).tobytes()).hexdigest(),
            "repaired": hashlib.sha256(np.ascontiguousarray(valid_fusion).tobytes()).hexdigest(),
        },
        "note": (
            "Controlled spatial export-to-ensemble error on ONE acquired MRI and ONE "
            "frozen checkpoint, NOT segmentation-accuracy improvement or expert "
            "ground-truth evaluation. Different model planes can predict differently; "
            "this experiment never assumes orientation equivariance."
        ),
    }


def run(image_path: Path, mask_path: Path, root: Path) -> dict:
    for path, expected in ((image_path, SOURCE_IMAGE_BLOB), (mask_path, SOURCE_MASK_BLOB)):
        if git_blob_sha1(path) != expected:
            raise ValueError(f"pinned MRI/mask source identity mismatch: {path}")
    info = json.loads((root / "native_pretrained_plane_export.json").read_text())
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != info.get("source_sha256"):
        raise ValueError("report/image SHA-256 mismatch")
    image = nib.load(str(image_path))
    reference = nib.load(str(mask_path))
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("acquired MRI and provided mask disagree in patient grid")
    start = (np.array(image.shape) - 64) // 2
    if np.any(start < 0):
        raise ValueError("pinned central crop unavailable")
    crop = tuple(slice(int(k), int(k) + 64) for k in start)
    transform = np.eye(4)
    transform[:3, 3] = start
    affine = image.affine @ transform
    planes = audit_exports(root, affine, (64, 64, 64), manifest=info)
    target = np.asarray(reference.dataobj)[crop]
    outcome = fuse_and_score(planes, target > 0, affine)
    return {
        "upstream_source_commit": UPSTREAM_SHA,
        "actual_frozen_pretrained_model": True,
        "model_loaded_parameter_sha256": info["loaded_parameter_digest"],
        "patient_frame_identity": "source NIfTI affine; no DICOM FrameOfReferenceUID inferred",
        "image_git_blob": SOURCE_IMAGE_BLOB,
        "provided_nonexpert_mask_git_blob": SOURCE_MASK_BLOB,
        "patient_count": 1,
        "checkpoint_count": 1,
        "threshold": .5,
        "plane_order": list(PLANE_TO_ACQUISITION_AXES),
        "clinical_ground_truth": False,
        "independent_third_party_replication": False,
        "claims_task_level_generalization": False,
        "ensemble": outcome,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    for key in ("image", "provided-mask", "predictions-root", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    report = run(args.image, args.provided_mask, args.predictions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "raw_vs_weak_mask": report["ensemble"]["raw_index_fusion_vs_provided_mask"],
        "repaired_vs_weak_mask": report["ensemble"]["provenance_repaired_fusion_vs_provided_mask"],
        "paired_delta_dice": report["ensemble"]["paired_delta_dice_repaired_minus_raw"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
