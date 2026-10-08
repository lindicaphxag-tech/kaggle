"""Weak-reference audit of SAME pretrained U-Net outputs before/after NIfTI export.

The pre-existing example brain mask is a project-provided comparison target,
NOT an expert clinical gold standard. Both exporters receive the SAME
model-plane probability tensor for each orientation; no retraining, threshold
selection, or cross-plane equivariance claim is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


SOURCE_IMAGE_BLOB = "81b1e5260ee4a579d8ac490cac113df9c1cb942f"
SOURCE_MASK_BLOB = "e0dd089b3544da4352e751a24dc27a828904bad7"


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()


def dice_coefficient(prediction: np.ndarray, reference: np.ndarray) -> float:
    a, b = np.asarray(prediction, dtype=bool), np.asarray(reference, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Dice requires a common voxel grid")
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * float(np.count_nonzero(a & b)) / total


def surface_points_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    data = np.asarray(mask, dtype=bool)
    if data.ndim != 3 or not np.any(data):
        return np.empty((0, 3), dtype=np.float64)
    boundary = data & ~binary_erosion(data, border_value=0)
    ijk = np.argwhere(boundary).astype(np.float64, copy=False)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def hd95_mm(
    prediction: np.ndarray,
    reference: np.ndarray,
    affine: np.ndarray,
) -> float | None:
    """Symmetric 95th-percentile *surface* distance in patient-space mm.

    Works for rotated or sheared affines by measuring 3-D Euclidean distance
    between the actual transformed surface points (not approximate axis zooms).
    Undefined for a pair containing an empty mask: return JSON null.
    """
    a = surface_points_mm(prediction, affine)
    b = surface_points_mm(reference, affine)
    if not len(a) or not len(b):
        return None
    ab = cKDTree(b).query(a, k=1)[0]
    ba = cKDTree(a).query(b, k=1)[0]
    return float(max(np.percentile(ab, 95), np.percentile(ba, 95)))


def centroid_distance_mm(
    prediction: np.ndarray,
    reference: np.ndarray,
    affine: np.ndarray,
) -> float | None:
    a = np.argwhere(np.asarray(prediction, dtype=bool))
    b = np.argwhere(np.asarray(reference, dtype=bool))
    if len(a) == 0 or len(b) == 0:
        return None
    return float(
        np.linalg.norm((a.mean(axis=0) - b.mean(axis=0)) @ affine[:3, :3].T)
    )


def score(
    prediction: np.ndarray,
    reference: np.ndarray,
    affine: np.ndarray,
) -> dict:
    return {
        "dice_against_provided_nonexpert_mask": dice_coefficient(prediction, reference),
        "hd95_mm_against_provided_nonexpert_mask": hd95_mm(prediction, reference, affine),
        "centroid_distance_mm_against_provided_nonexpert_mask": centroid_distance_mm(
            prediction, reference, affine
        ),
        "foreground_voxels": int(np.count_nonzero(prediction)),
    }


def run(image_path: Path, mask_path: Path, predictions_root: Path) -> dict:
    expected = ((image_path, SOURCE_IMAGE_BLOB), (mask_path, SOURCE_MASK_BLOB))
    for path, sha in expected:
        if git_blob_sha1(path) != sha:
            raise ValueError(f"Source file differs from immutable upstream blob: {path}")

    image = nib.load(str(image_path))
    mask = nib.load(str(mask_path))
    if image.shape != mask.shape or not np.allclose(
        image.affine, mask.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("Provided weak reference is not in the source MRI world grid")
    if any(n < 64 for n in image.shape):
        raise ValueError("Frozen 64^3 central crop is unavailable")

    start = (np.array(image.shape, dtype=int) - 64) // 2
    crop = tuple(slice(int(i), int(i) + 64) for i in start)
    label = np.asarray(mask.dataobj)[crop] > 0
    trans = np.eye(4)
    trans[:3, 3] = start
    expected_affine = image.affine @ trans

    result = {
        "source_image_git_blob": SOURCE_IMAGE_BLOB,
        "reference_mask_git_blob": SOURCE_MASK_BLOB,
        "reference_type": "upstream-supplied nonexpert brain mask; NOT clinical ground truth",
        "true_checkpoint_model": "original pinned T1 Unet3dBneck",
        "prediction_threshold": 0.5,
        "crop_rule": "predeclared centered 64x64x64 voxels, no mask-based selection",
        "source_shape": list(image.shape),
        "crop_start_ijk": [int(n) for n in start],
        "crop_source_mask_foreground": int(np.count_nonzero(label)),
        "clinical_accuracy_claim": False,
        "planes": {},
    }
    if not np.any(label):
        raise ValueError("Provided reference has no foreground in frozen crop")

    for plane in ("ax", "cor", "sag"):
        masks = {}
        fields = {}
        for exporter in ("original", "patched"):
            directory = predictions_root / "exports" / plane / exporter
            candidates = list(directory.glob("*.nii.gz"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one pinned learned export: {directory}")
            exported = nib.load(str(candidates[0]))
            if exported.shape != (64, 64, 64) or not np.allclose(
                exported.affine, expected_affine, atol=1e-5, rtol=0
            ):
                raise ValueError("Exporter does not preserve frozen source crop geometry")
            data = np.asarray(exported.dataobj, dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError("Nonfinite pretrained probability field")
            fields[exporter] = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
            masks[exporter] = data >= 0.5

        baseline = score(masks["original"], label, expected_affine)
        fixed = score(masks["patched"], label, expected_affine)
        if plane == "ax" and not np.array_equal(masks["original"], masks["patched"]):
            raise AssertionError("Axial negative control changed after export-only fix")
        if plane != "ax" and np.array_equal(masks["original"], masks["patched"]):
            raise AssertionError("Frozen learned nonaxial export witness disappeared")

        result["planes"][plane] = {
            "original": baseline,
            "patched": fixed,
            "paired_delta_dice_patched_minus_original": (
                fixed["dice_against_provided_nonexpert_mask"]
                - baseline["dice_against_provided_nonexpert_mask"]
            ),
            "export_probability_sha256": fields,
            "same_model_plane_prediction_used_for_both_exporters": True,
        }

    result["interpretation"] = (
        "Within-plane export semantics on one frozen acquired MRI/checkpoint, "
        "using an upstream-provided nonexpert comparison mask; "
        "a Dice/HD95 change here is not validated clinical improvement."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--provided-mask", required=True, type=Path)
    parser.add_argument("--predictions-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    result = run(a.image, a.provided_mask, a.predictions_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        plane: {
            "original_dice": round(v["original"]["dice_against_provided_nonexpert_mask"], 6),
            "patched_dice": round(v["patched"]["dice_against_provided_nonexpert_mask"], 6),
            "delta_dice": round(v["paired_delta_dice_patched_minus_original"], 6),
            "original_hd95_mm": v["original"]["hd95_mm_against_provided_nonexpert_mask"],
            "patched_hd95_mm": v["patched"]["hd95_mm_against_provided_nonexpert_mask"],
        }
        for plane, v in result["planes"].items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
