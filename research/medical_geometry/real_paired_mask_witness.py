"""Real paired brain MRI and existing-mask patient-space correctness witness.

Source: ecalabr/brain_mask example001, Git commit
ca789fb26b70e359b61a8d52968813dd772fb4b1.

The mask is a *provided example mask*, NOT verified manual clinical ground truth.
This assay corrupts only the representation, never the image anatomy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, binary_erosion
from scipy.spatial import cKDTree

SOURCE_SHA = "ca789fb26b70e359b61a8d52968813dd772fb4b1"
IMAGE_BLOB = "81b1e5260ee4a579d8ac490cac113df9c1cb942f"
MASK_BLOB = "e0dd089b3544da4352e751a24dc27a828904bad7"


def git_blob_id(content: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()


def authentic_file(path: Path, expected_blob: str) -> dict:
    raw = path.read_bytes()
    digest = git_blob_id(raw)
    if digest != expected_blob:
        raise ValueError("Input bytes differ from the declared upstream Git blob: " + path.name)
    return {"git_blob": digest, "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw)}


def validate_affine(a: np.ndarray) -> None:
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError("nonfinite / malformed affine")
    if not np.allclose(a[-1], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError("not a homogeneous affine")
    if abs(np.linalg.det(a[:3, :3])) < 1e-10:
        raise ValueError("singular affine")


def regrid_nearest(src: np.ndarray, a_src: np.ndarray,
                   out_shape: tuple[int, int, int],
                   a_out: np.ndarray) -> np.ndarray:
    """Map voxel centers exactly: i_src = inv(A_src) @ A_out @ i_out."""
    m = np.linalg.solve(a_src, a_out)
    k = np.rint(m)
    m = np.where(np.abs(m - k) < 1e-10, k, m)
    return affine_transform(src, m[:3, :3], offset=m[:3, 3],
                            output_shape=out_shape, order=0,
                            mode="constant", cval=0, prefilter=False)


def dsc(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    return (2 * int(np.count_nonzero(a & b)) / denom) if denom else 1.


def surface_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    # 6-neighbor face-connected erosion; convert every retained SURFACE voxel
    # CENTER to physical xyz, so shear/rotation is handled without EDT shortcuts.
    kernel = np.zeros((3, 3, 3), dtype=bool)
    kernel[1, 1, 1] = True
    for axis in range(3):
        lo = [1, 1, 1]
        hi = [1, 1, 1]
        lo[axis] = 0
        hi[axis] = 2
        kernel[tuple(lo)] = True
        kernel[tuple(hi)] = True
    idx = np.argwhere(mask & ~binary_erosion(mask, structure=kernel,
                                            border_value=0))
    if len(idx) == 0:
        raise ValueError("Mask lacks a physical boundary")
    return idx @ affine[:3, :3].T + affine[:3, 3]


def surface_metrics(mask_a: np.ndarray, mask_b: np.ndarray,
                    affine: np.ndarray) -> dict:
    a, b = surface_world(mask_a, affine), surface_world(mask_b, affine)
    da = cKDTree(b).query(a, k=1, workers=1)[0]
    db = cKDTree(a).query(b, k=1, workers=1)[0]
    distances = np.r_[da, db]
    return {
        "symmetric_hd95_center_mm": float(np.percentile(distances, 95)),
        "symmetric_mean_surface_center_mm": float(np.mean(distances)),
        "symmetric_hausdorff_center_mm": float(distances.max()),
        "boundary_center_metric": "voxel-center KDTree, 6-neighbor surface, not subvoxel mesh",
        "boundary_voxel_counts": [len(a), len(b)]
    }


def centroid_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask)
    if not len(idx):
        raise ValueError("Empty segmentation")
    return idx.mean(axis=0) @ affine[:3, :3].T + affine[:3, 3]


def draw_overlay(ax, img: np.ndarray, reference: np.ndarray,
                 candidate: np.ndarray, slice_index: int, title: str) -> None:
    sl = img[:, :, slice_index]
    lo, hi = np.percentile(sl[np.isfinite(sl)], [1, 99])
    ax.imshow(np.clip((sl.T - lo) / max(hi - lo, 1e-4), 0, 1),
              origin="lower", cmap="gray")
    for m, color, style, lw in [
        (reference, "#20D5CB", "solid", 1.5),
        (candidate, "#EA533F", "dashed", 1.1)
    ]:
        roi = m[:, :, slice_index]
        if roi.min() != roi.max():
            ax.contour(roi.T.astype(float), levels=[0.5],
                       colors=[color], linewidths=lw,
                       linestyles=style, origin="lower")
    ax.set_title(title, fontsize=9.6)
    ax.axis("off")


def run(image_path: Path, mask_path: Path, output: Path) -> dict:
    image_identity = authentic_file(image_path, IMAGE_BLOB)
    mask_identity = authentic_file(mask_path, MASK_BLOB)
    img_nii, seg_nii = nib.load(str(image_path)), nib.load(str(mask_path))
    img, seg = np.asarray(img_nii.dataobj), np.asarray(seg_nii.dataobj)
    if img.ndim != 3 or seg.ndim != 3 or img.shape != seg.shape:
        raise ValueError("paired images must have equal nonempty 3-D shape")
    a = np.asarray(img_nii.affine, np.float64)
    am = np.asarray(seg_nii.affine, np.float64)
    validate_affine(a)
    validate_affine(am)
    if not np.allclose(a, am, atol=1e-5, rtol=0):
        raise ValueError("image and example mask declare unequal patient grids")
    mask = seg > 0
    if not mask.any() or mask.all():
        raise ValueError("actual example mask is empty/full")

    # Predeclared representation corruption: reflect the FIRST spatial axis
    # without adjusting its output affine. No cherry-picking the best axis.
    H = np.eye(4)
    H[0, 0] = -1.
    H[0, 3] = img.shape[0] - 1
    flipped = np.flip(mask, axis=0).copy()
    stale = regrid_nearest(flipped, a, mask.shape, a).astype(bool)
    corrected_a = a @ H
    repaired = regrid_nearest(flipped, corrected_a, mask.shape, a).astype(bool)
    if not np.array_equal(repaired, mask):
        raise AssertionError("Signed-axis reindex failed lossless inverse")
    if not np.array_equal(stale, flipped):
        raise AssertionError("Stale-affine control unexpectedly resampled")

    # Independent NiBabel representation oracle, unrelated to SciPy resampler.
    ori = np.array([[0, -1], [1, 1], [2, 1]], dtype=float)
    native = seg_nii.as_reoriented(ori)
    nib_affine_difference = float(np.max(np.abs(native.affine - corrected_a)))
    if nib_affine_difference > 1e-6:
        raise AssertionError("Independent NiBabel affine contradicts repair")
    if not np.array_equal(np.asarray(native.dataobj)>0, flipped):
        raise AssertionError("Independent NiBabel index map contradicts repair")

    bad_dice, repaired_dice = dsc(mask, stale), dsc(mask, repaired)
    bad_surface = surface_metrics(mask, stale, a)
    repaired_surface = surface_metrics(mask, repaired, a)
    c_ref, c_bad, c_fixed = (
        centroid_world(x, a) for x in (mask, stale, repaired)
    )
    err = float(np.linalg.norm(c_bad - c_ref))
    residual = float(np.linalg.norm(c_fixed - c_ref))

    k = int(np.argmax(mask.sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.45),
                             constrained_layout=True)
    for ax, candidate, title in zip(axes, (mask, stale, repaired), (
        "(a) Real MRI + existing brain mask",
        "(b) Stale affine, unchanged MRI reference",
        "(c) Corrected affine, unchanged MRI reference"
    )):
        draw_overlay(ax, img, mask, candidate, k, title)
    axes[1].text(.02, .03, f"Dice {bad_dice:.3f}\nHD95 {bad_surface['symmetric_hd95_center_mm']:.2f} mm",
                  transform=axes[1].transAxes, fontsize=8.1,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.93))
    axes[2].text(.02, .03, f"Dice {repaired_dice:.3f}\nHD95 {repaired_surface['symmetric_hd95_center_mm']:.2f} mm",
                  transform=axes[2].transAxes, fontsize=8.1,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.93))
    fig.suptitle("ClinGeom | Real MRI + existing brain mask: fixed patient-space evaluation",
                 fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "fig_real_pair_mri_existing_mask.png"
    fig.savefig(png, dpi=220)
    plt.close(fig)

    report = {
        "input_source": "ecalabr/brain_mask example001",
        "source_commit": SOURCE_SHA,
        "real_mri_image": image_identity,
        "provided_existing_mask": mask_identity,
        "mask_evidence_status": "provided example brain mask, not independently verified expert label",
        "source_disclaimer": "source example images are already masked for anonymity; example brain-mask quality may not be correct",
        "shape": list(mask.shape),
        "image_affine": a.tolist(),
        "controlled_corruption": "fixed axis-0 array flip, stale original affine",
        "control_joint_reindex_dice": dsc(flipped, flipped),
        "stale_fixed_grid_dice": bad_dice,
        "repaired_fixed_grid_dice": repaired_dice,
        "stale_centroid_error_mm": err,
        "repaired_centroid_error_mm": residual,
        "stale_surface_mm": bad_surface,
        "repaired_surface_mm": repaired_surface,
        "independent_nibabel_affine_max_abs_difference": nib_affine_difference,
        "independent_nibabel_voxel_reindex_exact": True,
        "real_model_inference": False,
        "clinical_validation": False,
        "source_figure": png.name
    }
    (output / "real_pair_mri_mask_results.json").write_text(
        json.dumps(report, indent=2) + "\n")
    if bad_dice >= 1.0 or repaired_dice != 1.0 or residual > 1e-6:
        raise AssertionError("Real-pair test violated its frozen repair obligations")
    print(json.dumps({k: report[k] for k in (
        "shape", "stale_fixed_grid_dice", "repaired_fixed_grid_dice",
        "stale_centroid_error_mm", "stale_surface_mm",
        "independent_nibabel_affine_max_abs_difference"
    )}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.image, args.mask, args.out)


if __name__ == "__main__":
    main()
