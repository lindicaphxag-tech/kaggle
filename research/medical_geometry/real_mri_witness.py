"""Real 3-D MRI geometry and feature-grid witness (educational, not clinical).

Source image: NiBabel "Someone" structural MRI from their coordinate-system
documentation, pinned Git blob b7e917e3f891c74fd99cdf82ec579ee254a195d0.
The ROI is a *controlled marker*, NOT expert segmentation, tumor or diagnosis.
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
from scipy.ndimage import affine_transform


SOURCE_BLOB = "b7e917e3f891c74fd99cdf82ec579ee254a195d0"
SOURCE_COMMIT = "d6a0ba0056445c06407afb421749a8ed5c939c56"


def normalize(image):
    pixels = np.asarray(image, dtype=np.float32)
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise ValueError("No finite MRI intensities")
    lo, hi = np.percentile(finite, [1, 99])
    return np.asarray(np.clip((np.nan_to_num(pixels, nan=float(lo)) - lo) /
                   max(float(hi - lo), 1e-6), 0, 1), dtype=np.float32)


def dice(x, y):
    a, b = np.asarray(x, bool), np.asarray(y, bool)
    d = int(a.sum()) + int(b.sum())
    return 1.0 if not d else float(2 * np.logical_and(a, b).sum() / d)


def world(affine, index):
    return (np.asarray(affine) @ np.r_[np.asarray(index, dtype=float), 1])[:3]


def regrid_nearest(label, a_input, target_shape, a_target):
    """Voxel-center map: input i = inverse(A_input) @ A_target @ output j."""
    mat = np.linalg.solve(a_input, a_target)
    # Exact signed permutation often accumulates 1e-15 inversions at the
    # boundary, which order-0 interpolation incorrectly declares out-of-FOV.
    nearest_integer = np.rint(mat)
    mat = np.where(np.abs(mat - nearest_integer) < 1e-10, nearest_integer, mat)
    return affine_transform(
        label, mat[:3, :3], offset=mat[:3, 3], output_shape=target_shape,
        order=0, mode="constant", cval=0, prefilter=False,
    )


def controlled_roi(shape):
    """A non-clinical 3-D marker anchored in a deliberately asymmetric region."""
    if len(shape) != 3 or min(shape) < 20:
        raise ValueError("3-D image is too small for the controlled marker")
    center = np.array([round((shape[0] - 1) * 0.27),
                       round((shape[1] - 1) * 0.51),
                       round((shape[2] - 1) * 0.55)], dtype=int)
    ranges = np.ogrid[tuple(slice(0, n) for n in shape)]
    sq = sum(((v - center[axis]) / 3.0) ** 2
             for axis, v in enumerate(ranges))
    return (sq <= 1).astype(np.uint8), center


def render_geometry(volume, affine, out):
    shape = tuple(volume.shape)
    roi, anchor = controlled_roi(shape)
    flip = np.flip(roi, axis=0).copy()
    H = np.eye(4)
    H[0, 0], H[0, 3] = -1, shape[0] - 1
    corrected_affine = affine @ H

    stale_in_original = regrid_nearest(flip, affine, shape, affine)
    fixed_in_original = regrid_nearest(flip, corrected_affine, shape, affine)
    if not np.array_equal(fixed_in_original, roi):
        raise AssertionError("Certified signed flip failed exact inverse-export")
    voxel_common_transform_dice = dice(flip, flip)
    stale_patient_dice = dice(roi, stale_in_original)
    repaired_patient_dice = dice(roi, fixed_in_original)
    source_center = np.argwhere(roi).mean(0)
    wrong_center = np.argwhere(stale_in_original).mean(0)
    corrected_center = np.argwhere(fixed_in_original).mean(0)
    displacement = float(np.linalg.norm(world(affine, source_center) -
                                         world(affine, wrong_center)))
    repaired_error = float(np.linalg.norm(world(affine, source_center) -
                                          world(affine, corrected_center)))
    if displacement <= 1e-4 or stale_patient_dice >= 1:
        raise AssertionError("Corruption did not create a real patient-grid failure")

    k = int(anchor[2])
    norm = normalize(volume)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.15), constrained_layout=True)
    captions = (
        "a  Real MRI + controlled marker",
        "b  Stale affine on same MRI grid",
        "c  Certified reindex repair",
    )
    for ax, mask, title in zip(
        axes, (roi, stale_in_original, fixed_in_original), captions
    ):
        ax.imshow(norm[:, :, k].T, origin="lower", cmap="gray")
        ax.contour(roi[:, :, k].T, levels=[0.5], colors=["#13A89F"],
                   linewidths=1.7, origin="lower")
        ax.contour(mask[:, :, k].T, levels=[0.5], colors=["#DE574B"],
                   linewidths=1.0, linestyles="--", origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    axes[1].text(0.02, .02, f"Patient-grid Dice {stale_patient_dice:.2f}\n"
                   f"Centroid error {displacement:.2f} mm",
                   transform=axes[1].transAxes, color="#111111", fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
    axes[2].text(0.02, .02, f"Patient-grid Dice {repaired_patient_dice:.2f}\n"
                   f"Residual {repaired_error:.2g} mm",
                   transform=axes[2].transAxes, color="#111111", fontsize=9,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=.88))
    fig.suptitle(
        "ClinGeom | Genuine MRI, controlled ROI and computed physical-space mapping",
        fontsize=12
    )
    fig.savefig(out / "fig1_real_mri_patient_space.png", dpi=220)
    plt.close(fig)
    return dict(
        roi_type="synthetic controlled marker, not clinical ground truth",
        fixed_original_mri_reference_grid=True,
        stale_affine_corruption="flip array axis 0 but reuse source affine",
        flip_index_map=H.tolist(),
        voxel_dice_after_joint_wrong_reindex=voxel_common_transform_dice,
        stale_patient_grid_dice=stale_patient_dice,
        repaired_patient_grid_dice=repaired_patient_dice,
        original_roi_world_centroid_mm=world(affine, source_center).tolist(),
        stale_roi_world_centroid_mm=world(affine, wrong_center).tolist(),
        centroid_displacement_mm=displacement,
        repaired_residual_mm=repaired_error,
    )


def render_operators(volume, affine, out):
    import torch
    import torch.nn.functional as F
    n = 32
    if min(volume.shape) < n:
        raise ValueError("MRI too small for genuine 32^3 operator response")
    corner = tuple((s - n) // 2 for s in volume.shape)
    sl = tuple(slice(x, x + n) for x in corner)
    cropped = normalize(volume[sl])
    crop_map = np.eye(4)
    crop_map[:3, 3] = corner
    A_crop = affine @ crop_map
    x = torch.from_numpy(np.ascontiguousarray(cropped, dtype=np.float32))[None, None]
    weight = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32) / 27.0
    with torch.no_grad():
        conv = F.conv3d(x, weight, stride=4, padding=1).numpy()[0, 0]
        pool = F.avg_pool3d(x, 4, stride=4).numpy()[0, 0]
    if conv.shape != (8, 8, 8) or pool.shape != conv.shape:
        raise AssertionError("Unexpected operator response dimensions")

    # Feature lattice: conv center = (0, 0, 0), pool center = (1.5,...).
    start_conv, start_pool = np.zeros(3), np.full(3, 1.5)
    offset_world = A_crop[:3, :3] @ (start_pool - start_conv)
    phase_mm = float(np.linalg.norm(offset_world))
    if phase_mm <= 0:
        raise AssertionError("Distinct operator phases were lost")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), constrained_layout=True)
    axes[0].imshow(cropped[:, :, 16].T, cmap="gray", origin="lower")
    axes[1].imshow(conv[:, :, 4].T, cmap="gray", origin="lower")
    axes[2].imshow(pool[:, :, 4].T, cmap="gray", origin="lower")
    axes[3].imshow(cropped[:, :, 16].T, cmap="gray", origin="lower")
    ij = np.arange(1, 7)
    xx, yy = np.meshgrid(ij, ij)
    axes[3].scatter(4 * xx, 4 * yy, s=15, c="#12A79D", label="Conv centers")
    axes[3].scatter(4 * xx + 1.5, 4 * yy + 1.5, s=15,
                    marker="+", c="#DB534B", label="Pool centers")
    axes[3].legend(loc="lower right", fontsize=7)
    for ax, title in zip(axes, (
        "a  Real MRI crop (32³)", "b  Real Conv3D response (8³)",
        "c  Real AvgPool3D response (8³)", "d  Sampling lattices over MRI",
    )):
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"Equal shape, distinct model sampling centers | {phase_mm:.3f} mm",
                 fontsize=12)
    fig.savefig(out / "fig3_real_mri_operator_lattices.png", dpi=220)
    plt.close(fig)
    return dict(
        input_crop_corner=list(corner),
        input_crop_shape=[32, 32, 32],
        output_shape=[8, 8, 8],
        conv_operator="actual fixed box Conv3d(k=3,s=4,p=1), not learned",
        pool_operator="actual AvgPool3d(k=4,s=4,p=0), not learned",
        conv_center_start_in_input_voxels=start_conv.tolist(),
        pool_center_start_in_input_voxels=start_pool.tolist(),
        inter_operator_world_phase_mm=phase_mm,
        torch_version=torch.__version__,
        conv_response_sha256=hashlib.sha256(conv.tobytes()).hexdigest(),
        pool_response_sha256=hashlib.sha256(pool.tobytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = Path(args.image)
    source_bytes = path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_blob = hashlib.sha1(
        b"blob " + str(len(source_bytes)).encode("ascii") + b"\0" + source_bytes
    ).hexdigest()
    if source_blob != SOURCE_BLOB:
        raise ValueError("NIfTI does not match the pinned official NiBabel Git blob")
    im = nib.load(str(path))
    volume = np.asarray(im.dataobj, dtype=np.float32)
    affine = np.asarray(im.affine, dtype=np.float64)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("Expected finite real 3-D NIfTI data")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid NIfTI affine")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Singular NIfTI patient/world geometry")
    report = dict(
        source_description="NiBabel documentation: Someone structural brain MRI",
        source_repo="https://github.com/nipy/nibabel",
        source_commit=SOURCE_COMMIT,
        source_git_blob=source_blob,
        source_file_sha256=source_sha,
        shape=list(volume.shape),
        source_affine=affine.tolist(),
        geometry=render_geometry(volume, affine, out),
        operators=render_operators(volume, affine, out),
        clinical_claim=False,
        scope="Real MRI acquisition; controlled non-expert ROI and non-learned operator probes",
    )
    (out / "real_mri_witness.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(dict(
        source_sha256=source_sha,
        shape=report["shape"],
        stale_patient_dice=report["geometry"]["stale_patient_grid_dice"],
        repaired_patient_dice=report["geometry"]["repaired_patient_grid_dice"],
        centroid_mm=report["geometry"]["centroid_displacement_mm"],
        feature_phase_mm=report["operators"]["inter_operator_world_phase_mm"]
    ), sort_keys=True))


if __name__ == "__main__":
    main()
