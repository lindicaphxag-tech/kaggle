"""Deterministic ClinGeom counterexample benchmark.

Produces a compact JSON report showing cases where voxel-space overlap is
unchanged while patient-space meaning or model-feature geometry is wrong.
"""

from __future__ import annotations

import json

import numpy as np

from research.medical_geometry.clingeom import (
    FeatureLattice,
    apply_reindex,
    center_matched_affine,
    feature_affine,
    propagate_conv_lattice,
    voxel_dice,
    voxel_to_world,
)


def dice_lies_case() -> dict:
    shape = (65, 65, 65)
    affine = np.eye(4, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:14, 28:33, 30:35] = 1
    transformed = apply_reindex(mask, (0, 1, 2), (True, False, False))

    before = np.argwhere(mask).mean(axis=0)
    after = np.argwhere(transformed).mean(axis=0)
    patient_before = voxel_to_world(affine, before)
    patient_after_if_affine_forgotten = voxel_to_world(affine, after)
    error = float(np.linalg.norm(patient_before - patient_after_if_affine_forgotten))

    return {
        "case": "joint_array_flip_with_stale_affine",
        "voxel_dice_after_same_wrong_transform_on_prediction_and_reference": voxel_dice(
            transformed, transformed
        ),
        "patient_centroid_error_mm": error,
        "original_centroid_mm": patient_before.tolist(),
        "wrong_centroid_mm": patient_after_if_affine_forgotten.tolist(),
    }


def feature_lattice_case(spacing: tuple[float, float, float]) -> dict:
    input_affine = np.eye(4, dtype=np.float64)
    input_affine[:3, :3] = np.diag(spacing)
    lattice = propagate_conv_lattice(
        FeatureLattice.input_grid((64, 64, 64)),
        kernel_size=3,
        stride=4,
        padding=1,
    )
    exact = feature_affine(input_affine, lattice)
    shape_only = center_matched_affine(
        input_affine, (64, 64, 64), lattice.shape
    )
    offset = shape_only[:3, 3] - exact[:3, 3]
    return {
        "case": "shape_only_feature_affine_for_conv3d_k3_s4_p1",
        "input_spacing_mm": list(spacing),
        "feature_shape": list(lattice.shape),
        "model_derived_origin_mm": exact[:3, 3].tolist(),
        "shape_only_origin_mm": shape_only[:3, 3].tolist(),
        "origin_offset_vector_mm": offset.tolist(),
        "origin_offset_norm_mm": float(np.linalg.norm(offset)),
    }


def pooling_control() -> dict:
    lattice = propagate_conv_lattice(
        FeatureLattice.input_grid((64, 64, 64)),
        kernel_size=4,
        stride=4,
        padding=0,
    )
    exact = feature_affine(np.eye(4), lattice)
    shape_only = center_matched_affine(np.eye(4), (64, 64, 64), lattice.shape)
    return {
        "case": "shape_only_affine_valid_control_avgpool_k4_s4",
        "max_affine_error_mm": float(np.max(np.abs(exact - shape_only))),
    }


def main() -> None:
    report = {
        "schema_version": 1,
        "dice_lies": dice_lies_case(),
        "feature_geometry": [
            feature_lattice_case((1.0, 1.0, 1.0)),
            feature_lattice_case((0.7, 0.7, 5.0)),
            feature_lattice_case((0.8, 0.8, 3.0)),
        ],
        "negative_control": pooling_control(),
        "claim_boundary": (
            "Synthetic deterministic semantic counterexamples; not a clinical or model-performance benchmark."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
