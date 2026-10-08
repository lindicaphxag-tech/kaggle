"""Untrained-on-holdout external MRI pilot with authentic frozen T1 U-Net.

Frozen prospective protocol: exactly three independent public T1 subjects;
fixed original pretrained HDF5 and source-loader; each subject supplies three
64^3 image-centered off-axis regions chosen using image DIMENSIONS ONLY;
compare standard correctly oriented probabilities/logits against the
experimental nonlinear final-residual branch geometry interventions.
Patients, not overlapping crops, are the unit of evidence.
No mask-driven ROI selection, threshold tuning or post-hoc exclusions.
This is a cropped-domain-shift PILOT, not whole-brain clinical validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import nibabel as nib
import numpy as np
import yaml

from .brainmask_pretrained_native_export import BLOBS, blob
from .brainmask_pretrained_weakmask_eval import score
from .brainmask_true_hidden_fusion import (
    _digest_weights, _nonlinear_tail, _source_model,
    orient_acquisition_to_hidden, orient_hidden_to_acquisition,
)
from .brainmask_pretrained_plane_fusion import PLANE_TO_ACQUISITION_AXES


ROI_OFFSETS_VOXELS = (0, -24, 24)
ROI_EDGE = 64
VIEW_DISAGREEMENT_SCALE = 0.10
TRAINED_SPATIAL_OPERATOR_SCALE = 0.015
MAX_ALLOWED_BLEND_FACTOR = 0.5
METHODS = (
    "native_axial",
    "standard_probability_ensemble",
    "standard_logit_ensemble",
    "nonlinear_aligned_untransported",
    "nonlinear_aligned_conjugated",
    "incorrect_raw_index_control",
    "reliability_gated_fusion",
)


def roi_starts(shape: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Fixed left/right/central ROIs, no mask input or image-value peak search."""
    if len(shape) != 3 or min(shape) < ROI_EDGE or shape[0] < 112:
        raise ValueError("Predeclared 3-ROI grid unsupported; do not silently select others")
    center = (np.array(shape) - ROI_EDGE) // 2
    starts = []
    for shift in ROI_OFFSETS_VOXELS:
        start = center.copy()
        start[0] += shift
        if np.any(start < 0) or np.any(start + ROI_EDGE > shape):
            raise ValueError("Predeclared external crop falls outside source scan")
        starts.append(tuple(map(int, start)))
    return starts


def _mri_and_label(subject: dict) -> tuple:
    folder = Path(subject["directory"])
    image_path, mask_path = (folder / "image.nii.gz", folder / "mask.nii.gz")
    for path, name in ((image_path, "image"), (mask_path, "mask")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != subject[name + "_sha256"]:
            raise ValueError(f"Authenticated held-out {name} mutated: {subject['case_id']}")
    image_native = nib.load(str(image_path))
    mask_native = nib.load(str(mask_path))
    if image_native.shape != mask_native.shape or not np.allclose(
        image_native.affine, mask_native.affine, rtol=0, atol=1e-4
    ):
        raise ValueError("External source image and manually refined label physical grids differ")
    img = nib.as_closest_canonical(image_native)
    mask = nib.as_closest_canonical(mask_native)
    if img.shape != mask.shape or not np.allclose(img.affine, mask.affine, rtol=0, atol=1e-4):
        raise ValueError("Label is not physically aligned after shared RAS orientation")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-8:
        raise ValueError("Noninvertible MRI affine")
    return img, mask


def _renamed_score(pred: np.ndarray, truth: np.ndarray, affine: np.ndarray):
    binary = np.asarray(pred) >= 0.5
    original = score(binary, truth, affine)
    measured = original["hd95_mm_against_provided_nonexpert_mask"]
    if measured is None:
        # All six methods MUST have the same number of scored ROIs; dropping
        # empty predictions selectively corrupts patient-wise HD95 rankings.
        # Explicit fully finite empty-mask correction adopted after initial pilot:
        # (both empty) => zero, (only one empty) => crop's physical diameter.
        vec = np.array([
            [x, y, z] for x in (-63., 63.) for y in (-63., 63.)
            for z in (-63., 63.)
        ], np.float64)
        full_crop_world_diameter = float(np.max(
            np.linalg.norm(vec @ affine[:3, :3].T, axis=1)
        ))
        measured = (
            0.0 if not np.any(binary) and not np.any(truth)
            else full_crop_world_diameter
        )
        missing_policy = (
            "both_empty_perfect_0" if not np.any(binary) and not np.any(truth)
            else "one_empty_penalty_physical_crop_diameter"
        )
    else:
        missing_policy = "both_boundaries_present_measured_surface_hd95"
    return {
        "dice": original["dice_against_provided_nonexpert_mask"],
        "hd95_mm": measured,
        "hd95_raw_nullable_mm": original["hd95_mm_against_provided_nonexpert_mask"],
        "hd95_empty_policy": missing_policy,
        "centroid_mm": original["centroid_distance_mm_against_provided_nonexpert_mask"],
        "predicted_foreground_voxels": original["foreground_voxels"],
    }


def _source_model_and_cuts(source_root: Path, config: Path, weights: Path):
    import tensorflow as tf
    if blob(config) != BLOBS["source_config"] or blob(weights) != BLOBS["source_weights"]:
        raise ValueError("Wrong original T1 model config or checkpoint")
    sys.path.insert(0, str(source_root.resolve()))
    params = SimpleNamespace(**yaml.safe_load(config.read_text()))
    params.batch_size = 1
    params.train_dims = [ROI_EDGE] * 3
    params.infer_dims = [ROI_EDGE] * 3
    params.infer_patch_overlap = [1, 1, 1]
    params.policy = tf.keras.mixed_precision.Policy("float32")
    model, joint, head, changed = _source_model(params, weights, source_root)
    before, nonlinear_tail, true_spatial_convs, _ = _nonlinear_tail(model)
    if changed <= 0 or true_spatial_convs < 1:
        raise ValueError("No authentic loaded pretrained spatially learned convolution")
    return model, joint, head, before, nonlinear_tail


def _run_crop(
    scan: nib.Nifti1Image, subject_path: Path, start: tuple[int, int, int],
    source_util, joint, head, before_tail, nonlinear_tail,
) -> tuple[dict, dict]:
    import tensorflow as tf
    sl = tuple(slice(x, x + ROI_EDGE) for x in start)
    img_array = np.asarray(scan.dataobj, dtype=np.float32)[sl]
    if not np.isfinite(img_array).all():
        raise ValueError("External MRI contains NaN/Inf")
    T = np.eye(4)
    T[:3, 3] = start
    world_affine = scan.affine @ T
    subject_path.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(img_array, world_affine),
        str(subject_path / "patient_t1.nii.gz"),
    )
    canonical_probs, canonical_logits = [], []
    raw_early, mapped_early = [], []
    replay_oracles = {}
    for plane in ("ax", "cor", "sag"):
        x = source_util._load_single_study(
            str(subject_path), ["t1"], data_format="channels_last",
            plane=plane, norm=True, norm_mode="zscore",
            out_res=None, out_dims=None
        )
        if x.shape != (ROI_EDGE, ROI_EDGE, ROI_EDGE, 1) or not np.isfinite(x).all():
            raise ValueError("Authentic source reader failed external MRI orientation")
        with tf.device("/CPU:0"):
            h_tf, prediction_tf = joint(
                tf.convert_to_tensor(np.ascontiguousarray(x)[None]), training=False
            )
            logits_tf = head(h_tf, training=False)
            recovered = tf.math.sigmoid(logits_tf).numpy()[0, ..., 0]
            early_tf = before_tail(
                tf.convert_to_tensor(np.ascontiguousarray(x)[None]), training=False
            )
            nonlinear_recovered = nonlinear_tail(
                early_tf, training=False
            ).numpy()[0, ..., 0]
        actual = np.asarray(prediction_tf.numpy()[0, ..., 0], np.float32)
        error1 = float(np.max(np.abs(actual - recovered)))
        error2 = float(np.max(np.abs(actual - nonlinear_recovered)))
        if max(error1, error2) > 2e-5:
            raise ValueError(
                f"{plane}: original pretrained source cannot be exactly replayed "
                f"across authentic nonlinear/linear architecture cuts"
            )
        actual_logit = np.asarray(logits_tf.numpy()[0, ..., 0], np.float32)
        earlier = np.asarray(early_tf.numpy()[0], np.float32)
        canonical_probs.append(
            orient_hidden_to_acquisition(actual[..., None], plane)[..., 0]
        )
        canonical_logits.append(
            orient_hidden_to_acquisition(actual_logit[..., None], plane)[..., 0]
        )
        raw_early.append(earlier)
        mapped_early.append(orient_hidden_to_acquisition(earlier, plane))
        replay_oracles[plane] = {
            "original_prediction_vs_frozen_head_abs_max": error1,
            "original_prediction_vs_full_residual_tail_abs_max": error2,
        }

    mapped_mean = np.mean(
        np.stack(mapped_early).astype(np.float64), axis=0
    ).astype(np.float32)
    raw_mean = np.mean(
        np.stack(raw_early).astype(np.float64), axis=0
    ).astype(np.float32)
    with tf.device("/CPU:0"):
        aligned_naive = nonlinear_tail(
            tf.convert_to_tensor(mapped_mean[None]), training=False
        ).numpy()[0, ..., 0]
        raw_naive = nonlinear_tail(
            tf.convert_to_tensor(raw_mean[None]), training=False
        ).numpy()[0, ..., 0]
    rotated_outputs = []
    for plane in ("ax", "cor", "sag"):
        model_plane_input = orient_acquisition_to_hidden(mapped_mean, plane)
        with tf.device("/CPU:0"):
            native_prob = nonlinear_tail(
                tf.convert_to_tensor(np.ascontiguousarray(model_plane_input)[None]),
                training=False
            ).numpy()[0, ..., 0]
        rotated_outputs.append(
            orient_hidden_to_acquisition(native_prob[..., None], plane)[..., 0]
        )
    conjugated = np.mean(np.stack(rotated_outputs).astype(np.float64), axis=0)
    # New METHOD frozen before examining the 6 independent reserved subjects:
    # no labels, no learned weights or held-out mask performance needed.
    # Failure to orient learned spatial operators creates the measured
    # *actual original-model* operator commutator; disagreement between
    # independently oriented actual output predictions is a separate
    # representation-domain-shift warning. These jointly attenuate
    # a maximum 50% departure from the unchanged in-training axial model.
    # Not a certified safety guarantee: acceptance requires prospective
    # task results and nonzero useful fusion coverage.
    baseline_axial = canonical_probs[0].astype(np.float64)
    fprob = np.mean(np.stack(canonical_probs).astype(np.float64), axis=0)
    view_disagreement = np.maximum(
        np.abs(canonical_probs[1].astype(np.float64) - baseline_axial),
        np.abs(canonical_probs[2].astype(np.float64) - baseline_axial),
    )
    kernel_noncommutation = np.maximum(
        np.abs(rotated_outputs[1].astype(np.float64) - aligned_naive.astype(np.float64)),
        np.abs(rotated_outputs[2].astype(np.float64) - aligned_naive.astype(np.float64)),
    )
    reliability = np.exp(-np.minimum(
        50.0,
        view_disagreement / VIEW_DISAGREEMENT_SCALE
        + kernel_noncommutation / TRAINED_SPATIAL_OPERATOR_SCALE
    ))
    gated_probability = np.clip(
        baseline_axial
        + MAX_ALLOWED_BLEND_FACTOR * reliability * (fprob - baseline_axial),
        0.0, 1.0
    )
    import scipy.special
    probabilities = {
        "native_axial": canonical_probs[0],
        "standard_probability_ensemble": np.mean(
            np.stack(canonical_probs).astype(np.float64), axis=0
        ),
        "standard_logit_ensemble": scipy.special.expit(
            np.mean(np.stack(canonical_logits).astype(np.float64), axis=0)
        ),
        "nonlinear_aligned_untransported": aligned_naive,
        "nonlinear_aligned_conjugated": conjugated,
        "incorrect_raw_index_control": raw_naive,
        "reliability_gated_fusion": gated_probability,
    }
    if set(probabilities) != set(METHODS):
        raise ValueError("Missing preregistered analysis arm")
    if not all(np.isfinite(v).all() and v.shape == (64, 64, 64) for v in probabilities.values()):
        raise ValueError("Nonfinite predicted output or absent spatial coordinate")
    details = {
        "all_real_trained_model_cut_replays": replay_oracles,
        "learned_operator_conjugation_vs_untransported_prob_mae": float(
            np.mean(np.abs(conjugated - aligned_naive.astype(np.float64)))
        ),
        "models_trained_on_external_subjects": False,
        "reliability_gate_mean_weight": float(np.mean(reliability)),
        "reliability_gate_voxels_weight_above_0_2": int(np.count_nonzero(reliability > .2)),
    }
    return probabilities, {"affine": world_affine, "details": details}


def run(source_root: Path, config: Path, checkpoint: Path,
        manifest: Path, out: Path, expected_patients: int = 3) -> dict:
    import tensorflow as tf
    evidence = json.loads(manifest.read_text())
    subjects = evidence["subjects"]
    if not evidence.get("data_selection_label_blind") or len(subjects) != expected_patients:
        raise ValueError("Original prospectively selected cohort required: 3 subjects")
    if len(set(s["independent_subject"] for s in subjects)) != 3:
        raise ValueError("Reused subject in externally held-out cohort")
    model, joint, head, before, tail = _source_model_and_cuts(
        source_root, config, checkpoint
    )
    parameter_digest = _digest_weights(model.weights)
    from utilities import input_fn_util as source_util
    patients = []
    for subject in subjects:
        image, mask = _mri_and_label(subject)
        starts = roi_starts(image.shape)
        # Labels are not inspected to pick patients/ROIs. They are loaded
        # only inside this post-inference scoring step.
        mask_values = np.asarray(mask.dataobj)
        if not np.isfinite(mask_values).all():
            raise ValueError("Reference mask has invalid values")
        all_rois = []
        for roi_num, start in enumerate(starts):
            case_dir = out.parent / "external_work" / subject["case_id"] / f"r{roi_num:02d}"
            probs, diagnostics = _run_crop(
                image, case_dir, start, source_util, joint, head, before, tail
            )
            sl = tuple(slice(x, x + ROI_EDGE) for x in start)
            truth = mask_values[sl] > 0
            ground_truth_count = int(np.count_nonzero(truth))
            rows = {
                method: _renamed_score(
                    prediction, truth, diagnostics["affine"]
                )
                for method, prediction in probs.items()
            }
            all_rois.append({
                "roi_id": f"predeclared_r{roi_num:02d}",
                "crop_start_in_ras_image_voxels": start,
                "reference_foreground_voxels": ground_truth_count,
                "reference_fraction": ground_truth_count / float(ROI_EDGE ** 3),
                "metrics": rows,
                "replay_validation": diagnostics["details"],
            })
        # One subject, not three correlated crops, is the experimental unit.
        per_subject = {}
        for method in METHODS:
            d = [r["metrics"][method]["dice"] for r in all_rois]
            h = [r["metrics"][method]["hd95_mm"] for r in all_rois]
            per_subject[method] = {
                "mean_fixed_roi_dice": float(np.mean(d)),
                "mean_defined_fixed_roi_hd95_mm": float(np.mean(h)),
                "undefined_hd95_rois": sum(
                    r["metrics"][method]["hd95_raw_nullable_mm"] is None
                    for r in all_rois
                ),
            }
        patients.append({
            "case_id": subject["case_id"],
            "source_dataset": subject["dataset"],
            "image_sha256": subject["image_sha256"],
            "source_mask_sha256": subject["mask_sha256"],
            "ras_shape": list(image.shape),
            "roi_metrics": all_rois,
            "patient_unit_summary": per_subject,
        })
    primary = {}
    for method in METHODS:
        dice_values = [s["patient_unit_summary"][method]["mean_fixed_roi_dice"] for s in patients]
        hd_values = [s["patient_unit_summary"][method]["mean_defined_fixed_roi_hd95_mm"] for s in patients]
        primary[method] = {
            "subject_macro_mean_fixed_roi_dice": float(np.mean(dice_values)),
            "subject_macro_mean_hd95_mm_if_defined": (
                float(np.mean([v for v in hd_values if v is not None]))
                if any(x is not None for x in hd_values) else None
            ),
            "n_independent_patients": len(dice_values),
            "n_patients_with_defined_hd95": sum(x is not None for x in hd_values),
        }
    baseline = primary["standard_probability_ensemble"]
    difference = {
        m: {
            "macro_dice_minus_standard_probability": (
                v["subject_macro_mean_fixed_roi_dice"]
                - baseline["subject_macro_mean_fixed_roi_dice"]
            ),
            "macro_hd95_minus_standard_probability_mm": (
                v["subject_macro_mean_hd95_mm_if_defined"]
                - baseline["subject_macro_mean_hd95_mm_if_defined"]
                if v["subject_macro_mean_hd95_mm_if_defined"] is not None
                and baseline["subject_macro_mean_hd95_mm_if_defined"] is not None
                else None
            ),
        }
        for m, v in primary.items()
    }
    report = {
        "experiment": "authentic_frozen_pretrained_3d_unet_external_manual_mask_roi_pilot",
        "data_sources": ["NFBS publicly mirrored pybrook", "MedOtter/SynthStrip"],
        "dataset_revision": evidence["hf_source"]["revision"],
        "source_train_checkpoint_git_blob": BLOBS["source_weights"],
        "real_original_pretrained_parameter_digest": parameter_digest,
        "model_loaded_once_and_all_weights_frozen": True,
        "threshold_predeclared": 0.5,
        "hd95_analysis_amendment_after_initial_pilot": (
            "both masks empty: 0; exactly one empty: physical diagonal "
            "diameter of 64^3 crop; otherwise symmetric 95-percentile "
            "patient-world boundary distance; NEVER drop undefined ROI"
        ),
        "roi_offsets_axis0": list(ROI_OFFSETS_VOXELS),
        "roi_edge": ROI_EDGE,
        "image_normalization": "unchanged original TensorFlow source per-plane zscore",
        "roi_rule": "three fixed 64-cube 0,+/-24 voxel axis0 offsets from canonical RAS volume midpoint, no mask data",
        "same_original_checkpoint_all_subjects": True,
        "patient_level_unit_not_crop_level_pseudoreplication": True,
        "reference_status": "source public datasets describe masks as manually corrected/refined; no independent rater verification",
        "only_crop_pilot_not_whole_volume": True,
        "hd95_policy_originally_had_arm_specific_missingness": True,
        "initial_run_metric_change_is_posthoc_correction_not_preregistered": True,
        "training_or_hyperparameter_search_on_holdout": False,
        "claimed_to_outperform_strong_baselines": False,
        "clinical_accuracy_claim": False,
        "external_independent_reproduction": False,
        "patient_count": len(patients),
        "reserved_cohort_selection_offset": evidence["frozen_public_cohort_offset"],
        "reliability_gate_frozen_thresholds": {
            "view_disagreement_scale": VIEW_DISAGREEMENT_SCALE,
            "trained_spatial_operator_scale": TRAINED_SPATIAL_OPERATOR_SCALE,
            "max_fusion_weight": MAX_ALLOWED_BLEND_FACTOR,
        },
        "prospective_6subject_labels_not_used_to_tune_gate": True,
        "case_level": patients,
        "patient_macro_primary": primary,
        "paired_macro_differences_vs_standard_probability": difference,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("source-root", "config", "checkpoint", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=3)
    a = parser.parse_args()
    result = run(a.source_root, a.config, a.checkpoint, a.manifest, a.output, expected_patients=a.expected_patients)
    print(json.dumps({
        "subjects": result["patient_count"],
        "patient_macro_primary": result["patient_macro_primary"],
        "paired_difference": result["paired_macro_differences_vs_standard_probability"],
        "no_training": result["model_loaded_once_and_all_weights_frozen"],
        "whole_volume": not result["only_crop_pilot_not_whole_volume"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
