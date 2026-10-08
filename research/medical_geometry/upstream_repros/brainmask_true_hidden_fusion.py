"""Frozen authentic T1 U-Net INTERNAL-feature fusion: pre-head causal contrast.

The pretrained 3-D U-Net is unmodified and completely frozen. Three plane
inputs pass through the original learned encoder/decoder independently. Their
ACTUAL last-layer hidden features are either wrongly co-indexed or exactly
reoriented to the MRI acquisition voxel grid BEFORE the original pretrained
1x1x1 classifier and sigmoid. This is an experimental cross-view composition,
NOT the original trained architecture, not channel-semantic equivalence, and
NOT a clinical validation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import nibabel as nib
import numpy as np
import yaml

from .brainmask_pretrained_weakmask_eval import (
    SOURCE_IMAGE_BLOB, SOURCE_MASK_BLOB, git_blob_sha1, score,
)
from .brainmask_pretrained_plane_fusion import (
    PLANE_TO_ACQUISITION_AXES, audit_exports, UPSTREAM_SHA,
)


def orient_hidden_to_acquisition(
    hidden: np.ndarray, plane: str,
) -> np.ndarray:
    """Only transpose SPATIAL axes; channel identity stays explicitly fixed."""
    if plane not in PLANE_TO_ACQUISITION_AXES:
        raise ValueError("undocumented model plane")
    x = np.asarray(hidden)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError("expected finite [D,H,W,C] learned hidden feature")
    return np.transpose(x, PLANE_TO_ACQUISITION_AXES[plane] + (3,))


def _digest_weights(weights) -> str:
    return sha256(
        b"".join(np.ascontiguousarray(w.numpy()).tobytes() for w in weights)
    ).hexdigest()


def _source_model(params, checkpoint: Path, upstream_root: Path):
    import tensorflow as tf

    sys.path.insert(0, str(upstream_root.resolve()))
    from model.networks import Networks

    tf.keras.backend.clear_session()
    tf.random.set_seed(17)
    # This mutates upstream params once, identically to existing run.
    original = Networks(params)()
    before = [w.numpy().copy() for w in original.weights]
    original.load_weights(str(checkpoint))
    changed = sum(
        not np.array_equal(a, b.numpy())
        for a, b in zip(before, original.weights)
    )
    if changed <= 0:
        raise ValueError("source checkpoint did not populate actual network weights")
    classifier = [
        layer for layer in original.layers
        if type(layer) is tf.keras.layers.Conv3D
        and layer.filters == 1 and tuple(layer.kernel_size) == (1, 1, 1)
    ]
    if len(classifier) != 1:
        raise ValueError("ambiguous original pretrained 1x1 output head")
    head = classifier[0]
    if head is not original.layers[-2] and original.layers[-1].__class__.__name__ not in (
        "TFOpLambda", "SlicingOpLambda"
    ):
        # Source last ops vary across TF point releases, so independent
        # numerical head-vs-model equality below is the decisive oracle.
        pass
    joint = tf.keras.Model(
        inputs=original.inputs, outputs=(head.input, original.output)
    )
    return original, joint, head, changed


def run(
    source_root: Path, image_path: Path, mask_path: Path,
    config_path: Path, weights_path: Path, predictions_root: Path,
) -> dict:
    import tensorflow as tf
    for path, expected in ((image_path, SOURCE_IMAGE_BLOB), (mask_path, SOURCE_MASK_BLOB)):
        if git_blob_sha1(path) != expected:
            raise ValueError(f"non-pinned real MRI source: {path}")
    frozen = json.loads(
        (predictions_root / "native_pretrained_plane_export.json").read_text()
    )
    if frozen.get("upstream_source_commit") != UPSTREAM_SHA:
        raise ValueError("upstream code revision differs from original proof")
    checkpoint_sha = sha256(weights_path.read_bytes()).hexdigest()
    if checkpoint_sha != frozen.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA-256 differs from original actual model run")
    if sha256(image_path.read_bytes()).hexdigest() != frozen.get("source_sha256"):
        raise ValueError("MRI SHA-256 differs from original pretrained run")

    image = nib.load(str(image_path))
    supplied = nib.load(str(mask_path))
    if image.shape != supplied.shape or not np.allclose(
        image.affine, supplied.affine, atol=1e-5, rtol=0
    ):
        raise ValueError("source MRI and provided weak mask mismatch")
    crop_start = (np.array(image.shape, dtype=int) - 64) // 2
    if np.any(crop_start < 0):
        raise ValueError("original frozen 64-cube unavailable")
    crop = tuple(slice(int(i), int(i) + 64) for i in crop_start)
    reference = np.asarray(supplied.dataobj)[crop] > 0
    transform = np.eye(4)
    transform[:3, 3] = crop_start
    acquired_affine = image.affine @ transform
    exported = audit_exports(
        predictions_root, acquired_affine, (64, 64, 64), manifest=frozen
    )

    params = SimpleNamespace(**yaml.safe_load(config_path.read_text()))
    params.batch_size = 1
    params.train_dims = [64, 64, 64]
    params.infer_dims = [64, 64, 64]
    params.infer_patch_overlap = [1, 1, 1]
    params.policy = tf.keras.mixed_precision.Policy("float32")
    model, joint, classifier, changed = _source_model(
        params, weights_path, source_root
    )
    actual_parameter_hash = _digest_weights(model.weights)
    if actual_parameter_hash != frozen.get("loaded_parameter_digest"):
        raise ValueError("actual TensorFlow loaded parameters mismatch prior model")
    from utilities import input_fn_util as util

    study = predictions_root / "work" / "case_000"
    if not (study / "sample_t1.nii.gz").is_file():
        raise ValueError("missing original source-model NIfTI crop")
    scan = nib.load(str(study / "sample_t1.nii.gz"))
    if not np.allclose(scan.affine, acquired_affine, atol=1e-5, rtol=0):
        raise ValueError("source-loader patch affine differs from original MRI")

    raw_features, mapped_features, mapped_logits = [], [], []
    per_plane = {}
    hidden_channels = None
    for plane in PLANE_TO_ACQUISITION_AXES:
        source = util._load_single_study(
            str(study), ["t1"], data_format="channels_last", plane=plane,
            norm=True, norm_mode="zscore", out_res=None, out_dims=None,
        )
        if source.shape != (64, 64, 64, 1):
            raise ValueError("source TensorFlow loader returned unexpected shape")
        with tf.device("/CPU:0"):
            hidden_tf, predictions_tf = joint(
                np.ascontiguousarray(source)[None], training=False
            )
            head_output = tf.nn.sigmoid(
                classifier(hidden_tf, training=False)
            ).numpy()
        hidden = np.asarray(hidden_tf.numpy()[0], dtype=np.float32)
        predictions = np.asarray(predictions_tf.numpy()[0, ..., 0], np.float32)
        if (
            hidden.shape[:3] != (64, 64, 64)
            or hidden.shape[-1] < 2
            or not np.isfinite(hidden).all()
            or not np.isfinite(predictions).all()
        ):
            raise ValueError("true hidden features have invalid shape/content")
        if hidden_channels is None:
            hidden_channels = int(hidden.shape[-1])
        if hidden_channels != hidden.shape[-1]:
            raise ValueError("pretrained model channel count unstable across planes")
        replay_error = float(np.max(np.abs(head_output[0, ..., 0] - predictions)))
        original_error = float(np.max(
            np.abs(predictions - exported[plane]["original"])
        ))
        mapped = orient_hidden_to_acquisition(hidden, plane)
        # The pointwise FROZEN classifier must commute with the known
        # orientation permutation (modulo floating point).
        with tf.device("/CPU:0"):
            exact_mapped_logits = classifier(
                tf.convert_to_tensor(mapped[None]), training=False
            ).numpy()[0, ..., 0]
            mapped_head = tf.nn.sigmoid(exact_mapped_logits).numpy()
        mapped_error = float(np.max(
            np.abs(mapped_head - exported[plane]["patched"])
        ))
        if max(replay_error, original_error, mapped_error) > 2e-5:
            raise ValueError(
                f"{plane}: official pretrained frozen head does not reproduce "
                f"saved actual model prediction after exact plane transport"
            )
        raw_features.append(hidden)
        mapped_features.append(mapped)
        mapped_logits.append(exact_mapped_logits)
        per_plane[plane] = {
            "true_learned_hidden_shape": list(hidden.shape),
            "trained_model_to_replayed_classifier_max_abs_error": replay_error,
            "new_source_forward_to_prior_export_max_abs_error": original_error,
            "inverse_plane_feature_head_to_native_patched_export_max_abs_error": mapped_error,
            "actual_hidden_sha256": sha256(hidden.tobytes()).hexdigest(),
        }

    # Aggregation uses fixed weights. No training, weighting or ROI selection.
    wrong = np.mean(np.stack(raw_features).astype(np.float64), axis=0)
    aligned = np.mean(np.stack(mapped_features).astype(np.float64), axis=0)
    with tf.device("/CPU:0"):
        wrong_logits = classifier(
            tf.convert_to_tensor(wrong[None].astype(np.float32)), training=False
        )
        correct_logits = classifier(
            tf.convert_to_tensor(aligned[None].astype(np.float32)), training=False
        )
        wrong_probability = tf.nn.sigmoid(wrong_logits).numpy()[0, ..., 0]
        correct_probability = tf.nn.sigmoid(correct_logits).numpy()[0, ..., 0]
    # An affine pointwise classifier commutes with fixed convex averaging.
    # This is a *decisive* negative novelty control: the current pre-head
    # experiment is algebraically reducible to calibrated logit averaging,
    # even though authentic 32-channel model features are involved.
    logit_fusion = np.mean(np.stack(mapped_logits).astype(np.float64), axis=0)
    with tf.device("/CPU:0"):
        logit_fusion_probability = tf.nn.sigmoid(
            tf.convert_to_tensor(logit_fusion.astype(np.float32))
        ).numpy()
    logit_reduction_error = float(np.max(
        np.abs(correct_probability - logit_fusion_probability)
    ))
    if logit_reduction_error > 2e-5:
        raise ValueError(
            "frozen linear pretrained classifier failed logit-fusion "
            "equivalence oracle; cannot attribute any novel mechanism"
        )
    ordinary_probability_fusion = np.mean(
        np.stack([
            exported[p]["patched"] for p in PLANE_TO_ACQUISITION_AXES
        ]).astype(np.float64), axis=0
    )
    baseline_mask = ordinary_probability_fusion >= .5
    wrong_mask = wrong_probability >= .5
    correct_mask = correct_probability >= .5

    # Optional physical/semantic diagnostics are descriptive only;
    # feature-channel correspondence across differently oriented inference
    # is not implied by equal channel indices.
    ax = mapped_features[0].astype(np.float64)
    crossplane_mean_abs = {
        plane: float(np.mean(np.abs(ax - mapped_features[i])))
        for i, plane in enumerate(PLANE_TO_ACQUISITION_AXES)
        if i != 0
    }
    return {
        "study_count": 1,
        "source_upstream_commit": UPSTREAM_SHA,
        "frozen_original_model_checkpoint_sha256": checkpoint_sha,
        "reused_original_model_weights_sha256": actual_parameter_hash,
        "source_checkpoint_changes_weights": changed,
        "true_trained_internal_hidden_features": True,
        "head": "original exact pretrained Conv3D(1x1x1) followed by sigmoid",
        "hidden_channels": hidden_channels,
        "input_planes": list(PLANE_TO_ACQUISITION_AXES),
        "intervention": (
            "fixed equal weighting of authentic last-layer hidden features "
            "across three independently oriented model runs; original frozen "
            "classifier then sigmoid; no post-hoc hyperparameter selection"
        ),
        "per_plane_oracles": per_plane,
        "feature_disagreement_after_correct_orientation_mae": crossplane_mean_abs,
        "reference": "one upstream-provided nonexpert brain mask",
        "raw_index_hidden_feature_fusion": score(wrong_mask, reference, acquired_affine),
        "correct_patient_grid_hidden_feature_fusion": score(
            correct_mask, reference, acquired_affine
        ),
        "standard_inverse_export_probability_fusion": score(
            baseline_mask, reference, acquired_affine
        ),
        "prehead_feature_fusion_vs_aligned_logit_ensemble_max_abs_error": logit_reduction_error,
        "prehead_feature_fusion_is_reducible_to_aligned_logit_ensemble": True,
        "prehead_feature_mixing_independent_mechanism_claimed": False,
        "aligned_logit_ensemble": score(
            logit_fusion_probability >= .5, reference, acquired_affine
        ),
        "aligned_feature_vs_probability_fusion_mean_abs_difference": float(
            np.mean(np.abs(correct_probability.astype(np.float64) - ordinary_probability_fusion))
        ),
        "wrong_vs_correct_hidden_feature_fusion_voxel_disagreement": int(
            np.count_nonzero(wrong_mask != correct_mask)
        ),
        "all_original_weights_frozen": True,
        "clinical_ground_truth": False,
        "expert_annotated_holdout": False,
        "external_replication": False,
        "feature_channel_equivariance_proven": False,
        "trained_model_accuracy_improvement_claimed": False,
        "paper_scope": (
            "REAL learned internal feature transport before a FROZEN head "
            "for experimental cross-plane composition, not native original "
            "network fusion and not proof that channels of separately "
            "oriented feature extractions share semantics"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    for name in (
        "source-root", "image", "provided-mask", "config",
        "checkpoint", "predictions-root", "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    evidence = run(
        a.source_root, a.image, a.provided_mask,
        a.config, a.checkpoint, a.predictions_root,
    )
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "true_trained_internal_hidden_features": evidence["true_trained_internal_hidden_features"],
        "hidden_channels": evidence["hidden_channels"],
        "raw": evidence["raw_index_hidden_feature_fusion"],
        "aligned": evidence["correct_patient_grid_hidden_feature_fusion"],
        "native_probability": evidence["standard_inverse_export_probability_fusion"],
        "feature_semantics_not_proven": not evidence["feature_channel_equivariance_proven"],
        "exact_logit_reduction_error": evidence["prehead_feature_fusion_vs_aligned_logit_ensemble_max_abs_error"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
