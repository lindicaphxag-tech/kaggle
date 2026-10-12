#!/usr/bin/env python3
"""One-patient raw-format preflight ONLY; no source or target outcomes computed."""
from __future__ import annotations
import hashlib
import json
import pathlib
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery

PROTOCOL = pathlib.Path(__file__).with_name(
    "LIU2024_50_STROKE_EXTERNAL_PREOUTCOME_PROTOCOL_20261013.md"
)

def preflight():
    if not PROTOCOL.exists():
        raise RuntimeError("Original pre-outcome protocol missing")
    pp = PROTOCOL.read_bytes()
    paradigm = MotorImagery(
        n_classes=2, events=["left_hand", "right_hand"], fmin=8, fmax=30,
        resample=160, baseline=None, reject_by_annotation=False
    )
    X, labels, metadata = paradigm.get_data(
        dataset=Liu2024(), subjects=[1]
    )
    y = np.asarray(labels)
    if set(y.tolist()) != {"left_hand", "right_hand"}:
        raise RuntimeError("Unexpected label set; abort without performance scores")
    if X.ndim != 3 or X.shape[1] != 29:
        raise RuntimeError("Expected 29 pure EEG channels; EOG exclusion failed")
    if len(y) != 40 or np.sum(y == "left_hand") != 20 or np.sum(y == "right_hand") != 20:
        raise RuntimeError("Original 40-trial participant contract failed")
    if not np.isfinite(X).all():
        raise RuntimeError("Nonfinite EEG samples")
    return {
        "status": "LIU2024_ONLY_ONE_PATIENT_STRUCTURAL_PREFLIGHT_NO_AUC",
        "dataset": "Liu2024",
        "patient_id": 1,
        "trials": int(len(y)),
        "channels_EEG": int(X.shape[1]),
        "samples_per_trial": int(X.shape[2]),
        "class_counts": {"left_hand": int(np.sum(y == "left_hand")),
                         "right_hand": int(np.sum(y == "right_hand"))},
        "protocol_sha256": hashlib.sha256(pp).hexdigest(),
        "model_training_completed": False,
        "source_selection_performed": False,
        "any_target_AUC_computed": False,
        "clinical_claims": False,
    }

if __name__ == "__main__":
    r = preflight()
    out = pathlib.Path("/tmp/defaultreg-v24-liu2024/preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2), flush=True)
