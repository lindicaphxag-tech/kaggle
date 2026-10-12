#!/usr/bin/env python3
"""V24 source-only feasibility pilot on the frozen 50-patient clinical cohort.

All 50 patients are checked, but held outer patient #1 is NOT scored.
No target AUC, target model assessment or policy promotion is performed.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import sys
import numpy as np
from moabb.datasets import Liu2024

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import original_frozen_runner as old
from run_proper_predictive_posthoc import select_proper_source_nll

PROTOCOL = ROOT / "LIU2024_50_STROKE_EXTERNAL_PREOUTCOME_PROTOCOL_20261013.md"
SUBJECTS = tuple(range(1, 51))
EVENTS = ("left_hand", "right_hand")

def get_data():
    paradigm = old.MotorImagery(
        n_classes=2, events=list(EVENTS),
        fmin=8, fmax=30, resample=160,
        baseline=None, reject_by_annotation=False
    )
    dataset = Liu2024()
    covs, labels, groups, receipts = [], [], [], []
    for subject in SUBJECTS:
        eeg, events, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        yy = np.asarray(events)
        if eeg.ndim != 3 or eeg.shape[1] != 29:
            raise ValueError(f"Subject {subject}: expected exactly 29 pure EEG channels")
        if (len(yy) != 40 or int(np.sum(yy == EVENTS[0])) != 20
            or int(np.sum(yy == EVENTS[1])) != 20):
            raise ValueError(f"Subject {subject}: nonconforming 20/20 MI trials")
        if not np.isfinite(eeg).all():
            raise ValueError(f"Subject {subject}: nonfinite EEG")
        y = (yy == EVENTS[1]).astype(int)
        covs.append(old.covariances(eeg))
        labels.append(y)
        groups.extend([subject] * len(y))
        receipts.append({"subject": int(subject), "n_trials": int(len(y)),
                         "n_eeg_channels": int(eeg.shape[1])})
        print("SOURCE_PREFLIGHT_LOADED_SUBJECT", subject, flush=True)
    return (np.concatenate(covs), np.concatenate(labels),
            np.asarray(groups), receipts)


def main():
    if not PROTOCOL.exists():
        raise RuntimeError("Frozen clinical protocol missing")
    C, y, g, rec = get_data()
    source = g != 1
    assert len(set(g[source])) == 49 and int(np.sum(source)) == 1960
    frozen = old.source_path(C[source], y[source], g[source])
    proper, proper_path = select_proper_source_nll(
        C[source], y[source], g[source], frozen["feasible_full_lambdas"]
    )
    selector = {
        "macro_source_auc": float(frozen["lambda_source_macro"]),
        "historic_class_centred_covariance": float(frozen["lambda_covariance_full"]),
        "genuine_source_mean_predictive": float(proper),
    }
    record = {
        "status": "SOURCE_ONLY_LIU2024_50PATIENT_COHORT_OUTER1_FEASIBILITY_NO_TARGET_AUC",
        "subject_pool": list(SUBJECTS),
        "outer_held_patient": 1,
        "source_people": list(range(2,51)),
        "source_selected_lambdas": selector,
        "proper_nll_source_path_feasible_points": int(len(proper_path)),
        "original_grid_feasible_points": int(len(frozen["feasible_full_lambdas"])),
        "source_macro_auc_selected": float(
           frozen["criteria_by_lambda"][str(selector["macro_source_auc"])]["source_macro_auc"]
        ),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "all_50_subjects_retained": len(rec)==50,
        "target_person_labels_or_features_passed_to_selection": False,
        "target_predictions_or_target_auc_computed": False,
        "not_independently_registered": True,
        "denominators": rec,
    }
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output=pathlib.Path("/tmp/defaultreg-v24-liu2024/outer1_source_only.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(payload,encoding="utf-8")
    print(json.dumps({
        "status":record["status"],"lambda_by_source_criterion":selector,
        "source_auc":record["source_macro_auc_selected"],
        "source_only_sha256":hashlib.sha256(payload.encode()).hexdigest()
    },indent=2),flush=True)

if __name__=="__main__":
    main()
