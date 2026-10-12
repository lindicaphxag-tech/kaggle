#!/usr/bin/env python3
"""V27 original MOABB Liu2024 entire-cohort per-epoch overlap integrity scan.

No classifier fit, AUC, target labels used as outcomes, or model tuning.
Signals are only read to calculate comparisons; raw arrays are NOT written.
This is post-outcome global EEG-content access, not strict target-isolated Z0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery
from exact_epoch_identity_v27 import audit_exact_epoch_overlap

EVENTS=("left_hand", "right_hand")
SUBJECTS=tuple(range(1,51))
PROTOCOL=Path(__file__).resolve().parents[1]/"V27_EXACT_EPOCH_IDENTITY_PRE_OUTCOME_PROTOCOL.md"


def run():
    if not PROTOCOL.is_file():
        raise RuntimeError("missing precommitted retrospective V27 protocol")
    m=MotorImagery(
        n_classes=2, events=list(EVENTS), fmin=8, fmax=30,
        resample=160, baseline=None, reject_by_annotation=False)
    ds=Liu2024()
    records=[]
    for sid in SUBJECTS:
        X, original_labels, meta = m.get_data(dataset=ds, subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        original_labels=np.asarray(original_labels)
        y=(original_labels==EVENTS[1]).astype(np.uint8)
        if (X.ndim!=3 or X.shape[0]!=40 or X.shape[1]!=29
                or len(y)!=40 or set(original_labels.tolist())!=set(EVENTS)
                or int(y.sum())!=20 or not np.isfinite(X).all()):
            raise RuntimeError(f"fixed Liu2024 public release schema failed: {sid}")
        records.append({"subject":sid,"signal":X,"labels":y})
        print("V27_EPOCH_INPUT_VALIDATED",sid,"trials",len(y),flush=True)
    report=audit_exact_epoch_overlap(records)
    if (report["n_original_subject_ids"]!=50
            or report["n_total_trial_instances"]!=2000
            or report["epoch_shape_channels_samples"][0]!=29):
        raise AssertionError("fixed EEG release denominator drift")
    report.update({
        "dataset":"Liu2024",
        "pipeline":"MOABB 1.7.2 fixed 8-30Hz 160Hz bilateral MI",
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "known_previous_exact_full_record_duplicate_groups":[[4,5]],
        "extra_partial_overlap_components_beyond_full_record_identity":[
            g for g in report["exact_epoch_connected_components"]
            if len(g)>1 and g!=[4,5]
        ],
        "does_not_establish_physically_distinct_patients":True,
    })
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=run()
    body=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(body,encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "n_subject_ids":result["n_original_subject_ids"],
        "n_trial_instances":result["n_total_trial_instances"],
        "n_unique_exact_epochs":result["n_unique_exact_epoch_contents"],
        "exact_epoch_groups":result["n_exact_epoch_connected_groups"],
        "cross_subject_overlap_edges":result["cross_subject_exact_overlap_edges"],
        "extra_partial_overlap_groups":result["extra_partial_overlap_components_beyond_full_record_identity"],
        "receipt_sha256":hashlib.sha256(body.encode()).hexdigest(),
        "not_blinded_or_strict_Z0":True,
    },indent=2),flush=True)


if __name__=="__main__":
    main()
