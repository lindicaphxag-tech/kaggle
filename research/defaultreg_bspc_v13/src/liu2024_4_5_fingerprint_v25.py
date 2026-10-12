"""Forensic check: are Liu2024 EEG subjects 4 and 5 identical trial arrays?

This is an exploratory retrospective data-integrity diagnostic.
Identical AUC vectors across classifiers do not prove raw EEG duplication.
We compare participant-specific MOABB-preprocessed X and event labels
without exposing any identifiable raw samples in stdout/results.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery


def fingerprint(subject):
    paradigm=MotorImagery(
        n_classes=2, events=["left_hand","right_hand"],
        fmin=8,fmax=30,resample=160,baseline=None,reject_by_annotation=False,
    )
    X,lab,meta=paradigm.get_data(dataset=Liu2024(),subjects=[subject])
    if X.shape[1]!=29 or X.shape[0]!=40:
        raise ValueError("EEG acquisition format drift")
    return np.asarray(X,np.float64),np.asarray(lab)


if __name__=="__main__":
    a,ya=fingerprint(4)
    b,yb=fingerprint(5)
    if a.shape!=b.shape or ya.shape!=yb.shape:
        raise ValueError("invalid matched raw-array comparison")
    def digest(x):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    record={
        "status":"LIU2024_SUBJECT_4_5_PREPROCESSED_EEG_FINGERPRINT_AUDIT",
        "array_shape":list(a.shape),
        "labels_equal":bool(np.array_equal(ya,yb)),
        "full_eeg_arrays_bitwise_identical":bool(np.array_equal(a,b)),
        "subject_4_eeg_sha256":digest(a),
        "subject_5_eeg_sha256":digest(b),
        "max_absolute_difference":float(np.max(np.abs(a-b))),
        "mean_absolute_difference":float(np.mean(np.abs(a-b))),
        "metric_vector_coincidence_alone_is_not_raw_duplicate_evidence":True,
        "not_a_new_model_result_or_external_independent_validation":True,
    }
    path=Path("/tmp/defaultreg-v25-liu-audit/raw_subject4_5_fingerprint.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
