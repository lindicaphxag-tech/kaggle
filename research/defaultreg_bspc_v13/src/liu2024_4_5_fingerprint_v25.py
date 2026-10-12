"""Forensic EEG data integrity check for Liu2024 AUC-vector collisions.

Preprocessed values are compared from the original V24 MOABB pipeline.
The released EDF file hash is separately compared to distinguish a source
artifact from a preprocessing/data-loader collision. Only hashes and
aggregate identity tests appear in logs; never publish patient EEG samples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery


def read_one(subject):
    dataset=Liu2024()
    paradigm=MotorImagery(
        n_classes=2, events=["left_hand","right_hand"],
        fmin=8, fmax=30, resample=160, baseline=None,
        reject_by_annotation=False,
    )
    X,lab,meta=paradigm.get_data(dataset=dataset, subjects=[subject])
    if X.shape[1]!=29 or X.shape[0]!=40:
        raise ValueError("fixed dataset acquisition protocol drift")
    edf=Path(dataset.data_path(subject)[0])
    if not edf.is_file():
        raise RuntimeError("original released EDF file unavailable")
    h=hashlib.sha256()
    with edf.open("rb") as file:
        for chunk in iter(lambda:file.read(1<<20),b""):
            h.update(chunk)
    return np.asarray(X,dtype=np.float64),np.asarray(lab),h.hexdigest()


def fingerprint(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def check_pair(a,b):
    X,y,raw_sha=read_one(a)
    Z,labels,other_sha=read_one(b)
    if X.shape!=Z.shape or y.shape!=labels.shape:
        raise AssertionError("paired EEG feature dimension mismatch")
    return {
        "ids":[a,b],
        "shape":list(X.shape),
        "labels_equal":bool(np.array_equal(y,labels)),
        "preprocessed_trial_EEG_exactly_equal":bool(np.array_equal(X,Z)),
        "mean_absolute_preprocessed_difference":float(np.mean(np.abs(X-Z))),
        "preprocessed_sha_first":fingerprint(X),
        "preprocessed_sha_second":fingerprint(Z),
        "released_edf_file_sha_first":raw_sha,
        "released_edf_file_sha_second":other_sha,
        "released_edf_binary_files_identical":raw_sha==other_sha,
    }


if __name__=="__main__":
    pairs=[(4,5),(47,49)]
    results=[check_pair(a,b) for a,b in pairs]
    out={
        "status":"RETROSPECTIVE_LIU2024_DATA_PROVENANCE_FORENSIC_TWO_AUC_VECTOR_COLLISIONS",
        "dataset":"Liu2024",
        "tested_pairs":[list(x) for x in pairs],
        "same_published_MOABB_1_7_2_preprocessing":True,
        "source_data_identity_cause_NOT_assigned_without_further_examination":True,
        "not_patient_risk_or_model_performance_evidence":True,
        "results":results,
    }
    path=Path("/tmp/defaultreg-v25-liu-audit/raw_collision_pairs_provenance.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
