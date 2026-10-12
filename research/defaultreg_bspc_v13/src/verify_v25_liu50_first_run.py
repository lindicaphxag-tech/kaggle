"""Independent 50-patient Liu2024 immutable EEG first-run receipt audit.

Verifies frozen source choices, all participant-level AUC arithmetic and
flags coincident vectors; this does NOT inspect original raw EEG bytes.
It never refits a model nor tunes selection on target outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile
import numpy as np

SOURCES="FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json"
RESULTS="FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json"
EXPECTED_SOURCE_SHA="d56f6f7169689ad80715fc94635a86fce19e936bc1f4536fc2f7845468323227"
EXPECTED={
    "source_macro": .5940000000000001,
    "historical_class_centred": .59305,
    "proper_source_mean_gaussian": .58985,
    "v23_fail_closed_continuous": .5940000000000001,
    "LW": .59085, "OAS": .5898500000000001,
}


def read_once(z, suffix):
    found=[f for f in z.namelist() if f.endswith(suffix)]
    if len(found)!=1:
        raise AssertionError("missing or duplicate archive member "+suffix)
    return z.read(found[0])


def verify(archive):
    with zipfile.ZipFile(io.BytesIO(Path(archive).read_bytes())) as z:
        srcbytes=read_once(z,SOURCES)
        outbytes=read_once(z,RESULTS)
    if hashlib.sha256(srcbytes).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise AssertionError("pre-target lambda choice manifest SHA drift")
    sources=json.loads(srcbytes)
    result=json.loads(outbytes)
    if result["source_choices_SHA256_pre_target_evaluation"]!=EXPECTED_SOURCE_SHA:
        raise AssertionError("source SHA not matched in outcome manifest")
    assert result["status"]=="EXTERNAL_50_ACUTE_STROKE_Z0_LOPO_COMPLETE_UNREGISTERED"
    assert result["n_patients"]==50 and result["trials"]==2000
    assert result["n_eeg_channels"]==29
    assert len(sources)==len(result["participants"])==50
    assert {x["outer_subject"] for x in sources}==set(range(1,51))
    assert {x["held_patient"] for x in result["participants"]}==set(range(1,51))
    assert all(x["source_trial_count"]==1960 and x["n_source_people"]==49 for x in sources)
    assert all(x["source_people"]==sorted(set(range(1,51))-{x["outer_subject"]})
               for x in sources)
    assert all(x["no_outer_target_information_passed_to_selection"] for x in sources)
    assert all(x["n_trials"]==40 for x in result["participants"])
    assert all(x["source_choice_sha_verified"] for x in result["participants"])
    assert all(not x["v23_certified"] and x["v23_fallback_to_original_macro"] for x in sources)
    assert all(x["chosen_lambda"]["v23_fail_closed_continuous"]==
               x["chosen_lambda"]["source_macro"] for x in sources)
    patients=sorted(result["participants"],key=lambda x:x["held_patient"])
    auc_by={m:np.asarray([p["auc"][m] for p in patients]) for m in EXPECTED}
    for k,v in auc_by.items():
        assert np.isfinite(v).all() and np.all((v>=0)&(v<=1))
        assert abs(float(v.mean())-EXPECTED[k])<1e-12
        assert abs(float(v.mean())-result["mean_target_AUC_by_policy"][k])<1e-12
    macro=auc_by["source_macro"]
    proper=auc_by["proper_source_mean_gaussian"]
    difference=macro-proper
    assert abs(float(difference.mean())-result["v23_minus_proper_predictive"]["mean_AUC_difference"])<1e-12
    assert sum(difference < -.020000000001)==result["v23_minus_proper_predictive"]["harm_GT_0_02_patients"]
    assert sum(difference > .020000000001)==result["v23_minus_proper_predictive"]["improved_GT_0_02_patients"]
    assert np.array_equal(macro,auc_by["v23_fail_closed_continuous"])

    matched=[]
    keys=sorted(EXPECTED)
    for i,j in itertools.combinations(range(50),2):
        left=np.asarray([auc_by[k][i] for k in keys])
        right=np.asarray([auc_by[k][j] for k in keys])
        if np.array_equal(left,right):
            matched.append([patients[i]["held_patient"],patients[j]["held_patient"]])
    return {
        "status":"PASS_IMMUTABLE_LIU2024_50_PATIENT_FIRST_RUN_REAUDIT",
        "dataset":"Liu2024",
        "source_frozen_sha256":EXPECTED_SOURCE_SHA,
        "outcome_frozen_sha256":hashlib.sha256(outbytes).hexdigest(),
        "n_people":50,"n_trials":2000,
        "mean_auc":{k:float(v.mean()) for k,v in auc_by.items()},
        "macro_minus_proper_mean":float(difference.mean()),
        "macro_minus_proper_harmed_gt002":int(np.sum(difference<-.020000000001)),
        "macro_minus_proper_improved_gt002":int(np.sum(difference>.020000000001)),
        "all_v23_continuous_fallback_to_macro":True,
        "duplicate_per_person_AUC_vectors":matched,
        "duplicate_vector_does_not_establish_duplicate_raw_EEG":True,
        "no_raw_EEG_fingerprint_comparison_in_this_audit":True,
        "external_test_but_single_clinical_center_and_nonregistered":True,
    }


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("artifact_zip",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    d=verify(args.artifact_zip)
    text=json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text)
    print(text)
