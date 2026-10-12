"""V26 immutable old outcome 50-ID vs naive 49-ID target reweighting.

This is NOT a record-leakage repair. ALL old 50 LOSO source models remain
unchanged (including target duplicates during model fitting). The output
quantifies only the reporting denominator effect, never a new method gain.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np

SOURCE="FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json"
OUTCOME="FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json"
EXPECTED_SOURCE_SHA="d56f6f7169689ad80715fc94635a86fce19e936bc1f4536fc2f7845468323227"
KEEP_IDS=set(range(1,51))-{5}


def read_unique(z, suffix):
    m=[n for n in z.namelist() if n.endswith(suffix)]
    if len(m)!=1:raise AssertionError("archive member missing or ambiguous")
    return z.read(m[0])


def audit(path):
    with zipfile.ZipFile(io.BytesIO(Path(path).read_bytes())) as z:
        a=read_unique(z,SOURCE)
        b=read_unique(z,OUTCOME)
    if hashlib.sha256(a).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise AssertionError("original frozen clinical source selection SHA drift")
    d=json.loads(b)
    c=json.loads(a)
    if d["n_patients"]!=50 or len(d["participants"])!=50 or len(c)!=50:
        raise AssertionError("wrong original 50-ID denominator")
    if d["source_choices_SHA256_pre_target_evaluation"]!=EXPECTED_SOURCE_SHA:
        raise AssertionError("inconsistent source-choice lock")
    rows={int(p["held_patient"]):p for p in d["participants"]}
    assert set(rows)==set(range(1,51))
    assert [r["held_patient"] for r in d["participants"]]==list(range(1,51))
    assert rows[4]["auc"]==rows[5]["auc"]
    assert rows[47]["auc"]==rows[49]["auc"]
    methods=sorted(rows[1]["auc"])
    result={}
    for method in methods:
        full=np.asarray([rows[i]["auc"][method] for i in range(1,51)],float)
        kept=np.asarray([rows[i]["auc"][method] for i in sorted(KEEP_IDS)],float)
        mean50=float(np.mean(full))
        mean49=float(np.mean(kept))
        assert abs(mean50-d["mean_target_AUC_by_policy"][method])<1e-12
        result[method]={
            "original_50_ID_mean_auc":mean50,
            "naive_delete_ID5_49_target_mean_same_models":mean49,
            "only_denominator_effect":mean49-mean50,
        }
    return {
        "status":"PASS_RETROSPECTIVE_TARGET_REWEIGHTING_ONLY_NOT_GROUP_PURGE",
        "original_source_sha256":EXPECTED_SOURCE_SHA,
        "original_outcome_sha256":hashlib.sha256(b).hexdigest(),
        "original_identifier_count":50,
        "n_unique_processed_records_from_separate_all50_audit":49,
        "dropped_exact_duplicate_ID":5,
        "AUC_vectors_same_4_5":True,
        "AUC_vectors_same_47_49_but_RAW_EEG_different":True,
        "original_source_models_contain_record_duplication":True,
        "this_is_not_contamination_removal":True,
        "not_new_target_independent_test":True,
        "metrics":result,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_zip",type=Path)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    d=audit(x.original_zip)
    x.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n"
    x.out.write_text(body)
    print(json.dumps({
        "status":d["status"],
        "metrics":d["metrics"],
        "receipt_sha256":hashlib.sha256(body.encode()).hexdigest(),
    },indent=2))
