#!/usr/bin/env python3
"""Independent re-audit of NEW protocol-locked BNCI2014_001 EEG cohort.

Verifies immutable GitHub Actions artifact, exact pre-target lambda SHA,
pre-run protocol SHA, nine subject and 2592 trial denominators, labels,
source-only feasibility, complete per-subject AUC and harm arithmetic.
The original program's execution ordering was also checked in Actions
logs. This is not independent preregistration or multi-site validation.
"""
from __future__ import annotations
import argparse,hashlib,io,json,statistics,zipfile
from pathlib import Path
import numpy as np

ART_SHA="36ddb760b0198a713be08b96b0586e575be5b5e6c73e3cbd87ef1b1b2e88f9fd"
SOURCE_SHA="d782c2e00c0cbdd7c3a7dc641e3ffc4e902ca8d075f61f65f78d8bb7d36216c0"
PROTOCOL_SHA="82fec5c15625948f9de583acc7923395b48da6718dac7e86c51f91742228d508"


def unique(z,suffix):
    paths=[p for p in z.namelist() if p.endswith(suffix)]
    if len(paths)!=1:raise AssertionError((suffix,paths))
    return z.read(paths[0])


def verify(zip_path):
    blob=Path(zip_path).read_bytes()
    if hashlib.sha256(blob).hexdigest()!=ART_SHA:
        raise AssertionError("Original GitHub Actions ZIP digest changed")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        src=unique(z,"SOURCE_ONLY_LAMBDA_CHOICES_SHA_LOCK_BEFORE_TARGET_EVALUATION.json")
        res=json.loads(unique(z,"BNCI2014_001_SOURCE_ONLY_EXTERNAL_RESULTS.json"))
        proto=unique(z,"BNCI2014_001_EXTERNAL_FIRST_RUN_PROTOCOL_20261012.md")
    if hashlib.sha256(src).hexdigest()!=SOURCE_SHA or \
       hashlib.sha256(proto).hexdigest()!=PROTOCOL_SHA:
        raise AssertionError("Source-choice lock or pre-run research protocol replaced")
    choices=json.loads(src)
    if (res["status"]!="NEW_EXTERNAL_BNCI2014_001_EEG_FIRST_RUN_COMPLETE_UNREGISTERED" or
        res["source_lambda_choices_sha256_before_target_outcomes"]!=SOURCE_SHA or
        res["pre_run_protocol_sha256"]!=PROTOCOL_SHA or
        res["target_information_track"]!="Z0" or
        res["new_confirmatory_p_values"] is not None or
        res["no_global_preregistration_claim"] is not True):
        raise AssertionError("Evidence chronology invalid")
    if (res["n_outer_people"]!=len(choices)!=9 or
        res["n_total_trials"]!=2592 or res["n_channels"]!=22 or
        not res["all_participants_retained"]):
        raise AssertionError("Denominators not exactly preserved")
    by={int(x["outer_subject"]):x for x in choices}
    outcome={int(x["subject"]):x for x in res["per_subject_AUC"]}
    receipts={int(x["subject"]):x for x in res["participant_receipts"]}
    if not (set(by)==set(outcome)==set(receipts)==set(range(1,10))):
        raise AssertionError("Missing source people/outer trial rows")
    effects=[]
    allnames=("macro","historical_class_centred","true_source_mean_predictive","LW","OAS")
    for sid in range(1,10):
        q=by[sid];r=outcome[sid];t=receipts[sid]
        if not q["no_outer_target_data_in_source_selector"] or \
           set(q["source_people"]) != (set(range(1,10))-{sid}):
            raise AssertionError("Outer leakage or missing source person")
        if r["test_trials"]!=t["trials"]!=288 or t["class0"]!=144 or t["class1"]!=144:
            raise AssertionError("Dataset/class denominator mismatch")
        grid=np.array(q["source_feasible_lambda_grid"],float)
        old=np.array(q["source_class_centred_log_scores"],float)
        pred=np.array(q["source_proper_predictive_log_scores"],float)
        if not (len(grid)==len(old)==len(pred)) or len(grid)<2:
            raise AssertionError("Mismatched source lambda domains")
        if (np.diff(grid)<=0).any() or (pred-old<-1e-8).any():
            raise AssertionError("Source-domain order or Gaussian penalty invalid")
        if abs(grid[np.argmin(old)]-q["selected_lambda"]["historical_class_centred"])>1e-12:
            raise AssertionError("Frozen historical source minimizer unreconciled")
        if abs(grid[np.argmin(pred)]-q["selected_lambda"]["true_source_mean_predictive"])>1e-12:
            raise AssertionError("Correct source minimizer unreconciled")
        li=list(grid).index(q["selected_lambda"]["true_source_mean_predictive"])
        if abs(float(pred[li]-old[li])-q["source_penalty_at_proper_lambda"])>1e-12:
            raise AssertionError("Source mean-shift term incorrectly reported")
        if set(r["auc"])!=set(allnames) or any(not(0<=v<=1) for v in r["auc"].values()):
            raise AssertionError("EEG AUC missing or invalid")
        effects.append(r["auc"]["macro"]-r["auc"]["true_source_mean_predictive"])
    for name in allnames:
        measured=float(np.mean([outcome[sid]["auc"][name] for sid in range(1,10)]))
        if abs(measured-res["mean_AUC"][name])>1e-12:
            raise AssertionError("Mean AUC arithmetic inconsistent "+name)
    e=np.asarray(effects)
    if abs(e.mean()-res["macro_minus_proper_mean_AUC"])>1e-12:
        raise AssertionError("Paired macro-minus-proper mean incorrect")
    summary={
        "status":"PASS_NEW_EXTERNAL_N9_PROTOCOL_FROZEN_EEG_INDEPENDENT_ARTIFACT_AUDIT",
        "dataset":"BNCI2014_001","subjects":9,"trials":2592,
        "outer_subjects":list(range(1,10)),
        "original_github_zip_sha256":ART_SHA,
        "research_protocol_sha256":PROTOCOL_SHA,
        "source_only_choice_vector_sha256_before_target_auc":SOURCE_SHA,
        "mean_auc_by_selector":res["mean_AUC"],
        "mean_macro_minus_proper_auc":float(e.mean()),
        "macro_minus_proper_subjects_positive":int((e>1e-12).sum()),
        "macro_minus_proper_subjects_negative":int((e< -1e-12).sum()),
        "macro_minus_proper_subjects_tied":int((abs(e)<=1e-12).sum()),
        "source_shift_penalty_mean_at_proper":float(np.mean([x["source_penalty_at_proper_lambda"] for x in choices])),
        "source_shift_penalty_positive_people":int(sum(x["source_penalty_at_proper_lambda"]>0 for x in choices)),
        "source_macro_harmed_gt_002_vs_proper":int((e<-0.020000000001).sum()),
        "proper_harmed_gt_002_vs_source_macro":int((e>0.020000000001).sum()),
        "first_change_alpha_median":float(np.median([x["first_change_alpha_from_class_centred"] for x in choices])),
        "no_independent_registration_claim":True,
        "no_significance_claim":True,
        "scope":"One nine-person public MI dataset; protocol fixed in repo before this run, no independent registration or evidence of worldwide prior outcome blindness"
    }
    if summary["source_shift_penalty_positive_people"]!=9:
        raise AssertionError("Real source penalty sign wrong")
    return summary


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--out")
    a=ap.parse_args()
    r=verify(a.zip)
    txt=json.dumps(r,indent=2)+"\n"
    if a.out:Path(a.out).write_text(txt)
    print(txt)
