#!/usr/bin/env python3
"""P3 six further HELD DreyerB people with 100% prior-source-locked choices.

Read immutable SOURCE lock from successful original Actions 38390737635.
Recompute only source covariance features and verify exact SHA-256;
DO NOT RESELECT or recalibrate ANY policy. Then (and only then)
read 6 as-yet unreported public target IDs71--76 from same DreyerB
acquisition. If the 8-source data/preproc changed, hard FAIL before
any new target outcome is read. Not independent acquisitions.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import Dreyer2023B
from moabb.paradigms import MotorImagery
from p3_dreyer2023b_first_source_locked_two_targets import load,SRC
from p3_bnci004_four_operator_same_source import evaluate

NEW_HELD=(71,72,73,74,75,76)
METHODS=("source_macro","continuous","SCE","anisotropic","group_centered")
OLD_RUN=38390737635

def write(path,record):
    raw=(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()

def execute(old_json,out):
    prior_bytes=old_json.read_bytes()
    frozen=json.loads(prior_bytes)
    if (frozen["status"]!="NEW_TRACKED_DREYER_B_SOURCE_CHOICES_BEFORE_HELD_69_70"
        or frozen["source_public_ids"]!=list(range(61,69))
        or frozen["held_public_ids_NOT_REQUESTED_YET"]!=[69,70]
        or set(frozen["source_policy_choices"])!=set(METHODS)):
        raise ValueError("wrong pre-existing source lock")
    ds=Dreyer2023B()
    paradigm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
        channels=["C3","Cz","C4"],fmin=8,fmax=30,resample=160,
        baseline=None,reject_by_annotation=False)
    C,y,g,recs,sha=load(ds,paradigm,SRC)
    if sha!=frozen["source_preprocessed_covariances_sha256"]:
        raise RuntimeError("source data/preprocessing SHA drift, no target access authorized")
    choices=frozen["source_policy_choices"]
    certified={"status":"SOURCE_DATA_SHA_MUST_MATCH_PRIOR_UNCHANGED_LOCK",
      "prior_run":OLD_RUN,"prior_decisions_SHA256":hashlib.sha256(prior_bytes).hexdigest(),
      "prior_original_source_file":"P3_DREYER_B_SOURCE_LOCK_PRETARGET.json",
      "source_covariance_sha256":sha,"exact_source_sha_match":True,
      "policy_choices_not_refitted_or_readapted":True,
      "source_public_ids":list(SRC),
      "target_public_ids_still_not_read":list(NEW_HELD),
      "prior_targets_69_70_outcomes_now_known_BUT_not_used_in_new_decisions":True,
      "acquisition_count":1,
      "not_externally_independently_registered":True}
    digest=write(out/"P3_DREYER_B_SIX_TARGET_PRE_READ_LOCK.json",certified)
    print("P3_DREYER_B_SIX_NEW_TARGET_PRE_READ_SOURCE_SHA",digest,flush=True)
    records=[]
    for sid in NEW_HELD:
        Ct,yt,gt,receipt,target_sha=load(ds,paradigm,[sid])
        auc=evaluate(C,y,g,Ct,yt,choices)
        delta={m:float(auc[m]-auc["source_macro"]) for m in METHODS}
        record={"held_public_id":sid,"n_trials":len(yt),
                "target_feature_sha256":target_sha,
                "auc":auc,"paired_delta_vs_prior_macro":delta}
        records.append(record)
        print("P3_DREYER_B_SIX_NEW_TARGET_READ",sid,
              json.dumps(record,sort_keys=True),flush=True)
    effects={m:[r["paired_delta_vs_prior_macro"][m] for r in records]
             for m in METHODS}
    means={m:float(np.mean(v)) for m,v in effects.items()}
    report={"status":"SIX_FURTHER_HELD_DREYERB_WITH_PRIOR_FROZEN_SOURCE",
      "prior_source_lock_sha256":hashlib.sha256(prior_bytes).hexdigest(),
      "within_run_pre_read_lock_sha256":digest,
      "new_held_subject_ids":list(NEW_HELD),"new_held_people":6,
      "distinct_EEG_acquisitions_or_cohorts":1,
      "previously_exposed_held_69_70_not_used_for_source_policy":True,
      "no_source_hyperparameters_reoptimised":True,
      "mean_paired_delta_AUC":means,
      "method_per_person_effect":effects,
      "method_negative_harm_over_0p02_count":{
           m:sum(v<-.02 for v in arr) for m,arr in effects.items()},
      "method_positive_over_0p01_count":{
           m:sum(v>.01 for v in arr) for m,arr in effects.items()},
      "no_acquisition_level_confidence_or_coverage_claim":True,
      "no_external_preregistered_held_subject":True,
      "rows":records}
    sha=write(out/"P3_DREYER_B_SIX_NEW_SOURCE_LOCKED_RESULTS.json",report)
    print("P3_DREYER_B_SIX_NEW_TARGET_MEANS",json.dumps(means,sort_keys=True),flush=True)
    print("P3_DREYER_B_SIX_NEW_RESULTS_SHA256",sha,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--prior-lock",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();execute(args.prior_lock,args.out)
