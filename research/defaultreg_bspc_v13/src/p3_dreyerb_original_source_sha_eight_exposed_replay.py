#!/usr/bin/env python3
"""Replay eight PREVIOUSLY EXPOSED DreyerB held IDs69..76 on one exact source.
Hard-stop before target access unless source SHA matches original
authoritative Oct13 source file AND all policy choices unchanged.
This is retrospective same-source comparability repair,
NOT new unseen subjects, independent acquisitions or confirmatory test.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import Dreyer2023B
from moabb.paradigms import MotorImagery
from p3_dreyer2023b_first_source_locked_two_targets import load,SRC
from p3_bnci004_four_operator_same_source import evaluate

HELD=tuple(range(69,77))
CHOICES=("source_macro","continuous","SCE","anisotropic","group_centered")

def save(path,obj):
    b=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def main(lock_path,out):
    frozen=json.loads(lock_path.read_text())
    if frozen["source_public_ids"]!=list(SRC) or not set(CHOICES).issubset(frozen["source_policy_choices"]):
        raise ValueError("not original source subject/policy receipt")
    ds=Dreyer2023B()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
        channels=["C3","Cz","C4"],fmin=8,fmax=30,resample=160,
        baseline=None,reject_by_annotation=False)
    C,y,g,receipt,sha=load(ds,p,SRC)
    if sha!=frozen["source_preprocessed_covariances_sha256"]:
        raise RuntimeError("Exact original source SHA mismatch: no target read")
    if [(x["public_id"],x["n"],x["right"]) for x in receipt] != [
        (x["public_id"],x["n"],x["right"]) for x in frozen["source_receipts"]]:
        raise RuntimeError("Original source cohort accounting mismatch")
    source={"status":"EXACT_ORIGINAL_DREYERB_SOURCE_SHA_AND_PRIOR_CHOICES_RESTORED",
        "source_sha":sha,"source_matches_original_oct13_exactly":True,
        "frozen_source_policy_choices":frozen["source_policy_choices"],
        "held_ids_NOT_YET_READ_IN_THIS_REPLAY":list(HELD),
        "target_outcomes_ALL_PREVIOUSLY_EXPOSED":True,
        "fitted_classifier_weights_NOT_saved_originally":
            True,
        "recomputed_fit_on_exact_original_source_features":True,
        "not_strict_bitwise_same_classifier_parameters_proven":True,
        "still_not_external_preregistered":True}
    lock=save(out/"P3_DREYER_B_EIGHT_PERSON_SAME_SOURCE_PRE_READ.json",source)
    print("P3_DREYER_8_SAME_SOURCE_LOCK",lock,flush=True)
    rows=[]
    for person in HELD:
        Ct,yt,_,r,th=load(ds,p,[person])
        vals=evaluate(C,y,g,Ct,yt,frozen["source_policy_choices"])
        delta={m:float(vals[m]-vals["source_macro"]) for m in CHOICES}
        rows.append({"id":person,"n":len(yt),"auc":vals,"paired_delta":delta,
                     "target_covariance_sha":th})
        print("P3_DREYER_8_SAME_SOURCE_HELD_RESULT",person,
              json.dumps(delta,sort_keys=True),flush=True)
    per={m:[x["paired_delta"][m] for x in rows] for m in CHOICES}
    overall={m:float(np.mean(v)) for m,v in per.items()}
    a={m:float(np.mean(v[:2])) for m,v in per.items()}
    b={m:float(np.mean(v[2:])) for m,v in per.items()}
    result={"status":"EXACT_ORIGINAL_SOURCE_REPLAY_EIGHT_PREVIOUSLY_EXPOSED_HELD_DREYERB",
        "pre_target_replay_lock_sha256":lock,
        "original_source_feature_sha256":sha,
        "unchanged_source_policy_choices":True,
        "same_source_feature_bytes_for_all_eight_targets":True,
        "previously_exposed_target_records":list(HELD),
        "same_underlying_dreyer2023b_collection":True,
        "not_externally_registered":True,
        "not_new_confirmatory_eeg_data":True,
        "cannot_claim_bitwise_original_classifier_weights":True,
        "mean_delta_eight_held":overall,
        "mean_delta_original_two_69_70":a,
        "mean_delta_more_six_71_76":b,
        "negative_gain_worse_than_0p02_per_method":{
            m:sum(z<-.02 for z in v) for m,v in per.items()},
        "rows":rows}
    h=save(out/"P3_DREYERB_EIGHT_PEOPLE_ONE_EXACT_SOURCE_REPLAY.json",result)
    print("P3_DREYERB_EIGHT_SAME_SOURCE_MEANS",json.dumps(
        {"eight":overall,"first2":a,"later6":b},sort_keys=True),flush=True)
    print("P3_DREYERB_EIGHT_SAME_SOURCE_RESULT_SHA",h,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-lock",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args();main(x.original_lock,x.out)
