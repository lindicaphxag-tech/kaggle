#!/usr/bin/env python3
"""P3 previously untracked Dreyer2023B public-acquisition source lock.

Dreyer2023B is the B subset (IDs61..81, one cohort under the
same broader Dreyer2023 study; Dreyer2023A was analysed earlier).
Take B IDs61..68 as labelled training people and reserve IDs69/70.
Within this one separate cohort, all five policies share ONE
source-fitted model, same C3/Cz/C4 8-30Hz 160Hz preproc,
source chosen BEFORE target acquisition is requested. This is
NOT an externally preregistered, independently collected study
relative to Dreyer2023A, and two held people are not two datasets.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from moabb.datasets import Dreyer2023B
from moabb.paradigms import MotorImagery
from p3_bnci004_four_operator_same_source import old,select,evaluate

SRC=tuple(range(61,69))
TGT=(69,70)

def store(path,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def load(ds,paradigm,ids):
    C=[];y=[];g=[];receipts=[];s=hashlib.sha256()
    for sid in ids:
        X,events,_=paradigm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);events=np.asarray(events)
        if X.ndim!=3 or X.shape[1]!=3 or len(X)<30 or set(events.tolist())!={"left_hand","right_hand"} or not np.isfinite(X).all():
            raise RuntimeError(f"DreyerB unexpected input {sid}: {X.shape}")
        yy=np.asarray(events=="right_hand",dtype=np.int8)
        if not .1<yy.mean()<.9:raise ValueError("DreyerB class imbalance")
        cov=old.covariances(X)
        s.update(np.ascontiguousarray(cov).tobytes())
        C.append(cov);y.append(yy);g.extend([sid]*len(yy))
        receipts.append({"public_id":sid,"n":len(yy),"right":int(yy.sum()),"channels":3})
        print("P3_DREYER_B_PUBLIC_RELEASE_ACCESSED",sid,len(yy),flush=True)
    return np.concatenate(C),np.concatenate(y),np.asarray(g),receipts,s.hexdigest()

def run(out):
    ds=Dreyer2023B()
    paradigm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
            channels=["C3","Cz","C4"],fmin=8,fmax=30,
            resample=160,baseline=None,reject_by_annotation=False)
    C,y,g,receipts,src_hash=load(ds,paradigm,SRC)
    opts=select(C,y,g)
    lock={"status":"NEW_TRACKED_DREYER_B_SOURCE_CHOICES_BEFORE_HELD_69_70",
         "dataset":"Dreyer2023B",
         "data_subcohort_of_same_original_Dreyer2023_study":True,
         "dataset_reference":"Dreyer et al. 2023, MOABB Dreyer2023B public cohort",
         "source_public_ids":list(SRC),"held_public_ids_NOT_REQUESTED_YET":list(TGT),
         "source_preprocessed_covariances_sha256":src_hash,
         "source_receipts":receipts,"source_policy_choices":opts,
         "algorithm_families":["source_macro","continuous","SCE","anisotropic","group_centered"],
         "source_params":"C3 Cz C4, left/right MI, 8-30Hz, 160Hz, no held target features",
         "pipeline_commit":os.environ.get("GITHUB_SHA","LOCAL_NOT_GITHUB"),
         "source_lock_before_ANY_target_read":True,
         "external_preregistration":False}
    sha=store(out/"P3_DREYER_B_SOURCE_LOCK_PRETARGET.json",lock)
    print("P3_DREYER_B_SOURCE_FROZEN_SHA256_BEFORE_HELD",sha,flush=True)
    results=[]
    for sid in TGT:
        Ct,yt,_,rr,tgtsha=load(ds,paradigm,[sid])
        scores=evaluate(C,y,g,Ct,yt,opts)
        diffs={k:float(scores[k]-scores["source_macro"]) for k in scores}
        results.append({"held_public_id":sid,"source_choice_sha":sha,
          "n_held_binary_trials":len(yt),"auc":scores,
          "delta_vs_source_macro":diffs,
          "target_dataset_same_Dreyer_B_cohort":True,
          "target_covariance_SHA256":tgtsha})
        print("P3_DREYER_B_FIRST_HELD_OUTCOME",sid,
              json.dumps(results[-1],sort_keys=True),flush=True)
    methods=tuple(results[0]["auc"].keys())
    means={m:float(np.mean([r["delta_vs_source_macro"][m] for r in results]))
           for m in methods}
    report={"status":"NEW_TRACKED_DREYER_B_TWO_TARGET_FIVE_POLICY_COMMON_SOURCE",
         "source_lock_sha256":sha,"source_ids":list(SRC),"target_ids":list(TGT),
         "rows":results,"descriptive_mean_paired_delta_vs_macro":means,
         "acquisition_level_count":1,
         "Dreyer_A_previously_explored_not_independent_lab":True,
         "two_held_people_one_data_collection":True,
         "no_target_adaptive_reselection":True,
         "no_claim_of_external_preregistration":True,
         "not_proven_cross_acquisition_utility":True}
    dh=store(out/"P3_DREYER_B_FIRST_TWO_HELD_REAL_RESULTS.json",report)
    print("P3_DREYER_B_REAL_TWO_TARGET_MEAN_DELTA",json.dumps(means,sort_keys=True),flush=True)
    print("P3_DREYER_B_RESULT_SHA256",dh,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    x=ap.parse_args();run(x.out)
