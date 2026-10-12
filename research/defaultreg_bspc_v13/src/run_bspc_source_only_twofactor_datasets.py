#!/usr/bin/env python3
"""EXPOSED-acquisition SOURCE-only source-validation factorisation.

No held target persons (IDs 13+) are read. This is a posthoc
mechanism/estimand audit, not a new external transfer test.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from moabb.datasets import Cho2017, Dreyer2023A
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from exact_prediction_shift_partition import source_group_partition

NAMES={"Cho2017":Cho2017,"Dreyer2023A":Dreyer2023A}
SOURCE_IDS=tuple(range(1,13))
EVENTS=("left_hand","right_hand")
CHANNELS=("C3","Cz","C4")

def source_only(name):
    ds=NAMES[name]()
    paradigm=old.MotorImagery(n_classes=2,events=list(EVENTS),
          channels=list(CHANNELS),fmin=8,fmax=30,resample=160,
          baseline=None,reject_by_annotation=False)
    Cs=[];Ys=[];G=[];receipts=[]
    for sid in SOURCE_IDS:
        X,labels,meta=paradigm.get_data(dataset=ds,subjects=[sid])
        yy=(np.asarray(labels)==EVENTS[1]).astype(int)
        if (X.ndim!=3 or X.shape[1]!=3 or
            set(labels)!=set(EVENTS) or
            not np.isfinite(X).all() or
            np.bincount(yy,minlength=2).min()<20):
            raise ValueError("FROZEN_SOURCE_ONLY_CHANNEL_OR_CLASS_MISMATCH_"+str(sid))
        C=old.covariances(X)
        Cs.append(C);Ys.append(yy);G.extend([sid]*len(yy))
        receipts.append({"source_subject_id":sid,"trials":len(yy),
                         "class0":int((yy==0).sum()),
                         "class1":int((yy==1).sum()),
                         "source_cov_sha256":hashlib.sha256(C.tobytes()).hexdigest()})
        print("SOURCE_ONLY_TWOFACTOR_READ",name,sid,len(yy),flush=True)
    return np.concatenate(Cs),np.concatenate(Ys),np.asarray(G),receipts

def save(obj,path):
    raw=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()

def run(name,out):
    C,y,g,receipts=source_only(name)
    original=old.source_path(C,y,g)
    src=source_group_partition(C,y,g,original["feasible_full_lambdas"])
    selections=src["selection_summary"]
    if selections["source_macro_auc"]!=original["lambda_source_macro"]:
        raise AssertionError("source macro AUC selection != ORIGINAL")
    if selections["historical_class_centered"]!=original["lambda_covariance_full"]:
        raise AssertionError("historical held-source centered selection != ORIGINAL")
    at={}
    for name_,lam in selections.items():
        row=next(v for v in src["rows"] if v["lambda"]==lam)
        t=row["auc_invariant_translation_contribution"]
        k=row["contrast_shift_contribution"]
        at[name_]={
            "lambda":lam,"macro_auc":row["source_macro_auc"],
            "class_centred":row["class_centred_mean"],
            "proper_predictive":row["proper_predictive_mean"],
            "translation_component":t,"contrast_component":k,
            "score_aligned_contrast_component":
                row["contrast_shift_score_aligned_contribution"],
            "score_orthogonal_contrast_component":
                row["contrast_shift_score_orthogonal_contribution"],
            "total_auc_invariant_location_plus_orthogonal_fraction":
                float((t+row["contrast_shift_score_orthogonal_contribution"])/(t+k))
                if t+k>1e-12 else None,
            "translation_fraction":float(t/(t+k)) if t+k>1e-12 else None
        }
    summary={
        "status":"COMPLETE_SOURCE_ONLY_POST_OUTCOME_MECHANISM",
        "dataset":name,"n_source_people":len(SOURCE_IDS),
        "source_only_ids":list(SOURCE_IDS),
        "no_target_subjects_opened":True,
        "has_prior_exposed_target_results_elsewhere":True,
        "not_independently_preregistered":True,
        "n_source_trials":int(len(y)),
        "source_receipts":receipts,
        "source_original_grid_lambda":original["lambda_source_macro"],
        "historical_original_centered_lambda":original["lambda_covariance_full"],
        "lambda_selections":selections,
        "selected_points":at,
        "all_full_grid_rows":src["rows"],
        "source_macro_agrees_with_original":True,
        "source_covscore_agrees_with_original":True,
        "no_new_algorithm_performance_claim":True,
    }
    output=out/(name.upper()+"_SOURCE_ONLY_SHIFT_TWOFACTOR_POSTOUTCOME.json")
    print("SOURCE_ONLY_TWOFACTOR_JSON_SHA",name,save(summary,output),flush=True)
    print("SOURCE_ONLY_TWOFACTOR_MAIN",json.dumps(
        {k:summary[k] for k in (
            "dataset","n_source_people","n_source_trials",
            "lambda_selections","selected_points")},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=list(NAMES),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.dataset,args.out)
