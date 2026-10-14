#!/usr/bin/env python3
"""New named public source-locked acquisition BNCI2014_002 (14 people).

Right-hand vs feet motor imagery; three predeclared *index-labelled* EEG
channels EEG7/8/9 from the 15 available Laplacian montage channels.
They are NOT the C3/Cz/C4 anatomical electrodes used in other papers.
Source IDs1--10. All 27-grid source criteria and simple baseline
parameters frozen and SHA256-written before reading target IDs11--14.
A new project run is NOT independent external preregistration; the four
people belong to one acquisition study. No source-person forecast
novelty is claimed from leaky shortcut cross-validation.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2014_002
from moabb.paradigms import MotorImagery
import p1_second_model_shrinkage_lda_forecast as lda
from p1_alexmi_new_source_lock_two_held import held_score

SOURCE=tuple(range(1,11))
HELD=(11,12,13,14)
EVENTS=("right_hand","feet")
CHANNELS=("EEG7","EEG8","EEG9")

def save(path,obj):
    b=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def read_one(paradigm,dataset,uid):
    x,l,_=paradigm.get_data(dataset=dataset,subjects=[uid])
    x=np.asarray(x,dtype=np.float64);l=np.asarray(l)
    if x.ndim!=3 or x.shape[1]!=len(CHANNELS) or len(x)<25:
        raise RuntimeError(f"BNCI2014_002 unexpected epoch geometry for {uid}: {x.shape}")
    if set(l.tolist())!=set(EVENTS) or not np.isfinite(x).all():
        raise ValueError(f"unknown BNCI2014_002 labels or non-finite data for {uid}")
    y=(l==EVENTS[0]).astype(int)
    if not .10<y.mean()<.90:
        raise ValueError("unexpected target class imbalance")
    c=lda.old.covariances(x)
    return c,y,{"public_id":int(uid),"epochs":int(len(y)),
                "right_hand_epochs":int(y.sum()),"channels":list(CHANNELS)}

def choice_only(validation,sids):
    ids=set(map(int,sids))
    if ids!=set(SOURCE):raise ValueError("source-validation identity changed")
    n=len(ids)
    macro=max(lda.GRID,key=lambda l:(float(np.mean(
        [auc for sid,auc,nll in validation[l] if sid in ids])),l))
    proper=min(lda.GRID,key=lambda l:(float(np.mean(
        [nll for sid,auc,nll in validation[l] if sid in ids])),l))
    return {"source_macro":float(macro),
            "source_predictive_nll":float(proper),
            "n_source_people":n,
            "source_criteria_only_not_individual_forecast":True}

def execute(out):
    grid=tuple(lda.old.FULL_GRID)
    if len(grid)!=27 or len(set(grid))!=27:raise ValueError("wrong original grid")
    lda.GRID=grid
    ds=BNCI2014_002()
    paradigm=MotorImagery(n_classes=2,events=list(EVENTS),channels=list(CHANNELS),
                          fmin=8,fmax=30,resample=160,
                          baseline=None,reject_by_annotation=False)
    cs=[];ys=[];groups=[];receipts=[];sha=hashlib.sha256()
    for uid in SOURCE:
        c,y,rec=read_one(paradigm,ds,uid)
        sha.update(np.ascontiguousarray(c).tobytes())
        cs.append(c);ys.append(y);groups.extend([uid]*len(y));receipts.append(rec)
        print("P1_BNCI002_SOURCE_ONLY_PUBLIC_ID_READ",uid,len(y),flush=True)
    source=np.concatenate(cs);labels=np.concatenate(ys);g=np.array(groups)
    cv,sids=lda.source_validation(source,labels,g)
    selected=choice_only(cv,sids)
    source_receipt={
       "status":"BNCI2014_002_10_SOURCE_BEFORE_ANY_OF_FOUR_HELD_IDS",
       "source_ids":list(SOURCE),
       "held_target_ids_UNREAD_at_this_stage":list(HELD),
       "original_27_grid":list(grid),
       "dataset":"BNCI2014_002","DOI":"10.1007/s00500-012-0895-4",
       "task":"two-class right_hand vs feet motor imagery",
       "channel_names_index_labelled_not_equivalent_to_C3CzC4":list(CHANNELS),
       "filter_fmin_fmax":[8,30],"resample_Hz":160,
       "source_feature_sha256":sha.hexdigest(),
       "source_trial_receipts":receipts,
       "source_choices":selected,
       "fixed_LDA_lambda":0.1,"fixed_tangent_logistic_C":1,
       "source_criteria_use_ONLY_source_groups":True,
       "no_source_subject_forecast_accuracy_claim":True,
       "no_target_selected_lambda_or_geometry":True,
       "not_externally_independently_preregistered":True,
       "not_four_independent_acquisitions":True,
       "runner_commit":os.environ.get("GITHUB_SHA","LOCAL")}
    lock=save(out/"P1_BNCI002_FROZEN_SOURCE_TEN_BEFORE_TARGETS.json",source_receipt)
    print("P1_BNCI002_FROZEN_SOURCE_PRETARGET_SHA256",lock,flush=True)
    rows=[]
    sel={"source_macro":selected["source_macro"],
         "source_predictive_nll":selected["source_predictive_nll"]}
    for uid in HELD:
        c,y,rec=read_one(paradigm,ds,uid)
        auc=held_score(source,labels,c,y,sel)
        row={"public_id":uid,"receipt":rec,"held_auc":auc,
             "paired_macro_minus_proper":float(auc["source_macro"]-auc["proper_predictive_NLL"]),
             "source_choices_frozen_before_held_fetch":True}
        rows.append(row)
        print("P1_BNCI002_FIRST_READ_HELD_RESULT",uid,
              json.dumps(row,sort_keys=True),flush=True)
    policies=("source_macro","proper_predictive_NLL","fixed_lambda_0p1","tangent_logistic_C1")
    means={k:float(np.mean([r["held_auc"][k] for r in rows])) for k in policies}
    result={
       "status":"NEW_PROJECT_NAMED_BNCI2014_002_SOURCE_LOCKED_HELD_FOUR",
       "original_source_lock_sha256":lock,
       "source_ids":list(SOURCE),"held_ids":list(HELD),
       "four_people_one_named_acquisition":True,
       "original_27_grid_matched":True,
       "held_trial_count":sum(r["receipt"]["epochs"] for r in rows),
       "source_choice":selected,
       "means":means,
       "mean_macro_minus_proper":float(means["source_macro"]-means["proper_predictive_NLL"]),
       "held_person_deltas":[r["paired_macro_minus_proper"] for r in rows],
       "n_positive":sum(r["paired_macro_minus_proper"]>0 for r in rows),
       "fixed_source_only_choices_before_all_held":True,
       "no_externally_registered_sampling_or_population_CI":True,
       "no_person_specific_forecast_value_established":True,
       "not_an_exact_channel_montage_replication_of_AlexMI":True,
       "results":rows}
    result_sha=save(out/"P1_BNCI002_NEW_FOUR_HELD_REAL_EEG_RESULTS.json",result)
    print("P1_BNCI002_NEW_REAL_EEG_SUMMARY",json.dumps({
        "means":means,"macro_minus_proper":result["mean_macro_minus_proper"],
        "n_positive":result["n_positive"],"n_held":4},sort_keys=True),flush=True)
    print("P1_BNCI002_REAL_RESULT_SHA256",result_sha,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();execute(a.out)
