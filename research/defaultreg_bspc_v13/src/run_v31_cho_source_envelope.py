#!/usr/bin/env python3
"""V31 strict source-locked Cho2017: new envelope vs frozen classical baselines.

No held TARGET read until source choices are SHA-frozen on disk. This is a
first-run evaluation, not independently registered and NOT a clinical test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import Cho2017

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from run_proper_predictive_posthoc import select_proper_source_nll
from source_contrast_envelope_v31 import (
    source_envelope,select_source_envelope,direction,pairwise_dominance)

SRC=tuple(range(1,13))
HELD=tuple(range(13,33))
SENSORS=("C3","Cz","C4")
CLASSES=("left_hand","right_hand")
FREEZE_NAME="CHO2017_V31_SOURCE_CHOICES_BEFORE_ANY_TARGET_READ.json"
TARGET_NAME="CHO2017_V31_REAL_FIRST_RUN_TARGET_AUC.json"

def digest(body):
    return hashlib.sha256(body).hexdigest()

def save(obj,path):
    body=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(body)
    return digest(body)

def fetch_person(paradigm,dataset,sid):
    eeg,labels,_=paradigm.get_data(dataset=dataset,subjects=[sid])
    lab=np.asarray(labels)
    if (eeg.ndim!=3 or eeg.shape[1]!=3 or
        set(lab.tolist())!=set(CLASSES) or
        not np.isfinite(eeg).all()):
        raise RuntimeError(f"FROZEN_CHO_RELEASE_CHANNEL_OR_EVENT_FAILURE_{sid}")
    yy=(lab==CLASSES[1]).astype(np.int64)
    if np.bincount(yy,minlength=2).min()<20:
        raise RuntimeError(f"FROZEN_CHO_RELEASE_TRIAL_FAILURE_{sid}")
    return old.covariances(eeg),yy,{"sid":int(sid),
                                   "n_trials":int(len(yy)),
                                   "n_positive":int(yy.sum()),
                                   "n_channels":int(eeg.shape[1])}

def load_source(paradigm,dataset):
    cov=[];labels=[];ids=[];receipts=[]
    for sid in SRC:
        c,y,r=fetch_person(paradigm,dataset,sid)
        cov.append(c);labels.append(y);ids.extend([sid]*len(y))
        receipts.append(r)
        print("V31_SOURCE_READ",sid,flush=True)
    return (np.concatenate(cov),np.concatenate(labels),
            np.asarray(ids),receipts)

def freeze_only_source(C,y,groups,receipt,protocol):
    if set(groups.tolist())!=set(SRC):
        raise RuntimeError("source denominator differs from precommitted protocol")
    inv=old.reference(C)
    X=old.tangent(C,inv)
    st=old.spectral_state(X,y)
    grid=old.source_path(C,y,groups)
    feasible=grid["feasible_full_lambdas"]
    proper,proper_rows=select_proper_source_nll(C,y,groups,feasible)
    model=source_envelope(X,y,groups,shape_ridge=.01)
    robust=select_source_envelope(model,feasible,radius=1.)
    ablation=select_source_envelope(model,feasible,radius=0.)
    macro=grid["lambda_source_macro"]
    sel={"envelope_r1":float(robust["selected_lambda"]),
         "source_macro_grid":float(macro),
         "proper_source_mean_gaussian":float(proper),
         "historical_class_centred":float(grid["lambda_covariance_full"]),
         "envelope_r0":float(ablation["selected_lambda"])}
    predicted=pairwise_dominance(
        model,direction(model,sel["envelope_r1"]),
        direction(model,sel["source_macro_grid"]),radius=1.)
    return inv,st,{
        "study":"V31_CHO2017_SOURCE_LOCK_BEFORE_TARGET_IO",
        "protocol_sha256":digest(protocol.read_bytes()),
        "source_subjects":list(SRC),"held_subjects_unread":list(HELD),
        "sensor_names":list(SENSORS),
        "source_dataset_receipts":receipt,
        "selected_lambdas":sel,
        "feasible_lambdas":list(map(float,feasible)),
        "envelope_shape_ridge":.01,"envelope_radius":1.,
        "conditional_margin_prediction_vs_macro":predicted,
        "not_unconditional_transfer_or_gaussian_coverage_guarantee":True,
        "no_target_feature_moment_or_label_read_at_selection":True,
    }

def evaluate(paradigm,dataset,inv,st,Cs,ys,manifest):
    source_X=old.tangent(Cs,inv)
    records=[]
    for sid in HELD:
        C,y,r=fetch_person(paradigm,dataset,sid)
        Xt=old.tangent(C,inv)
        scores={
            method:old.scalar_scores(st,Xt,lam)
            for method,lam in manifest["selected_lambdas"].items()}
        for kind in ("LW","OAS"):
            sc,_=old.analytic_scores(source_X,ys,Xt,kind)
            scores[kind]=sc
        aucs={name:float(roc_auc_score(y,sc)) for name,sc in scores.items()}
        if not all(0<=value<=1 for value in aucs.values()):
            raise RuntimeError("invalid AUC")
        diff=aucs["envelope_r1"]-aucs["source_macro_grid"]
        records.append({"subject":int(sid),"n_trials":r["n_trials"],
                        "auc":aucs,"primary_delta":float(diff)})
        print("V31_HELD_TARGET_EVALUATED",sid,flush=True)
    if set(r["subject"] for r in records)!=set(HELD):
        raise RuntimeError("incomplete target denominator")
    keys=list(records[0]["auc"])
    mean={k:float(np.mean([r["auc"][k] for r in records])) for k in keys}
    d=np.asarray([r["primary_delta"] for r in records])
    rng=np.random.default_rng(20261013)
    ix=rng.integers(0,len(d),size=(20000,len(d)))
    bounds=np.quantile(d[ix].mean(axis=1),[.025,.975]).tolist()
    return {
        "status":"V31_CHO2017_TRUE_FIRST_RUN_COMPLETE" if len(records)==20 else "FAIL",
        "source_choice_sha256":manifest["source_lock_sha256"],
        "all_target_data_read_after_source_lock":True,
        "n_source_subjects":len(SRC),"n_held_subjects":len(HELD),
        "n_target_trials":int(sum(r["n_trials"] for r in records)),
        "macro_auc_by_method":mean,
        "primary_mean_delta_envelope_minus_source_macro":float(d.mean()),
        "primary_descriptive_participant_bootstrap_95":bounds,
        "harmed_gt_002":int((d < -.020000000001).sum()),
        "improved_gt_002":int((d > .020000000001).sum()),
        "exact_two_sided_sign_flip_pvalue":signflip_p(d),
        "protocol_committed_but_not_independently_registered":True,
        "no_unconditional_transfer_guarantee":True,
        "participant_rows":records,
    }

def signflip_p(d):
    """One predeclared two-sided sign-flip comparison, exact up to 20 people."""
    # 2**20 x 20 temporary matrix would cost memory; chunk deterministically.
    arr=np.asarray(d,dtype=float)
    n=len(arr)
    target=abs(float(arr.sum()))-1e-12
    total=1<<n
    extremes=0
    for base in range(0,total,8192):
        inds=np.arange(base,min(base+8192,total),dtype=np.uint32)
        signs=1-2*((inds[:,None]>>np.arange(n,dtype=np.uint32))&1).astype(np.int8)
        extremes+=int((np.abs(signs@arr)>=target).sum())
    return float(extremes/total)

def main(out):
    protocol=Path(__file__).resolve().parents[1]/"V31_CHO2017_SOURCE_ENVELOPE_FIRST_RUN_PROTOCOL.md"
    if not protocol.is_file():
        raise RuntimeError("missing pre-outcome protocol")
    paradigm=old.MotorImagery(
        n_classes=2,events=list(CLASSES),channels=list(SENSORS),
        fmin=8,fmax=30,resample=160,baseline=None,reject_by_annotation=False)
    ds=Cho2017()
    Cs,ys,gs,receipt=load_source(paradigm,ds)
    inv,st,manifest=freeze_only_source(Cs,ys,gs,receipt,protocol)
    out.mkdir(parents=True,exist_ok=True)
    lock=save(manifest,out/FREEZE_NAME)
    # NO HELD SUBJECT IS READ ABOVE THIS LINE.
    print("V31_SOURCE_CHOICES_SHA256_LOCKED_BEFORE_HELD_IO",lock,flush=True)
    manifest["source_lock_sha256"]=lock
    result=evaluate(paradigm,ds,inv,st,Cs,ys,manifest)
    result["protocol_sha256"]=manifest["protocol_sha256"]
    save(result,out/TARGET_NAME)
    print("V31_TARGET_SUMMARY",json.dumps({
        k:result[k] for k in (
        "status","macro_auc_by_method",
        "primary_mean_delta_envelope_minus_source_macro",
        "harmed_gt_002","exact_two_sided_sign_flip_pvalue"
        )}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
