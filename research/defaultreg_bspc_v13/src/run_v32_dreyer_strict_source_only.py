#!/usr/bin/env python3
"""Strict source SHA-first Dreyer2023A, V32 nonisotropic contrast-variance LDA.

Source IDs 1..12 / HELD 13..24. NEVER read held persons before writing
SHA-256 source manifest. This experiment is not independently preregistered.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from moabb.datasets import Dreyer2023A
from pyriemann.classification import MDM

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from run_proper_predictive_posthoc import select_proper_source_nll
from noise_debiased_subject_operator_v32 import fit,weight,score,group_cv

SOURCE_IDS=tuple(range(1,13))
TARGET_IDS=tuple(range(13,25))
CH=("C3","Cz","C4")
EVENTS=("left_hand","right_hand")
ALPHAS=(0.,.5,2.)
BETA=.25
SOURCE_FILE="V32_DREYER_SOURCE_CHOICES_LOCKED_BEFORE_TARGET.json"
TARGET_FILE="V32_DREYER_NEW_ACQUISITION_ALL_HELD_RESULTS.json"
PROTO="V32_DREYER_STRICT_SOURCE_ONLY_FIRST_RUN_PROTOCOL.md"

def sha(x):
    return hashlib.sha256(x).hexdigest()

def save(o,path):
    data=(json.dumps(o,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    return sha(data)

def person(paradigm,dataset,sid):
    X,labels,meta=paradigm.get_data(dataset=dataset,subjects=[sid])
    y=(np.asarray(labels)==EVENTS[1]).astype(np.int64)
    if (X.ndim!=3 or X.shape[1]!=3 or set(labels)!=set(EVENTS)
        or not np.isfinite(X).all() or np.bincount(y,minlength=2).min()<20):
        raise RuntimeError(f"FROZEN_SOURCE_OR_TARGET_PROTOCOL_SCHEMA_CHANGED_{sid}")
    C=old.covariances(X)
    return C,y,{"subject":int(sid),"n_epochs":int(len(y)),
                "n_left":int(np.sum(y==0)),"n_right":int(np.sum(y==1)),
                "source_or_target_receipt":sha(C.tobytes())[:24]}

def load_sources(paradigm,dataset):
    Cs=[];Ys=[];Gs=[];receipt=[]
    for sid in SOURCE_IDS:
        C,y,r=person(paradigm,dataset,sid)
        Cs.append(C);Ys.append(y);Gs.extend([sid]*len(y));receipt.append(r)
        print("V32_SOURCE_COVARIANCES_READ",sid,len(y),flush=True)
    return np.concatenate(Cs),np.concatenate(Ys),np.array(Gs),receipt

def original_paths(C,y,g):
    path=old.source_path(C,y,g)
    feasible=path["feasible_full_lambdas"]
    new=group_cv(C,y,g,feasible,ALPHAS,beta=BETA)
    macro=path["lambda_source_macro"]
    if abs(new["baseline"]["lambda"]-macro)>1e-12:
        raise AssertionError("V32 alpha0 source baseline must match exact V24 source grid")
    if abs(new["baseline"]["macro_auc"]-
           path["criteria_by_lambda"][str(macro)]["source_macro_auc"])>1e-10:
        raise AssertionError("source model-CV baseline mismatch")
    proper,_=select_proper_source_nll(C,y,g,feasible)
    risk_scalar=max((r for r in new["candidate_group_source_cv"] if r["alpha"]==0),
                    key=lambda r:(r["risk_score"],r["lambda"]))
    return path,new,{
      "source_macro_original":{"lambda":float(macro),"alpha":0.},
      "new_risk_operator":{"lambda":new["risk_best"]["lambda"],
                           "alpha":new["risk_best"]["alpha"]},
      "mean_operator_ablation":{"lambda":new["mean_best"]["lambda"],
                                "alpha":new["mean_best"]["alpha"]},
      "risk_scalar_ablation":{"lambda":risk_scalar["lambda"],"alpha":0.},
      "proper_predictive_nll":{"lambda":float(proper),"alpha":0.},
      "historical_class_centred":{"lambda":float(path["lambda_covariance_full"]),"alpha":0.},
      "fixed_full_shrink":{"lambda":1.0,"alpha":0.},
    }

def prepare_source_models(C,y,g,choices):
    inv=old.reference(C)
    Xt=old.tangent(C,inv)
    st=old.spectral_state(Xt,y)
    new_fit=fit(Xt,y,g)
    # Structural novelty does not follow if new alpha=0 or weight parallel.
    r=choices["new_risk_operator"]
    w=weight(new_fit,r["lambda"],r["alpha"])
    baseline_same_lam=weight(new_fit,r["lambda"],0.)
    cosine=float((w@baseline_same_lam)/
          (np.linalg.norm(w)*np.linalg.norm(baseline_same_lam)))
    if abs(cosine)>1+1e-10:
        raise AssertionError("impossible cosine")
    lr=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000).fit(Xt,y)
    mdm=MDM(metric="riemann").fit(C,y)
    if list(mdm.classes_)!=[0,1]:
        raise AssertionError("MDM class ordering not frozen")
    return inv,Xt,st,new_fit,lr,mdm,{
        "source_debiased_psd_trace":float(np.trace(new_fit.between_debiased_psd)),
        "raw_between_contrast_trace":new_fit.between_raw_trace,
        "finite_trial_contrast_noise_trace":new_fit.noise_removed_trace,
        "risk_selected_non_scalar_direction_cosine":cosine,
        "selected_matrix_operator_nontrivial":bool(
            r["alpha"]>0 and np.trace(new_fit.between_debiased_psd)>1e-12 and
            cosine<1-1e-9),
    }

def freeze(C,y,g,receipts,protocol):
    if set(np.unique(g))!=set(SOURCE_IDS):
        raise AssertionError("source IDs unexpected")
    base,new,choice=original_paths(C,y,g)
    inv,Xt,st,m,lr,mdm,metrics=prepare_source_models(C,y,g,choice)
    doc={
      "scope":"STRICT_DREYER2023A_FIRST_RUN_SOURCE_ONLY",
      "protocol_sha256":sha(protocol.read_bytes()),
      "source_ids":list(SOURCE_IDS),"target_ids_unread":list(TARGET_IDS),
      "channels":list(CH),"events":list(EVENTS),
      "source_epoch_receipts":receipts,
      "selected_choices":choice,
      "source_cv_summary":{
        "alpha0_macro_lambda":float(base["lambda_source_macro"]),
        "alpha0_macro_auc":float(new["baseline"]["macro_auc"]),
        "new_risk_selection":new["risk_best"],
        "new_mean_selection":new["mean_best"],
        "number_candidate_pairs":new["n_fit_candidate_pairs"],
        "beta":BETA,"alphas":list(ALPHAS)},
      "operator_source_geometry":metrics,
      "all_models_fit_before_held_target_io":True,
      "no_held_target_info_for_policy_selection":True,
      "not_independently_preregistered":True
    }
    return doc,(inv,Xt,st,m,lr,mdm)

def signflip(d):
    # Exact two sided enumeration 2^12 with one reference distribution.
    s=np.array(d)
    ids=np.arange(1<<len(s),dtype=np.uint32)
    signs=1-2*((ids[:,None]>>np.arange(len(s),dtype=np.uint32))&1).astype(np.int8)
    return float(np.mean(np.abs(signs@s)>=abs(float(s.sum()))-1e-12))

def evaluate(paradigm,dataset,model,Csrc,y,manifest):
    inv,Xtr,st,m,lr,mdm=model
    choices=manifest["selected_choices"]
    rows=[]
    for subject in TARGET_IDS:
        Ct,yt,receipt=person(paradigm,dataset,subject)
        X=old.tangent(Ct,inv)
        preds={name:score(m,X,rule["lambda"],rule["alpha"])
               for name,rule in choices.items()}
        # Exact original pooled-LDA path parity against V24 scoring.
        ref=old.scalar_scores(st,X,choices["source_macro_original"]["lambda"])
        if not np.allclose(ref,preds["source_macro_original"],atol=1e-9,rtol=1e-9):
            raise AssertionError("target baseline classifier path no longer original")
        for kind in ("LW","OAS"):
            scores,shrink=old.analytic_scores(Xtr,y,X,kind)
            preds[kind]=scores
        preds["tangent_LogReg"]=lr.decision_function(X)
        distances=mdm.transform(Ct)
        preds["riemannian_MDM"]=distances[:,0]-distances[:,1]
        values={k:float(roc_auc_score(yt,val)) for k,val in preds.items()}
        delta=values["new_risk_operator"]-values["source_macro_original"]
        rows.append({"subject":int(subject),"trials":int(len(yt)),
                     "auc":values,"primary_delta":delta,"receipt":receipt})
        print("V32_FROZEN_HELD_TARGET_AUC",subject,len(yt),flush=True)
    if set(r["subject"] for r in rows)!=set(TARGET_IDS):
        raise AssertionError("target missing")
    methods=list(rows[0]["auc"])
    d=np.array([r["primary_delta"] for r in rows])
    boot=d[np.random.default_rng(20261013).integers(0,len(d),(20000,len(d)))].mean(axis=1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    mean={k:float(np.mean([r["auc"][k] for r in rows])) for k in methods}
    return {
      "status":"V32_DREYER_ALL_12_HELD_TARGETS_COMPLETED",
      "source_choices_sha256":manifest["source_sha256"],
      "n_source_participants":len(SOURCE_IDS),
      "n_held_participants":len(TARGET_IDS),
      "n_held_trials":int(sum(r["trials"] for r in rows)),
      "all_helt_target_data_read_after_source_sha":True,
      "macro_auc_by_method":mean,
      "mean_primary_delta_new_matrix_risk_minus_original_scalar_macro":float(d.mean()),
      "descriptive_person_bootstrap95":ci,
      "exact_two_sided_group_signflip_p":signflip(d),
      "n_harmed_gt_002":int((d<-.020000000001).sum()),
      "n_benefited_gt_002":int((d>.020000000001).sum()),
      "n_pos":int((d>1e-12).sum()),
      "n_neg":int((d< -1e-12).sum()),
      "not_prospective_external_registration":True,
      "not_unconditional_target_guarantee":True,
      "all_held_person_outcomes":rows,
    }

def main(out):
    protocol=Path(__file__).resolve().parents[1]/PROTO
    if not protocol.exists():
        raise RuntimeError("no pre-outcome protocol")
    paradigm=old.MotorImagery(
        n_classes=2,events=list(EVENTS),channels=list(CH),
        fmin=8,fmax=30,resample=160,baseline=None,
        reject_by_annotation=False)
    data=Dreyer2023A()
    C,y,g,receipts=load_sources(paradigm,data)
    manifest,model=freeze(C,y,g,receipts,protocol)
    out.mkdir(parents=True,exist_ok=True)
    digest=save(manifest,out/SOURCE_FILE)
    print("V32_SOURCE_SHA_LOCKED_BEFORE_DREYER_HELD_IO",digest,flush=True)
    manifest["source_sha256"]=digest
    result=evaluate(paradigm,data,model,C,y,manifest)
    result["protocol_sha256"]=manifest["protocol_sha256"]
    save(result,out/TARGET_FILE)
    print("V32_REAL_HELD_SUMMARY",json.dumps({
      "status":result["status"],
      "mean_delta":result["mean_primary_delta_new_matrix_risk_minus_original_scalar_macro"],
      "exact_p":result["exact_two_sided_group_signflip_p"],
      "n_harmed_gt_002":result["n_harmed_gt_002"],
      "all_methods":result["macro_auc_by_method"],
      "structurally_novel":manifest["operator_source_geometry"]["selected_matrix_operator_nontrivial"],
      "selected":manifest["selected_choices"]["new_risk_operator"],
    },sort_keys=True),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    main(parser.parse_args().out)
