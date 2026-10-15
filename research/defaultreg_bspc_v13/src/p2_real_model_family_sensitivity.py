#!/usr/bin/env python3
"""Retrospective P2 model-family sensitivity on Liu2024 public EEG.

Outputs only aggregate and held-content-group paired AUC summaries;
NEVER exports EEG samples, event sequences or patient metadata.
Identity grouping examines all test waveforms: NOT prospective Z0.
Target labels are read only by the final scorer, not fit/tune stages.
Requires MOABB==1.7.2 (same processing as P2 primary manuscript).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/"defaultreg_proper_predictive"))
from record_identity_groups_v26 import assign_recording_groups,check_disjoint_training
import original_frozen_runner as old

METHODS=("logistic_C1","ridge_alpha10")
def classifier(name):
    if name=="logistic_C1":
        return make_pipeline(StandardScaler(),LogisticRegression(
            C=1.,solver="liblinear",max_iter=1800,random_state=20261013))
    if name=="ridge_alpha10":
        return make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.))
    raise ValueError("unknown frozen classifier")

def fit_scores(C, y, mask, held, algo):
    # Reference is ALWAYS computed from training source, never target.
    ref=old.reference(C[mask])
    Xtrain=old.tangent(C[mask],ref)
    Xtest=old.tangent(C[held],ref)
    model=classifier(algo)
    model.fit(Xtrain,y[mask])
    score=np.asarray(model.decision_function(Xtest),dtype=float)
    if len(score)!=int(held.sum()) or not np.isfinite(score).all():
        raise ValueError("invalid held scores")
    return score

def run():
    ds=Liu2024()
    pr=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                    fmin=8,fmax=30,resample=160,baseline=None,
                    reject_by_annotation=False)
    records=[];Cs=[];ys=[];ids=[]
    for sid in range(1,51):
        X,labels,_=pr.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        labels=np.asarray(labels)
        y=np.asarray(labels=="right_hand",dtype=np.uint8)
        if (X.ndim!=3 or X.shape[0]!=40 or X.shape[1]!=29
                or len(y)!=40 or int(y.sum())!=20
                or not np.isfinite(X).all()):
            raise RuntimeError("Liu2024 schema/preprocessing drift: "+str(sid))
        records.append({"subject":sid,"signal":X,"labels":y})
        Cs.append(old.covariances(X))
        ys.append(y);ids.extend([sid]*len(y))
        print("P2_FIXED_PUBLIC_INPUT_CHECKED",sid,flush=True)
    identity,groups=assign_recording_groups(records)
    if identity["duplicate_groups"]!=[[4,5]] or identity["independent_processed_record_count"]!=49:
        raise AssertionError("unanticipated recorded-content identity: stop before reporting")
    C=np.concatenate(Cs); y=np.concatenate(ys); ids=np.asarray(ids)
    del records,Cs,ys
    rows=[]
    reps=list(identity["representative_ids"])
    for target in reps:
        held=ids==target
        src_old=ids!=target
        src_clean=(ids!=target)&(ids!=5)
        safe=list(map(int,np.unique(ids[src_clean])))
        check_disjoint_training(identity,target,safe)
        if len(safe)!=48 or int(held.sum())!=40 or int(src_old.sum())!=1960 or int(src_clean.sum())!=1920:
            raise AssertionError("training/held record group count mismatch")
        r={"record_group":int(target),
           "overlap_affected_target_group":bool(target==4),
           "outcomes":{}}
        for algo in METHODS:
            pred_old=fit_scores(C,y,src_old,held,algo)
            pred_clean=fit_scores(C,y,src_clean,held,algo)
            a=float(roc_auc_score(y[held],pred_old))
            b=float(roc_auc_score(y[held],pred_clean))
            r["outcomes"][algo]={"original_fit_auc":a,"purged_fit_auc":b,"paired_delta":b-a}
        rows.append(r)
        print("P2_SAME_TARGET_MODEL_PAIR_COMPLETE",target,flush=True)
    report={"status":"NEW_RETROSPECTIVE_MODEL_CLASS_FIT_SENSITIVITY",
       "data":"Liu2024 public processed digital EEG",
       "processing":"MOABB 1.7.2 / 8-30Hz / 160Hz / 29 EEG channels",
       "scope":"already exposed acquisition; model-family exploratory robustness",
       "not_strict_Z0_due_global_signal_identity_discovery":True,
       "not_new_prospective_cohort":True,"not_patient_identity_or_causal_leakage":True,
       "models_have_frozen_fixed_hyperparameters":list(METHODS),
       "n_held_content_groups":49,
       "same_held_content_for_original_and_purged":True,
       "group_overlap_target":4,
       "summary":{},"rows":rows}
    for algo in METHODS:
        z=np.asarray([r["outcomes"][algo]["paired_delta"] for r in rows])
        out=np.asarray([r["outcomes"][algo]["original_fit_auc"] for r in rows])
        clean=np.asarray([r["outcomes"][algo]["purged_fit_auc"] for r in rows])
        assert np.isfinite(z).all()
        report["summary"][algo]={
           "mean_original_fit_auc":float(out.mean()),"mean_purged_fit_auc":float(clean.mean()),
           "all49_mean_change":float(z.mean()),"overlap_group_change":float(z[reps.index(4)]),
           "unaffected48_mean_change":float(np.delete(z,reps.index(4)).mean()),
           "median_change":float(np.median(z)),
           "n_harm_over_002":int((z<-.020000001).sum()),
           "n_help_over_002":int((z>.020000001).sum()),
           "n_positive":int((z>0).sum()),"n_negative":int((z<0).sum())}
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.write_text(payload,encoding="utf-8")
    print("P2_MODEL_CLASS_SENSITIVITY_FINISHED",
          json.dumps(result["summary"],sort_keys=True),flush=True)
    print("SHA256",hashlib.sha256(payload.encode()).hexdigest(),flush=True)

if __name__=="__main__":main()
