#!/usr/bin/env python3
"""AlexMI prospective-within-this-workflow source-choice lock, public MI EEG.

Held outcome IDs 7,8 are NEVER loaded before the source-only lock
file is written and sha256 recorded. This is a new repo-targeted
external acquisition (AlexMI right hand vs feet), but it is NOT
independently externally preregistered, two held people are ONE
acquisition dataset, and same pipeline family as old P1 studies.
The source NLL uses training group fitted class means, NOT validation
class-centred residuals. No target features or labels select lambda.
"""
import argparse,hashlib,json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from moabb.datasets import AlexMI
from moabb.paradigms import MotorImagery
from p1_second_model_shrinkage_lda_forecast import (
    fit, measure, source_validation, choice, GRID, old,
)

SOURCE_IDS=(1,2,3,4,5,6)
HELD_IDS=(7,8)
EVENTS=("right_hand","feet")
CHANNELS=("C3","Cz","C4")

def write(path,obj):
    payload=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

def load_one(ds,paradigm,sid):
    X,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
    X=np.asarray(X,dtype=np.float64);labels=np.asarray(labels)
    if X.ndim!=3 or X.shape[1]!=3 or len(X)!=40 or not np.isfinite(X).all():
        raise ValueError(f"AlexMI subject{sid} unexpected shape: {X.shape}")
    if set(labels.tolist())!=set(EVENTS):
        raise ValueError(f"unexpected AlexMI labels: {set(labels)}")
    y=np.asarray(labels=="right_hand",dtype=int)
    if int(y.sum())!=20:raise ValueError("imbalanced or wrong labelled trials")
    return old.covariances(X),y,{"public_id":sid,"trials":len(y),
            "positive":int(y.sum()),"channels":3}

def held_score(Csrc,ysrc,Ch,yh,selected):
    ref=old.reference(Csrc)
    xtr=old.tangent(Csrc,ref)
    xte=old.tangent(Ch,ref)
    macro=fit(xtr,ysrc,selected["source_macro"])
    proper=fit(xtr,ysrc,selected["source_predictive_nll"])
    fixed=fit(xtr,ysrc,0.1)
    m={ "source_macro": measure(macro,xte,yh)[0],
        "proper_predictive_NLL":measure(proper,xte,yh)[0],
        "fixed_lambda_0p1":measure(fixed,xte,yh)[0]}
    # Independent simple decoder control on same source-only tangent coordinates.
    logistic=make_pipeline(StandardScaler(),LogisticRegression(
                solver="liblinear",C=1.0,max_iter=2500,random_state=20261013))
    logistic.fit(xtr,ysrc)
    m["tangent_logistic_C1"]=float(roc_auc_score(yh,logistic.decision_function(xte)))
    return m

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    ds=AlexMI()
    paradigm=MotorImagery(n_classes=2,events=list(EVENTS),
                  channels=list(CHANNELS),fmin=8,fmax=30,
                  resample=160,baseline=None,reject_by_annotation=False)
    Csrc=[];ysrc=[];groups=[];receipts=[]
    # We do NOT request any held ID during source loading.
    for sid in SOURCE_IDS:
        C,y,rec=load_one(ds,paradigm,sid)
        Csrc.append(C);ysrc.append(y);groups.extend([sid]*len(y));receipts.append(rec)
        print("P1_ALEXMI_SOURCE_ONLY_PUBLIC_ID_LOADED",sid,flush=True)
    Csrc=np.concatenate(Csrc);ysrc=np.concatenate(ysrc)
    g=np.asarray(groups,dtype=int)
    validation,sids=source_validation(Csrc,ysrc,g)
    selected=choice(validation,sids)
    if list(sids)!=list(SOURCE_IDS):raise AssertionError("source contamination")
    info={"status":"PRE_HELD_7_8_FIXED_SOURCE_CHOICES",
          "dataset":"AlexMI_AlexandreMotorImagery",
          "dataset_DOI":"10.5281/zenodo.806022",
          "processing":"3 channels C3 Cz C4; right_hand vs feet; 8-30Hz; 160Hz",
          "source_public_ids":list(SOURCE_IDS),
          "held_public_ids_not_yet_read":list(HELD_IDS),
          "source_receipts":receipts,"source_parameters":selected,
          "fixed_LDA_lambda_0p1":.1,
          "extra_control":"tangent LogisticRegression C1",
          "common_grid":list(GRID),
          "no_target_labels_or_features_used_for_selection":True,
          "source_nll_uses_fitted_training_class_means":True,
          "not_externally_preregistered":True,
          "not_eight_independent_acquisitions":True,
          "repo_search_found_no_previous_AlexMI_code_but_not_global_proof_of_unseen":True}
    lock=write(args.out/"P1_ALEXMI_SOURCE_FROZEN_BEFORE_RESERVED_IDS.json",info)
    print("P1_ALEXMI_BEFORE_TARGET_SIGNAL_CHOICES_SHA256",lock,flush=True)
    rows=[]
    for sid in HELD_IDS:
        Ch,yh,receipt=load_one(ds,paradigm,sid)
        results=held_score(Csrc,ysrc,Ch,yh,selected)
        real=float(results["source_macro"]-results["proper_predictive_NLL"])
        forecast=float(selected["forecast_macro_minus_predictive_auc"])
        rows.append({"public_id":sid,"receipt":receipt,
          "auc":results,
          "actual_macro_minus_proper_NLL":real,
          "source_only_forecast_margin":forecast,
          "always_positive_sign_correct":bool(real>0),
          "source_forecast_sign_correct":bool(forecast*real>0),
          "no_target_optimized_models":True})
        print("P1_ALEXMI_RESERVED_HELD_NEW_OUTCOME",sid,
              json.dumps(rows[-1],sort_keys=True),flush=True)
    avg=lambda key:float(np.mean([r[key] for r in rows]))
    report={"status":"NEW_REPO_FIRST_ALEXMI_HELD_PUBLIC_ACQUISITION_DESCRIPTIVE_ONLY",
      "lock_SHA256":lock,"source_ids":list(SOURCE_IDS),"held_ids":list(HELD_IDS),
      "target_records_are_same_acquisition_dataset":True,
      "external_preregistration_was_not_performed":True,
      "no_TARGET_label_read_before_source_lock_within_workflow":True,
      "source_choice":selected,"per_held_person":rows,
      "summary":{"n_held_persons":len(rows),
        "mean_source_macro_minus_proper_NLL_AUC":avg("actual_macro_minus_proper_NLL"),
        "direction_correct_source_forecast":sum(r["source_forecast_sign_correct"] for r in rows),
        "direction_correct_always_positive":sum(r["always_positive_sign_correct"] for r in rows),
        "mean_source_macro_AUC":float(np.mean([r["auc"]["source_macro"] for r in rows])),
        "mean_proper_NLL_AUC":float(np.mean([r["auc"]["proper_predictive_NLL"] for r in rows])),
        "mean_fixed_lambda_0p1_AUC":float(np.mean([r["auc"]["fixed_lambda_0p1"] for r in rows])),
        "mean_tangent_logistic_C1_AUC":float(np.mean([r["auc"]["tangent_logistic_C1"] for r in rows]))},
      "not_sufficient_for_independently_confirmed_method_or_clinical_gain":True}
    digest=write(args.out/"P1_ALEXMI_TWO_RESERVED_NEW_REAL_EEG_RESULTS.json",report)
    print("P1_ALEXMI_NEW_REAL_EEG_SUMMARY",json.dumps(report["summary"],sort_keys=True),flush=True)
    print("P1_ALEXMI_NEW_REAL_EEG_RESULT_SHA256",digest,flush=True)
if __name__=="__main__":main()
