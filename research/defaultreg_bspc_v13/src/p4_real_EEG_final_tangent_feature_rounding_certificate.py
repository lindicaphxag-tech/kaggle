#!/usr/bin/env python3
"""REAL EEG final-feature fixed-point quantization sign certificate, P4.

All results are numerical representation comparisons on publicly exposed
BNCI2014_001 released participant 1; NOT independent EEG acquisition,
cross-subject AUC gain or rounding guarantee through MNE filtering,
covariance computation, eigendecomposition or tangent mapping.

The only rigorously controlled perturbation is exact conversion of the
finite binary64 FINAL tangent feature values into rational grid coordinates.
The input feature errors are VERIFIED entry-by-entry as exact fractions,
not assumed from physical EDF calibration. Pair signs are certified for
all points in [0,1] with the separate rational covariance resolvent.
"""
import argparse,hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as sp
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from p4_covariance_coordinate_resolvent_bound import certify
from p4_final_feature_entrywise_error_to_covariance_certificate import (
    fit_pooled,error_bound,
)

DATA_SUBJECT=1
POS_NEG_PER_CLASS=32
SCALING=1<<18
ETA=F(1,2*SCALING)
PAIRS=12

def exact_double(z):
    assert isinstance(z,float)
    return F.from_float(z)

def quantize_features_float(X):
    if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():
        raise ValueError("expected valid 3D final tangent features")
    quant=[]
    max_err=F(0)
    for row in X:
        vec=[]
        for raw in row:
            val=exact_double(float(raw))
            qi=round(val*SCALING) # Fraction -> EXACT nearest integer
            q=F(qi,SCALING)
            diff=abs(val-q)
            if diff>ETA:raise AssertionError("input bound violated")
            max_err=max(max_err,diff)
            vec.append(sp.Rational(q.numerator,q.denominator))
        quant.append(vec)
    return sp.Matrix(quant),max_err

def run():
    ds=BNCI2014_001()
    paradigm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
       channels=["C3","C4"],fmin=8,fmax=30,resample=160,
       baseline=None,reject_by_annotation=False)
    X,label,_=paradigm.get_data(dataset=ds,subjects=[DATA_SUBJECT])
    X=np.asarray(X,dtype=np.float64);label=np.asarray(label)
    if X.ndim!=3 or X.shape[1]!=2 or set(label.tolist())!=set(["left_hand","right_hand"]):
        raise ValueError("wrong expected two-channel public EEG input")
    y=np.asarray(label=="right_hand",dtype=int)
    idx0=np.flatnonzero(y==0)[:POS_NEG_PER_CLASS]
    idx1=np.flatnonzero(y==1)[:POS_NEG_PER_CLASS]
    if len(idx0)!=POS_NEG_PER_CLASS or len(idx1)!=POS_NEG_PER_CLASS:
        raise ValueError("need exactly 32 per class")
    take=np.r_[idx0,idx1]
    C=old.covariances(X[take])
    ref=old.reference(C)
    feat=np.asarray(old.tangent(C,ref),dtype=np.float64)
    if feat.shape!=(2*POS_NEG_PER_CLASS,3):
        raise ValueError("expected three tangent coordinates from two EEG electrodes")
    q,maxerr=quantize_features_float(feat)
    classes=[0]*POS_NEG_PER_CLASS+[1]*POS_NEG_PER_CLASS
    ridge=F(1,5)
    S,d,_=fit_pooled(q,classes,ridge)
    epsS,epsD,epsV=error_bound(q,classes,ETA)
    # Pair identities fixed WITHOUT observing ANY certification decisions.
    rng=np.random.default_rng(20261013)
    pair_indices=[(int(a),int(b)) for a,b in zip(
        rng.choice(np.arange(POS_NEG_PER_CLASS,2*POS_NEG_PER_CLASS),PAIRS,replace=False),
        rng.choice(np.arange(POS_NEG_PER_CLASS),PAIRS,replace=False))]
    count={"POS":0,"NEG":0,"UNKNOWN":0}
    records=[]
    for k,(positive,negative) in enumerate(pair_indices):
        v=(q.row(positive)-q.row(negative)).T
        z=certify(S,v,d,epsS,epsV,epsD,"0","1")
        outcome=z["status"]
        count[outcome]+=1
        records.append({"sample_pair_precommitted_index":k,
             "positive_trial_local_idx":positive,
             "negative_trial_local_idx":negative,
             "conditional_whole_lambda_sign":outcome,
             "exact_rational_base_margin_lower":z.get("whole_interval_base_margin_lower"),
             "exact_rational_uncertainty_upper":z.get("whole_interval_score_error_upper")})
        print("P4_REAL_FINAL_TANGENT_QUANTIZATION_PAIR",k,outcome,flush=True)
    if len(records)!=PAIRS or sum(count.values())!=PAIRS:
        raise AssertionError("pair accounting")
    return {"status":"REAL_PUBLIC_EEG_FINAL_FEATURE_QUANTIZATION_BOX_AUDIT",
       "source_dataset":"BNCI2014_001 public EEG, participant 1, previously exposed",
       "processing":"2 EEG electrodes C3/C4, 8-30Hz, 160Hz, 3 tangent features",
       "binary_task":"left vs right hand imagery",
       "n_per_class":POS_NEG_PER_CLASS,
       "n_final_features_per_epoch":3,
       "sampled_precommitted_pair_count":PAIRS,
       "machine_input_is_exactly_binary64_feature_output_not_physically_true_EEG":True,
       "quantization_grid_resolution":str(F(1,SCALING)),
       "max_verified_exact_binary64_to_rational_grid_abs_error":str(maxerr),
       "rigorously_verified_entrywise_bound_eta":str(ETA),
       "pooled_covariance_ridge":str(ridge),
       "epsilon_S":str(epsS),"epsilon_d":str(epsD),"epsilon_v":str(epsV),
       "whole_lambda_interval":[0,1],
       "counts":count,"pair_results":records,
       "input_error_box_checked_as_exact_binary64_fractions":True,
       "no_end_to_end_EEG_pipeline_error_certification":True,
       "no_new_unseen_EEG_acquisition_or_decoder_gain":True,
       "no_patient_identity_claim":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    z=run();data=json.dumps(z,sort_keys=True,indent=2)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(data)
    print("P4_REAL_FINAL_EEG_FEATURE_QUANTIZATION_CERTIFICATE",json.dumps(
        {"status":z["status"],"counts":z["counts"],
         "epsilon":z["rigorously_verified_entrywise_bound_eta"]},sort_keys=True))
    print("P4_REAL_FEATURE_CERTIFICATE_SHA256",hashlib.sha256(data.encode()).hexdigest())
