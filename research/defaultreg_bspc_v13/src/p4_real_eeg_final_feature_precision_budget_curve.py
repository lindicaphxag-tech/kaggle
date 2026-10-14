#!/usr/bin/env python3
"""REAL EEG final-tangent-feature quantization precision budget, exact rational.

Same PREVIOUSLY EXPOSED BNCI2014_001 person 1 64 trials/12 fixed
positive-negative pairs as existing P4 final-feature experiment.
No classification accuracy / true EEG acquisition error guarantee.
Compare representable grids b=18,14,10,8,6,4,2 bits and exact
entrywise eta=1/(2^(b+1)) through the same whole-lambda (0..1)
covariance-coordinate rational sign certificate. No source/target split
or independent clinical inference. Every quantization input bound is
verified against the original finite binary64 feature as a Fraction.
"""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as sp
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import p4_real_EEG_final_tangent_feature_rounding_certificate as base

BITS=(18,14,10,8,6,4,2)
EXPECTED_PAIR_COUNT=12
def quantize(X,bits):
    factor=1<<bits;eta=F(1,2*factor);mx=F(0);rows=[]
    for row in X:
        out=[]
        for value in row:
            v=F.from_float(float(value))
            q=F(round(v*factor),factor)
            err=abs(v-q)
            if err>eta:raise AssertionError("exact quantization bound broken")
            mx=max(mx,err)
            out.append(sp.Rational(q.numerator,q.denominator))
        rows.append(out)
    return sp.Matrix(rows),eta,mx

def run():
    ds=BNCI2014_001()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
        channels=["C3","C4"],fmin=8,fmax=30,resample=160,
        baseline=None,reject_by_annotation=False)
    X,label,_=p.get_data(dataset=ds,subjects=[1])
    X=np.asarray(X,dtype=np.float64);lab=np.asarray(label)
    if X.ndim!=3 or X.shape[1]!=2 or set(lab.tolist())!={"left_hand","right_hand"}:
        raise ValueError("Unexpected real EEG features")
    y=(lab=="right_hand").astype(int)
    n=base.POS_NEG_PER_CLASS
    take=np.r_[np.flatnonzero(y==0)[:n],np.flatnonzero(y==1)[:n]]
    if len(take)!=2*n:raise ValueError("not enough balanced EEG epochs")
    C=base.old.covariances(X[take])
    w=base.old.reference(C)
    features=np.asarray(base.old.tangent(C,w),dtype=np.float64)
    if features.shape!=(64,3):raise ValueError("3D only")
    rng=np.random.default_rng(20261013)
    pairs=list(zip(rng.choice(np.arange(n,2*n),EXPECTED_PAIR_COUNT,replace=False),
                   rng.choice(np.arange(n),EXPECTED_PAIR_COUNT,replace=False)))
    rec=[]
    for b in BITS:
        q,eta,mx=quantize(features,b)
        S,d,_=base.fit_pooled(q,[0]*n+[1]*n,F(1,5))
        es,ed,ev=base.error_bound(q,[0]*n+[1]*n,eta)
        counts=dict(POS=0,NEG=0,UNKNOWN=0)
        details=[]
        for k,(a,c) in enumerate(pairs):
            v=(q.row(int(a))-q.row(int(c))).T
            ans=base.certify(S,v,d,es,ev,ed,"0","1")
            label=ans["status"]
            counts[label]+=1
            details.append({"pair_k":k,"status":label,
              "score_margin":ans["whole_interval_base_margin_lower"],
              "error_bound":ans["whole_interval_score_error_upper"]})
        row={"grid_fractional_bits":b,
             "step":str(F(1,2**b)),"eta":str(eta),
             "observed_max_exact_roundoff":str(mx),
             "max_input_error_leq_eta":True,
             "counts":counts,
             "strictly_certified_over_all_lambda_and_input_box":
                 counts["POS"]+counts["NEG"],
             "pair_details":details}
        rec.append(row)
        print("P4_REAL_EEG_FEATURE_PRECISION_BUDGET",json.dumps(
           {"bits":b,"counts":counts,"certified":row[
            "strictly_certified_over_all_lambda_and_input_box"]},sort_keys=True),flush=True)
    assert rec[0]["counts"]==dict(POS=8,NEG=4,UNKNOWN=0)
    assert all(sum(v["counts"].values())==12 for v in rec)
    return {"status":"EXACT_REAL_FINAL_FEATURE_QUANTIZATION_PRECISION_CURVE",
            "dataset":"Previously analysed BNCI2014_001 released ID1",
            "trial_count":64,"sample_pair_count":12,
            "precision_bits_list":list(BITS),"precision_rows":rec,
            "same_prior_seed_and_pair_indices":True,
            "finite_binary64_final_feature_error_verified_exactly":True,
            "no_roundoff_bound_through_EEG_FILTERING_OR_TANGENT":True,
            "not_cross_subject_decoding_accuracy":True,
            "not_independent_new_acquisition":True}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();j=run()
    raw=json.dumps(j,indent=2,sort_keys=True)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(raw)
    print("P4_PRECISION_BUDGET_CERTIFICATE_SHA256",hashlib.sha256(raw.encode()).hexdigest())
