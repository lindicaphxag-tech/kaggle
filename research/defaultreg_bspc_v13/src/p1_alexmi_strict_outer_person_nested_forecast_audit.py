#!/usr/bin/env python3
"""Strict NESTED source-person forecast with zero held-person training paths.

Original p1_second_model_shrinkage_lda_forecast.choice derives out-of-fold
source-person forecast using validation rows from models trained on the
person held out of that OUTER source forecast. This routine prevents
that source-person re-entry by entirely removing source h BEFORE
retraining source-covariance geometry, inner CV λ choice and classifier.

AlexMI IDs 1..6 are previously source-used, targets 7..8 already exposed.
This is a STRICTER RETROSPECTIVE source-forecast reanalysis, NOT a
fresh confirmatory dataset or independent performance gain claim.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from moabb.datasets import AlexMI
from moabb.paradigms import MotorImagery
import p1_second_model_shrinkage_lda_forecast as lda
from p1_alexmi_new_source_lock_two_held import load_one

SOURCE=(1,2,3,4,5,6)
TARGETS=(7,8)

def dump(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()

def select_from_validation(validation,ids):
    macro=max(lda.GRID,key=lambda l:(np.mean([auc for sid,auc,nll in validation[l] if sid in ids]),l))
    proper=min(lda.GRID,key=lambda l:(np.mean([nll for sid,auc,nll in validation[l] if sid in ids]),l))
    return macro,proper

def strictly_held_forecast(C,y,g):
    ids=tuple(sorted(set(g.tolist())))
    assert ids==SOURCE
    out=[]
    for h in ids:
        keep=g!=h
        c0=C[keep];y0=y[keep];g0=g[keep]
        # Every inner training fold, geometry, and validation λ score
        # excludes outer held h completely.
        inner,inner_ids=lda.source_validation(c0,y0,g0)
        assert h not in inner_ids
        a,b=select_from_validation(inner,set(inner_ids))
        inv=lda.old.reference(c0)
        Xt=lda.old.tangent(c0,inv)
        Xh=lda.old.tangent(C[~keep],inv)
        M=lda.fit(Xt,y0,a);N=lda.fit(Xt,y0,b)
        auc_a=lda.measure(M,Xh,y[~keep])[0]
        auc_b=lda.measure(N,Xh,y[~keep])[0]
        out.append({"outer_held_source_person":int(h),
                    "inner_subjects":list(map(int,inner_ids)),
                    "inner_macro_choice":a,"inner_proper_choice":b,
                    "out_of_person_macro_auc":auc_a,
                    "out_of_person_proper_auc":auc_b,
                    "paired_delta":auc_a-auc_b,
                    "held_person_excluded_from_all_inner_model_training":True,
                    "outer_held_person_used_once_for_source_person_forecast_calibration":True})
        print("P1_NESTED_SOURCE_PERSON",json.dumps(out[-1],sort_keys=True),flush=True)
    return out

def main(path):
    lda.GRID=tuple(lda.old.FULL_GRID)
    assert len(lda.GRID)==27
    ds=AlexMI();p=MotorImagery(n_classes=2,events=["right_hand","feet"],
        channels=["C3","Cz","C4"],fmin=8,fmax=30,resample=160,
        baseline=None,reject_by_annotation=False)
    c=[];y=[];groups=[];receipts=[]
    for sid in SOURCE:
        Ci,yi,meta=load_one(ds,p,sid)
        c.append(Ci);y.append(yi);groups.extend([sid]*len(yi))
        receipts.append(meta)
    C=np.concatenate(c);y=np.concatenate(y);g=np.asarray(groups)
    nested=strictly_held_forecast(C,y,g)
    inner,ids=lda.source_validation(C,y,g)
    a,b=select_from_validation(inner,set(ids))
    earlier=lda.choice(inner,ids)
    # New forecast does not use the methodologically invalid
    # earlier crossfit of validation folds trained on outer h.
    good=float(np.mean([r["paired_delta"] for r in nested]))
    prior=float(earlier["forecast_macro_minus_predictive_auc"])
    pre={"status":"P1_FULLY_NESTED_SOURCE_PERSON_FORECAST_BEFORE_TARGET_REREAD",
         "source_ids":list(SOURCE),"already_exposed_target_ids_not_yet_reloaded":list(TARGETS),
         "27_grid":list(lda.GRID),"source_records":receipts,
         "nested_source_person_forecast":nested,
         "source_decision":{"source_macro":a,"proper_predictive_nll":b},
         "strict_nested_forecast_macro_minus_proper":good,
         "former_leaky_row_crossfit_forecast_for_audit_only":prior,
         "former_choice_method_source_outer_h_used_in_other_fold_training":True,
         "outer_subject_never_in_inner_cv_model_train_or_validation":True,
         "strict_nested_predictions_target_blind":True,
         "all_source_and_target_experiment_previously_exposed_projectwide":True,
         "not_external_preregistration":True}
    sha=dump(path/"P1_ALEXMI_STRICT_NESTED_SOURCE_ONLY_FORECAST_BEFORE_REREAD.json",pre)
    print("P1_STRICT_SOURCE_NESTED_FORECAST_SHA",sha,flush=True)
    rows=[]
    inv=lda.old.reference(C);Xt=lda.old.tangent(C,inv)
    mm=lda.fit(Xt,y,a);nn=lda.fit(Xt,y,b)
    for h in TARGETS:
        Ci,yi,_=load_one(ds,p,h)
        Xh=lda.old.tangent(Ci,inv)
        av=lda.measure(mm,Xh,yi)[0];bv=lda.measure(nn,Xh,yi)[0]
        delta=av-bv
        rows.append({"public_held_id":h,"macro_auc":av,"proper_auc":bv,
            "observed_delta":delta,"strict_nested_forecast":good,
            "old_forecast_for_comparison_only":prior,
            "signed_correct_strict":bool(good*delta>0),
            "signed_correct_trivial_positive":bool(delta>0)})
    post={"status":"EXPOSED_ALEXMI_STRICTLY_NESTED_SOURCE_FORECAST_AUDIT",
          "source_lock_sha256":sha,
          "source_decisions":{"macro":a,"proper_nll":b},
          "nested_person_oof_forecast":good,
          "old_contaminated_oof_estimator_forecast":prior,
          "n_held_people":2,"one_named_acquisition":True,
          "not_independent_experiment_or_new_target_outcome":True,
          "two_held_outcomes":rows}
    h=dump(path/"P1_ALEXMI_STRICT_NESTED_SOURCE_FORECAST_RETROSPECTIVE_RESULT.json",post)
    print("P1_NESTED_SOURCE_FORECAST_FINAL",json.dumps(post,sort_keys=True),flush=True)
    print("P1_NESTED_SOURCE_FORECAST_RESULT_SHA",h,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();main(args.out)
