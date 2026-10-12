"""V25 real 3-channel Weibo2014 SOURCE-fold Bernstein rank-cell audit.

This is retrospective mechanism-only analysis of an ALREADY EXPOSED cohort:
the first-run Weibo2014 target outcomes were evaluated in V24. We do NOT
relabel this as independent external or pre-data confirmation. The original
Z0 source-only selection and tangent-space model are reused unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

import original_frozen_runner as old
from run_new_weibo_v24 import load_public, source_models
from bernstein_rank_cells_v25 import bernstein_rank_path
from spectral_path_certificate_v23 import certified_path_rank_stability


def audit(outer_person=1, segments=12):
    if outer_person not in range(1,11):
        raise ValueError("only original Weibo2014 10 subjects")
    C,y,g,receipt=load_public()
    source = g!=outer_person
    Cs,ys,gs=C[source],y[source],g[source]
    path=old.source_path(Cs,ys,gs)
    la=float(path["lambda_covariance_full"])
    lb=float(path["lambda_source_macro"])
    # Also audit full (0,1) as a fixed, selection-independent comparator.
    intervals={"full":[0.,1.]}
    if la!=lb:
        intervals["between_source_choices"]=[min(la,lb),max(la,lb)]
    models=source_models(Cs,ys,gs)
    if len(models)!=9:
        raise RuntimeError("source validation person denominator drift")
    results={}
    for name,(low,high) in intervals.items():
        rows=[]
        for state,X,yy in models:
            if len(state[0])!=6:
                raise RuntimeError("fixed 3ch EEG tangent-space dimension must be 6")
            new=bernstein_rank_path(state,X,yy,low,high,segments=segments)
            base=certified_path_rank_stability(
                state,X,yy,low,high,segments=segments)
            if abs(new["observed_endpoint_auc_low"]-
                   base["observed_endpoint_auc_low"])>1e-12 or abs(
                   new["observed_endpoint_auc_high"]-
                   base["observed_endpoint_auc_high"])>1e-12:
                raise AssertionError("pairwise scores no longer match frozen spectral path")
            rows.append({
                "old_V23_certified_fraction":base["certified_stable_fraction"],
                "new_V25_bernstein_certified_fraction":new["certain_fixed_sign_fraction"],
                "delta_stability_certificate": (
                    new["certain_fixed_sign_fraction"]-
                    base["certified_stable_fraction"]
                ),
                "v25_auc_interval":[new["uniform_path_auc_lower_bound"],
                                    new["uniform_path_auc_upper_bound"]],
                "source_trial_pair_count":new["source_trial_pair_count"],
                "uncertain_pairs_v25":new["uncertain_pairs"],
            })
        results[name]={
            "mean_old_v23_stable_fraction":float(np.mean([
                x["old_V23_certified_fraction"] for x in rows])),
            "mean_new_v25_stable_fraction":float(np.mean([
                x["new_V25_bernstein_certified_fraction"] for x in rows])),
            "mean_stability_difference":float(np.mean([
                x["delta_stability_certificate"] for x in rows])),
            "number_source_people_improved":sum(
                x["delta_stability_certificate"]>1e-10 for x in rows),
            "number_source_people_worse":sum(
                x["delta_stability_certificate"] < -1e-10 for x in rows),
            "n_source_people":len(rows),
            "per_source_person":rows,
        }
    return {
        "status":"RETROSPECTIVE_V25_SOURCE_BERNSTEIN_RANK_AUDIT",
        "dataset":"Weibo2014",
        "outer_held_person":int(outer_person),
        "source_people":9,
        "source_spectral_dimension":6,
        "selected_historical_lambda":la,
        "selected_macro_lambda":lb,
        "subinterval_count":segments,
        "not_unseen_dataset_or_target_performance_evidence":True,
        "target_features_or_labels_not_passed_to_source_diagnostic":True,
        "conditional_numerical_certificates_not_formally_verified":True,
        "interval_results":results,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--outer-person",type=int,default=1)
    parser.add_argument("--segments",type=int,default=12)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    result=audit(a.outer_person,a.segments)
    body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(body)
    print(json.dumps({
        "status":result["status"],
        "full_interval":{k:v for k,v in result["interval_results"]["full"].items()
                         if k!="per_source_person"},
        "source_selected_interval":{k:v for k,v in result["interval_results"].get(
            "between_source_choices",{}).items() if k!="per_source_person"},
        "receipt_sha256":hashlib.sha256(body.encode()).hexdigest()
    },indent=2))


if __name__=="__main__":
    main()
