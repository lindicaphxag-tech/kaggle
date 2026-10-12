#!/usr/bin/env python3
"""BNCI2014_002 native 15-channel SOURCE grouped anatomy, post-exposure only.

This dataset has already served as an OUTER TARGET elsewhere; it is
not a fresh target replication or a new held-out result here.
"""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as o
from exact_prediction_shift_partition import source_group_partition

def main(out):
    C,y,g,receipts,spec,n_channels,names=o.load_dataset("BNCI2014_002")
    if set(map(int,np.unique(g)))!=set(range(1,15)):
        raise AssertionError("original BNCI source IDs changed")
    if n_channels!=15:
        raise AssertionError("native original 15-channel montage changed")
    original=o.source_path(C,y,g)
    report=source_group_partition(C,y,g,original["feasible_full_lambdas"])
    s=report["selection_summary"]
    if s["source_macro_auc"]!=original["lambda_source_macro"]:
        raise AssertionError("original 27-point macro source path differs")
    if s["historical_class_centered"]!=original["lambda_covariance_full"]:
        raise AssertionError("original historical class-centred path differs")
    summary={}
    for criterion,lam in s.items():
        v=next(r for r in report["rows"] if r["lambda"]==lam)
        loc=v["auc_invariant_translation_contribution"]
        aligned=v["contrast_shift_score_aligned_contribution"]
        ortho=v["contrast_shift_score_orthogonal_contribution"]
        total=loc+aligned+ortho
        summary[criterion]={
            "lambda":lam,"held_source_macro_auc":v["source_macro_auc"],
            "mean_logscore_gap":v["source_mean_shift_gap"],
            "location":loc,"score_aligned":aligned,"score_orthogonal":ortho,
            "auc_invariant_location_plus_orthogonal_fraction":
                (loc+ortho)/total if total>1e-12 else None,
            "location_fraction":loc/total if total>1e-12 else None,
        }
    output={
      "status":"SOURCE_ONLY_NATIVE_BNCI_POST_OUTCOME_DIAGNOSTIC",
      "dataset":"BNCI2014_002","n_source_people":14,
      "n_source_trials":int(len(y)),"n_eeg_channels":n_channels,
      "tangent_feature_dimension":n_channels*(n_channels+1)//2,
      "source_ids":list(range(1,15)),
      "not_independent_target_validation":True,
      "outer_target_results_previously_exposed":True,
      "original_source_27_grid_parity":True,
      "group_receipts":receipts,
      "selected_rows":summary,
      "all_grid_rows":report["rows"],
    }
    out.mkdir(parents=True,exist_ok=True)
    f=out/"BNCI2014_002_NATIVE_15CHANNEL_SOURCE_THREEFACTOR.json"
    raw=(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    f.write_bytes(raw)
    print("BNCI_NATIVE_TWOFACTOR_SHA",hashlib.sha256(raw).hexdigest(),flush=True)
    print("BNCI_NATIVE_TWOFACTOR_MAIN",json.dumps(
        {k:output[k] for k in ("n_source_people","n_source_trials",
          "n_eeg_channels","tangent_feature_dimension","selected_rows")},
        sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    main(p.parse_args().out)
