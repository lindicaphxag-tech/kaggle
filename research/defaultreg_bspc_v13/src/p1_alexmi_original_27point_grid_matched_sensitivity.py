#!/usr/bin/env python3
"""RETROSPECTIVE, grid-matched independent LDA AlexMI sensitivity.
Reuses ALREADY EXPOSED AlexMI public IDs7-8, which have outcomes
in a prior 8-grid run. Thus only sensitivity to the full
27-point FULL_GRID is measured; NOT another external confirmation.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import AlexMI
from moabb.paradigms import MotorImagery
import p1_second_model_shrinkage_lda_forecast as lda
from p1_alexmi_new_source_lock_two_held import load_one,held_score

SRC=(1,2,3,4,5,6)
HELD=(7,8)

def dump(path,obj):
    b=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def main(folder):
    grid=tuple(lda.old.FULL_GRID)
    if len(grid)!=27 or len(set(grid))!=27:raise AssertionError("not canonical 27-grid")
    lda.GRID=grid
    ds=AlexMI()
    paradigm=MotorImagery(n_classes=2,events=["right_hand","feet"],
                channels=["C3","Cz","C4"],fmin=8,fmax=30,
                resample=160,baseline=None,reject_by_annotation=False)
    c=[];y=[];groups=[];receipts=[]
    for sid in SRC:
        C,yy,rec=load_one(ds,paradigm,sid)
        c.append(C);y.append(yy);groups.extend([sid]*len(yy));receipts.append(rec)
    c=np.concatenate(c);y=np.concatenate(y);g=np.asarray(groups,dtype=int)
    validation,ids=lda.source_validation(c,y,g)
    selected=lda.choice(validation,ids)
    original={"dataset":"AlexMI","source_ids":list(SRC),
        "previously_exposed_target_ids_not_yet_read_in_THIS_run":list(HELD),
        "grid":list(grid),"source_decisions":selected,"source_receipts":receipts,
        "historical_eight_point_alexmi_results_already_exposed":True,
        "full_grid_matches_original_P1_27_point_family":True,
        "not_new_independent_external_validation":True}
    frozen=dump(folder/"P1_ALEXMI_27_GRID_SOURCE_REPLAY_LOCK.json",original)
    print("P1_ALEXMI_FULL_GRID_SOURCE_LOCK_SHA",frozen,flush=True)
    rows=[]
    for sid in HELD:
        C,yy,rec=load_one(ds,paradigm,sid)
        vals=held_score(c,y,C,yy,selected)
        margin=vals["source_macro"]-vals["proper_predictive_NLL"]
        rows.append({"held_id":sid,"auc":vals,"macro_minus_proper_nll":margin,
                     "same_person_previously_exposed_in_8grid":True})
        print("P1_ALEXMI_27_GRID_HELD_OUTCOME",sid,
              json.dumps(rows[-1],sort_keys=True),flush=True)
    means={name:float(np.mean([r["auc"][name] for r in rows]))
           for name in rows[0]["auc"]}
    result={"status":"PREVIOUSLY_EXPOSED_ALEXMI_27_GRID_SENSITIVITY",
      "original_source_lock_sha256":frozen,
      "comparison_to_8grid_posthoc":True,
      "held_people_not_new_in_project":True,
      "n_held_people":2,"distinct_acquisitions":1,
      "source_choices":selected,"source_grid":list(grid),
      "rows":rows,"means":means,
      "macro_minus_proper_nll_mean":
         float(np.mean([r["macro_minus_proper_nll"] for r in rows])),
      "can_make_person_specific_forecast_gain_claim":False}
    sha=dump(folder/"P1_ALEXMI_CANONICAL_27GRID_REAL_RESULTS.json",result)
    print("P1_ALEXMI_FULL_GRID_REAL_SOURCE_NO_NOVELTY",
          json.dumps({"means":means,"delta":result["macro_minus_proper_nll_mean"],
                      "source_choices":selected},sort_keys=True),flush=True)
    print("P1_ALEXMI_FULL_GRID_RESULT_SHA",sha,flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--out",type=Path,required=True)
    x=a.parse_args();main(x.out)
