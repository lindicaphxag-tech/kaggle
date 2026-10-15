#!/usr/bin/env python3
"""Real but already exposed BNCI2014_001 nine-person source-forecast action audit.
Forecast thresholds are POST-HOC and do not create prospective utility claims.
"""
import argparse,hashlib,json
from decimal import Decimal as D
from pathlib import Path

THRESHOLDS=("0","0.0025","0.005","0.01","0.02")
def mean(xs):return sum(xs,D(0))/D(len(xs))
def audit(j):
    if not j["all_subject_outcomes_previously_exposed_projectwide"]:
        raise ValueError("outcome chronology is mandatory")
    rows=j["rows"]
    if len(rows)!=9 or [x["target_id"] for x in rows]!=list(range(1,10)):
        raise ValueError("wrong immutable scalar artifact")
    p=[x["source_forecast_macro_minus_predictive_auc"] for x in rows]
    y=[x["observed_macro_minus_predictive_auc"] for x in rows]
    if any(x<=0 for x in p):
        raise ValueError("trivial-direction control premise changed")
    baseline=mean(y)
    policies=[]
    for t in THRESHOLDS:
        threshold=D(t)
        realized=[v if f>threshold else D(0) for f,v in zip(p,y)]
        relative=[x-v for x,v in zip(realized,y)]
        policies.append({
            "posthoc_forecast_threshold":t,
            "macro_policy_selections":sum(z>threshold for z in p),
            "mean_realized_auc_delta_over_proper":str(mean(realized)),
            "mean_realized_auc_delta_over_always_macro":str(mean(relative)),
            "records_worse_than_macro_by_more_than_0_005":
                sum(z< -D("0.005") for z in relative)})
    assert policies[0]["macro_policy_selections"]==9
    assert policies[2]["macro_policy_selections"]==5
    assert D(policies[2]["mean_realized_auc_delta_over_always_macro"])<0
    return {"status":"POST_OUTCOME_NINE_HELD_EEG_DECISION_BASELINE_ONLY",
        "previously_exposed_subjects":9,"all_nine_forecasts_positive":True,
        "actual_positive_subjects":sum(z>0 for z in y),
        "always_macro_vs_proper_mean_auc":str(baseline),
        "postoutcome_per_subject_oracle_gain_vs_proper":
            str(mean([max(z,D(0)) for z in y])),
        "thresholds_chosen_after_target_outcomes":True,
        "no_new_prospective_EEG_acquisition":True,
        "threshold_policies":policies}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    arg=ap.parse_args();raw=arg.input.read_bytes()
    r=audit(json.loads(raw,parse_float=D))
    r["frozen_input_SHA256"]=hashlib.sha256(raw).hexdigest()
    data=json.dumps(r,sort_keys=True,indent=2)+"\n"
    arg.out.parent.mkdir(parents=True,exist_ok=True)
    arg.out.write_text(data)
    print("P1_EXPOSED_FORECAST_ACTION_NULL",json.dumps(r,sort_keys=True))
