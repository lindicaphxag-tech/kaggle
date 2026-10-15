#!/usr/bin/env python3
"""Audit two ORIGINAL previously exposed GitHub Actions EEG result artifacts.
Target records: BNCI2014_004 ID9, Weibo2014 IDs9/10. Same 5 policies
per acquisition, but source models differ across acquisitions.
No fresh held-out EEG, no inference across 3 independent acquisitions.
"""
import argparse,hashlib,json
from pathlib import Path

METHODS=("source_macro","continuous","SCE","anisotropic","group_centered")

def audit(b,w):
    if b["status"]!="EXECUTED_RETROSPECTIVE_FOUR_OPERATOR_COMMON_SOURCE_BNCI004":
        raise ValueError("not original BNCI result")
    if w["status"]!="EXECUTED_RETROSPECTIVE_SECOND_ACQUISITION_COMMON_SOURCE_FOUR_OPERATORS":
        raise ValueError("not original Weibo result")
    if (not b["one_held_person_only"] or
        not b["cannot_call_id9_independently_blinded"] or
        not w["all_target_outcomes_previously_exposed_to_project"] or
        not w["two_subjects_one_acquisition_not_two_independent_acquisitions"]):
        raise ValueError("source/target exposure assertions must remain true")
    rows=[("BNCI2014_004",9,b["target_auc"])]
    rows.extend(("Weibo2014",int(x["target_id"]),x["auc"]) for x in w["rows"])
    if len(rows)!=3 or [(x,y) for x,y,_ in rows]!=[
        ("BNCI2014_004",9),("Weibo2014",9),("Weibo2014",10)]:
        raise ValueError("not the same source-locked target records")
    records=[]
    for ds,number,auc in rows:
        if set(auc)!=set(METHODS) or any(not 0<=float(v)<=1 for v in auc.values()):
            raise ValueError("nonmatching method matrix / invalid AUC")
        base=auc["source_macro"]
        delta={m:float(auc[m]-base) for m in METHODS}
        records.append({"acquisition":ds,"public_id":number,
            "paired_delta_auc":delta,
            "source_macro_oracle_regret":max(auc.values())-base})
    bymethod={}
    for method in METHODS:
        z=[x["paired_delta_auc"][method] for x in records]
        bymethod[method]={
            "descriptive_record_mean_auc_delta":sum(z)/3,
            "min_held_record_auc_delta":min(z),
            "max_held_record_auc_delta":max(z),
            "harm_over_0_02_count":sum(v<-.02 for v in z),
            "meaningful_gain_over_0_005_count":sum(v>.005 for v in z),
            "acquisition_means":{
                ds:sum(t["paired_delta_auc"][method] for t in records
                       if t["acquisition"]==ds)/
                   sum(t["acquisition"]==ds for t in records)
                for ds in ("BNCI2014_004","Weibo2014")}}
    assert bymethod["continuous"]["descriptive_record_mean_auc_delta"]==0
    assert max(x["source_macro_oracle_regret"] for x in records)<.000402
    assert bymethod["SCE"]["harm_over_0_02_count"]==1
    return {"status":"TWO_PREVIOUSLY_EXPOSED_ACQUISITIONS_REPLAY",
        "immutable_original_run_ids":[38341955398,38362787873],
        "distinct_acquisitions":2,"target_records":3,
        "all_outcomes_exposed_prior_to_this_audit":True,
        "three_records_are_NOT_three_independent_acquisitions":True,
        "different_source_pools_across_acquisitions":True,
        "no_prospective_target_or_new_algorithms":True,
        "no_population_significance_test":True,
        "best_retrospective_constant_policy_is_macro_or_equal_continuous":
            all(bymethod[x]["descriptive_record_mean_auc_delta"]<=0
                for x in METHODS),
        "records":records,"by_method":bymethod}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--bnci",type=Path,required=True)
    p.add_argument("--weibo",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    rb=x.bnci.read_bytes();rw=x.weibo.read_bytes()
    result=audit(json.loads(rb),json.loads(rw))
    result["original_artifact_byte_hash"]={
        "bnci_sha256":hashlib.sha256(rb).hexdigest(),
        "weibo_sha256":hashlib.sha256(rw).hexdigest()}
    data=json.dumps(result,indent=2,sort_keys=True)+"\n"
    x.out.parent.mkdir(parents=True,exist_ok=True)
    x.out.write_text(data)
    print("P3_EXPOSED_FIXED_POLICY_MATRICES_REPLAY_PASS",
          json.dumps({m:z["descriptive_record_mean_auc_delta"]
                      for m,z in result["by_method"].items()},sort_keys=True))
