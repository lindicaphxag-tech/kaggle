#!/usr/bin/env python3
"""P3 pre-target manifest contract. SYNTHETIC dry-run, no EEG outcome.

A successful check means that the manifest/commitment schema behaves as
intended, not that two independent unseen EEG datasets were evaluated.
"""
import hashlib,json
from copy import deepcopy

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(b): return hashlib.sha256(b).hexdigest()

REQUIRED={"baseline","candidates","source_groups","reserved_target_groups",
          "source_features_sha256","pipeline_commit","target_information_budget",
          "primary_estimand","pretarget_frozen"}
def validate(m,source_bytes):
    if set(m)!=REQUIRED: raise ValueError("explicit frozen schema required")
    if not m["pretarget_frozen"]: raise ValueError("unfrozen")
    if m["target_information_budget"]!="Z0_no_target_features_or_labels":
        raise ValueError("target information leakage")
    if m["primary_estimand"]!="paired_target_record_macro_auc_regret":
        raise ValueError("mismatched endpoint")
    if len(set(m["candidates"]))!=len(m["candidates"]):
        raise ValueError("duplicate method IDs")
    if m["baseline"] not in m["candidates"]: raise ValueError("missing baseline")
    if not m["source_groups"] or len(m["reserved_target_groups"])<2:
        raise ValueError("need source groups and >=2 reserved acquisitions")
    sources={(d,str(x)) for d,ids in m["source_groups"].items() for x in ids}
    target={(d,str(x)) for d,ids in m["reserved_target_groups"].items() for x in ids}
    if sources & target: raise ValueError("source/target ID overlap")
    if not m["pipeline_commit"] or len(m["pipeline_commit"])<7:
        raise ValueError("missing pipeline revision")
    if digest(source_bytes)!=m["source_features_sha256"]:
        raise RuntimeError("SOURCE DRIFT: stop BEFORE reading target")
    return digest(canonical(m))

def test():
    source=b"synthetic-EEG-p3-source-feature-bytes-only"
    m={"baseline":"grid_macro","candidates":["grid_macro","proper_nll","sce"],
       "source_groups":{"demo_source":[1,2,3,4]},
       "reserved_target_groups":{"demo_external_A":[5,6],
                                 "demo_external_B":[10,11]},
       "source_features_sha256":digest(source),
       "pipeline_commit":"0000000-SYNTHETIC",
       "target_information_budget":"Z0_no_target_features_or_labels",
       "primary_estimand":"paired_target_record_macro_auc_regret",
       "pretarget_frozen":True}
    lock=validate(m,source)
    try: validate(m,source+b"stale")
    except RuntimeError: pass
    else: raise AssertionError("fail-closed SHA drift test FAILED")
    x=deepcopy(m); x["target_information_budget"]="use_target_statistics"
    try: validate(x,source)
    except ValueError: pass
    else: raise AssertionError("target information budget fail-closed FAILED")
    return {"synthetic_commitment_sha256":lock,
            "drift_and_information_leak_blocked":True,
            "uses_real_EEG_or_held_target_outcomes":False,
            "independent_external_acquisitions_completed":0}
if __name__=="__main__":
    print(json.dumps(test(),indent=2))
