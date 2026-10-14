#!/usr/bin/env python3
"""P3 Dreyer2023B source SHA drift forensic, NO target persons loaded.

Repeatedly re-read only source public IDs61-68 with same source
preprocessing, and hash each subject's covariance features in both
ordered and multiset-of-labelled-trial forms. This determines whether
source byte nondeterminism appears WITHIN a single runner or only
BETWEEN independent runs. DOES NOT assert clinical/recruitment identity.
No target subjects 69-76 are read.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from moabb.datasets import Dreyer2023B
from moabb.paradigms import MotorImagery
from p3_dreyer2023b_first_source_locked_two_targets import load,SRC

OLD_SOURCE_SHA="7146797ee8d6caf71c57ee141604102b2d61217456cbfe3e00820b83e172b2a7"
RECENT_SOURCE_SHA="9f650460378e695189c3fa145bf0a29fe010b24cd92ba4aa1196e7439ada7f65"

def pass_one(ds,p):
    c,y,g,receipts,h=load(ds,p,SRC)
    details=[]
    for sid in SRC:
        ix=(g==sid)
        cc=np.ascontiguousarray(c[ix],dtype=np.float64)
        yy=y[ix]
        hashes=[hashlib.sha256(cc[i].tobytes()+bytes([int(yy[i])])).hexdigest()
                for i in range(len(yy))]
        traces=np.trace(cc,axis1=1,axis2=2)
        if not np.isfinite(cc).all() or not np.all(traces>0):
            raise ValueError("nonfinite or nonpositive covariance trace")
        # Relative shape / global amplitude fingerprints are diagnostic
        # numerical roundings, not exact equality certificates.
        normalized=cc/traces[:,None,None]
        shape_bin=np.rint(normalized*1e9).astype("<i8")
        trace_bin=np.rint(np.log10(traces)*1e8).astype("<i8")
        shape_fingerprints=[
            hashlib.sha256(np.ascontiguousarray(shape_bin[i]).tobytes()
            +bytes([int(yy[i])])).hexdigest() for i in range(len(yy))]
        amp_fingerprints=[
            hashlib.sha256(trace_bin[i].tobytes()
            +bytes([int(yy[i])])).hexdigest() for i in range(len(yy))]
        details.append({"public_id":sid,"n":len(yy),
            "positive":int(yy.sum()),
            "quantized_relative_covariance_9decimals_hash":
                hashlib.sha256(b"".join(bytes.fromhex(h) for h in shape_fingerprints)).hexdigest(),
            "quantized_log10_trace_8decimals_hash":
                hashlib.sha256(b"".join(bytes.fromhex(h) for h in amp_fingerprints)).hexdigest(),
            "ordered_epoch_digest":hashlib.sha256(
                b"".join(bytes.fromhex(z) for z in hashes)).hexdigest(),
            "sorted_epoch_digest":hashlib.sha256(
                b"".join(bytes.fromhex(z) for z in sorted(hashes))).hexdigest(),
            "feature_mean_trace":float(np.trace(cc,axis1=1,axis2=2).mean()),
            "feature_mean_logdet":float(np.mean(np.linalg.slogdet(cc)[1]))})
    return {"source_aggregate_sha256":h,"per_subject":details,"subjects":list(SRC),
            "n_total":len(y),"positive_total":int(y.sum())}

def main(path):
    ds=Dreyer2023B()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                   channels=["C3","Cz","C4"],fmin=8,fmax=30,
                   resample=160,baseline=None,reject_by_annotation=False)
    rounds=[]
    for k in range(2):
        z=pass_one(ds,p);rounds.append(z)
        print("P3_DREYER_SOURCE_REPLAY_ROUND",k+1,z["source_aggregate_sha256"],flush=True)
    a,b=rounds
    comparison=[]
    for x,y in zip(a["per_subject"],b["per_subject"]):
        if x["public_id"]!=y["public_id"] or x["n"]!=y["n"] or x["positive"]!=y["positive"]:
            raise AssertionError("source subject labels/counts drift")
        comparison.append({"public_id":x["public_id"],
            "ordered_exact":x["ordered_epoch_digest"]==y["ordered_epoch_digest"],
            "labelled_epoch_multiset_exact":
                x["sorted_epoch_digest"]==y["sorted_epoch_digest"],
            "trace_difference":x["feature_mean_trace"]-y["feature_mean_trace"]})
    report={"status":"SOURCE_ONLY_THIRD_PARTY_RETRIEVAL_REPRO_FORENSIC",
       "source_ids":list(SRC),"n_read_rounds":2,"held_target_ids_NEVER_REQUESTED":[69,70,71,72,73,74,75,76],
       "previous_original_source_sha256":OLD_SOURCE_SHA,
       "previous_separate_run_source_sha256":RECENT_SOURCE_SHA,
       "new_rounds":rounds,"within_run_same_aggregate":
           a["source_aggregate_sha256"]==b["source_aggregate_sha256"],
       "within_run_per_person_comparisons":comparison,
       "source_release_raw_bytes_NOT_independently_fingerprinted":True,
       "cannot_assign_drift_cause_from_this_test_alone":True,
       "not_independent_held_subject_validation":True}
    data=json.dumps(report,indent=2,sort_keys=True)+"\n"
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data)
    print("P3_SOURCE_DRIFT_DIAG",json.dumps(
        {"within_run_same":report["within_run_same_aggregate"],
         "per_subject":comparison},sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();main(a.out)
