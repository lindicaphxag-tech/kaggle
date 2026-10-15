#!/usr/bin/env python3
"""Cross-GitHub-run original P3 DreyerB SOURCE fingerprints only.

Receipts are from two independent CI runs that loaded *only* SOURCE
IDs 61..68 twice, never any target. The SHA-256 exact float64 bytes
differ, while both independently hashed quantised SHAPE and AMPLITUDE
signatures agree per-person. This is evidence of tiny numerical
differences, NOT a proof of numerical-error bounds, raw-file byte
identity, or exact fitted-model identity.
"""
import argparse,hashlib,json,math
from pathlib import Path
EXPECTED={
  "original":"7146797ee8d6caf71c57ee141604102b2d61217456cbfe3e00820b83e172b2a7",
  "alternate":"9f650460378e695189c3fa145bf0a29fe010b24cd92ba4aa1196e7439ada7f65",
}
EXPECTED_IDS=list(range(61,69))

def validate(x,tag):
    if x.get("status")!="SOURCE_ONLY_THIRD_PARTY_RETRIEVAL_REPRO_FORENSIC":
        raise ValueError("unrecognised source-only forensic")
    if x.get("held_target_ids_NEVER_REQUESTED")!=[69,70,71,72,73,74,75,76]:
        raise ValueError("target exposure scope unrecognised")
    if not x.get("within_run_same_aggregate") or x.get("n_read_rounds")!=2:
        raise ValueError("source was not reproducible within runner")
    rounds=x["new_rounds"]
    if len(rounds)!=2:raise ValueError("expected two source-only loads")
    for r in rounds:
        if r["source_aggregate_sha256"]!=EXPECTED[tag]:
            raise ValueError("unexpected original source hash")
        if [z["public_id"] for z in r["per_subject"]]!=EXPECTED_IDS:
            raise ValueError("unexpected source people")
    for k in ("ordered_epoch_digest","sorted_epoch_digest",
              "quantized_relative_covariance_9decimals_hash",
              "quantized_log10_trace_8decimals_hash"):
        if any(a[k]!=b[k] for a,b in zip(rounds[0]["per_subject"],
                                         rounds[1]["per_subject"])):
            raise ValueError("intra-run source-person drift")
    return rounds[0]

def compare(original,alternate):
    a=validate(original,"original");b=validate(alternate,"alternate")
    if a["source_aggregate_sha256"]==b["source_aggregate_sha256"]:
        raise ValueError("wrong cross-run comparison")
    comparisons=[]
    for x,y in zip(a["per_subject"],b["per_subject"]):
        if (x["public_id"]!=y["public_id"] or
            x["n"]!=y["n"] or x["positive"]!=y["positive"]):
            raise ValueError("source record accounting changed")
        trace_delta=abs(x["feature_mean_trace"]-y["feature_mean_trace"])
        ld_delta=abs(x["feature_mean_logdet"]-y["feature_mean_logdet"])
        result={
          "public_id":x["public_id"],"n":x["n"],
          "bytewise_ordered_epoch_digest_equal":
             x["ordered_epoch_digest"]==y["ordered_epoch_digest"],
          "bytewise_labelled_epoch_multiset_digest_equal":
             x["sorted_epoch_digest"]==y["sorted_epoch_digest"],
          "quantized_1e9_covariance_shape_digest_equal":
             x["quantized_relative_covariance_9decimals_hash"]==
             y["quantized_relative_covariance_9decimals_hash"],
          "quantized_1e8_log_trace_digest_equal":
             x["quantized_log10_trace_8decimals_hash"]==
             y["quantized_log10_trace_8decimals_hash"],
          "abs_mean_trace_difference":trace_delta,
          "abs_mean_logdet_difference":ld_delta
        }
        comparisons.append(result)
    counts={k:sum(bool(z[k]) for z in comparisons)
            for k in (
              "bytewise_ordered_epoch_digest_equal",
              "bytewise_labelled_epoch_multiset_digest_equal",
              "quantized_1e9_covariance_shape_digest_equal",
              "quantized_1e8_log_trace_digest_equal")}
    if counts!={"bytewise_ordered_epoch_digest_equal":0,
                "bytewise_labelled_epoch_multiset_digest_equal":0,
                "quantized_1e9_covariance_shape_digest_equal":8,
                "quantized_1e8_log_trace_digest_equal":8}:
        raise AssertionError("forensic changed; re-audit before interpretation")
    if any(z["abs_mean_trace_difference"]>1e-11 or
           z["abs_mean_logdet_difference"]>1e-11 for z in comparisons):
        raise AssertionError("mean source float discrepancy unexpectedly material")
    return {
      "status":"TWO_INDEPENDENT_CI_RUNS_SAME_QUANTIZED_SOURCE_NOT_BYTE_IDENTICAL",
      "source_original_SHA256":EXPECTED["original"],
      "source_alternate_SHA256":EXPECTED["alternate"],
      "source_persons":8,"source_person_ids":EXPECTED_IDS,
      "within_each_run_repeat_exact":True,
      "cross_run_field_equal_counts":counts,
      "max_mean_trace_abs_diff":max(z["abs_mean_trace_difference"] for z in comparisons),
      "max_mean_logdet_abs_diff":max(z["abs_mean_logdet_difference"] for z in comparisons),
      "cross_run_per_person":comparisons,
      "input_artefact_runs":[38548578442,38548578441],
      "approximate_fingerprints_not_exact_data_identity_proof":True,
      "no_IEEE754_numerical_error_enclosure":True,
      "not_proof_of_source_fitted_model_identity":True,
      "not_target_acquisition_or_held_outcome_measurement":True,
      "all_held_targets_never_read_by_inputs":True,
      "cannot_assign_cause_to_Blas_MNE_raw_or_dataset_by_hash_alone":True}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--alternate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    arg=p.parse_args()
    rawA=arg.original.read_bytes();rawB=arg.alternate.read_bytes()
    report=compare(json.loads(rawA),json.loads(rawB))
    report["input_receipt_SHA256"]={
        "original":hashlib.sha256(rawA).hexdigest(),
        "alternate":hashlib.sha256(rawB).hexdigest()}
    out=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    arg.out.parent.mkdir(parents=True,exist_ok=True)
    arg.out.write_text(out)
    print("P3_CROSS_RUN_SOURCE_DRIFT_REPRO_PASS",json.dumps({
       "source_person_count":report["source_persons"],
       "counts":report["cross_run_field_equal_counts"],
       "max_trace_diff":report["max_mean_trace_abs_diff"],
       "max_logdet_diff":report["max_mean_logdet_abs_diff"]},
       sort_keys=True),flush=True)
