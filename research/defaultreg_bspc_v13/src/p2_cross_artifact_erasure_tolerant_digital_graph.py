#!/usr/bin/env python3
"""Reconcile TWO independent actual-public-EDF CI artifacts into a release-specific
erasure-tolerant 50-ID DIGITAL-CONTENT graph, not a patient-identity graph.

The 50-file census is fixed from Actions 38388066957; the detailed
prereported-pair topology is from Actions 38543795987.
A graph edge at the inspected K=28 of 29 named EEG channels and
L=100 consecutive native storage records is retrospective,
not an externally calibrated population specificity threshold.
A 256-bit SHA digest match is cryptographic content evidence, not
formal zero collision probability or raw acquisition identity.
"""
import argparse,collections,hashlib,json
from pathlib import Path

COUNT=50
NPAIRS=1225
CH=29
R=320
K=28
L=100
PAIRS=((4,5),(11,15),(47,49))
CENSUS_RUN=38388066957
TOPOLOGY_RUN=38543795987

def digest(p):
    raw=p.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()

def classify(census,topology):
    if (census.get("status")!="PUBLIC_NEMAR_FIFTY_EDF_WHOLE_COHORT_SCALAR_AUDIT"
        or census.get("n_public_ids_rehashed")!=COUNT
        or census.get("n_unordered_comparisons")!=NPAIRS
        or not census.get("all_full_file_sha256_match_versioned_external_manifest")
        or census.get("full_EDF_sha256_unique_ids")!=COUNT):
        raise ValueError("invalid complete SHA-fixed fifty-file census")
    if (topology.get("status")!="PUBLIC_EDF_DIGITAL_BLOCK_OVERLAP_CONTIGUOUS_SPANS_REAL"
        or not topology.get("full_binary_SHA256_recomputed_and_confirmed_6_files")
        or topology.get("n_analysed_prenominated_pairs")!=3):
        raise ValueError("invalid six-file real-byte topology")
    if census["source_provenance_commit"]!=topology["versioned_provenance_commit"]:
        raise ValueError("the independent runs used DIFFERENT release commits")
    rows=census["pairwise_scalar_records"]
    tbl={tuple(z["public_ids"]):z for z in rows}
    if len(tbl)!=NPAIRS or set(tbl)!=set((i,j) for i in range(1,51)
                                                for j in range(i+1,51)):
        raise ValueError("not the full non-repeating triangular ID matrix")
    topo={tuple(z["released_public_ID_pair"]):z for z in topology["pairs"]}
    if set(topo)!=set(PAIRS):raise ValueError("topology prereported pairs changed")
    for pair,t in topo.items():
        b=tbl[pair]
        if b["aligned_equal_native_EEG_storage_blocks"]!=t[
             "total_aligned_exact_EEG_storage_blocks"]:
            raise ValueError("two independent scripts disagree on same-release pair")
        if t["n_native_EDF_records"]!=R or t["n_named_EEG_channels"]!=CH:
            raise ValueError("EDF geometry")
        if any(int(k) in (25,26,27) for k in
               t["nonconstant_exact_channel_count_per_native_record_histogram"]):
            raise ValueError("cannot infer >=28 from >=25 run-length receipt")
        if (t["exact_native_blocks_with_genuine_within_block_time_variation"]
            !=t["total_aligned_exact_EEG_storage_blocks"]):
            raise ValueError("weak constant-block evidence requires special treatment")
    if [tbl[k]["aligned_equal_native_EEG_storage_blocks"]
        for k in PAIRS]!=[9280,171,6957]:
        raise ValueError("known cases not recomputed correctly")
    if any(x["aligned_equal_native_EEG_storage_blocks"]!=0
           for key,x in tbl.items() if key not in PAIRS):
        raise ValueError("positive unregistered pairs require their own span audit")
    out=[]
    for pair,r in sorted(tbl.items()):
        t=topo.get(pair)
        strict_full=(r["exact_time_varying_EEG_full_channels"]==CH)
        if t:
            max_span=t["longest_run_with_at_least_25_nonconstant_EEG_channel_blocks"]
            observed_per_record=sorted(int(k) for k in
                   t["nonconstant_exact_channel_count_per_native_record_histogram"])
            all_informative=t["exact_native_blocks_with_genuine_within_block_time_variation"]
        else:
            max_span=0
            observed_per_record=[0]
            all_informative=0
        if max_span>L and r["aligned_equal_native_EEG_storage_blocks"]==0:
            raise AssertionError("cannot have robust overlap without any exact blocks")
        out.append({
          "pair":list(pair),
          "whole_file_SHA256_equal":False,
          "all_29_complete_time_varying_EEG_channel_streams_equal":strict_full,
          "nonconstant_aligned_native_blocks":all_informative,
          "longest_contiguous_run_with_at_least_28_matching_nonconstant_channels":
             max_span,
          "meets_retrospective_28channel_100native_record_rule":max_span>=L,
          "not_confirmed_same_patient_or_raw_acquisition":True})
    exact=[x["pair"] for x in out if x[
      "all_29_complete_time_varying_EEG_channel_streams_equal"]]
    tolerant=[x["pair"] for x in out if x[
      "meets_retrospective_28channel_100native_record_rule"]]
    if exact!=[[4,5]] or tolerant!=[[4,5],[47,49]]:
        raise AssertionError("rule audit contradicts independent source artifacts")
    # These are *digital-product* connected components, NOT distinct patients.
    parent=list(range(COUNT+1))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for a,b in tolerant:parent[root(b)]=root(a)
    components=collections.defaultdict(list)
    for i in range(1,51):components[root(i)].append(i)
    return {
      "status":"TWO_INDEPENDENT_COMPLETED_REAL_EDF_RECEIPTS_RECONCILED",
      "source_runs":[CENSUS_RUN,TOPOLOGY_RUN],
      "n_checked_published_EDF_ids":COUNT,"n_complete_pairs":NPAIRS,
      "retrospective_erasure_tolerant_rule":{
        "minimum_matching_time_varying_EEG_channels_per_native_record":K,
        "consecutive_native_records":L,"eeg_channels":CH,
        "thresholds_selected_AFTER_seeing_exploratory_release":True},
      "comparators":{
        "whole_EDF_SHA_identity_edges":0,
        "all_29_full_EEG_channel_content_identity_edges":len(exact),
        "full_channel_identity_pairs":exact,
        "one_channel_mismatch_tolerant_contiguous_edges":len(tolerant),
        "one_channel_mismatch_tolerant_pairs":tolerant,
        "other_unlinked_pairs_under_inspected_rule":NPAIRS-len(tolerant)},
      "digital_content_graph_connected_components":sorted(components.values()),
      "digital_content_graph_component_count":len(components),
      "THIS_IS_NOT_A_PATIENT_COUNT":True,
      "preprocessed_EDF_release_version_only":True,
      "no_source_MAT_conversion_or_clinical_identity_proof":True,
      "no_population_false_positive_or_power_claim":True,
      "pairwise_nonidentifying_scalar_evidence":out
    }

def run(c,t,out):
    x,hx=digest(c);y,hy=digest(t)
    z=classify(x,y)
    z["independent_input_artifact_sha256"]={
        "all_fifty_full_pair_census":hx,
        "six_file_temporal_nonconstant_topology":hy}
    data=json.dumps(z,sort_keys=True,indent=2)+"\n"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(data)
    print("P2_PUBLIC_EDF_ERASURE_TOLERANT_GRAPH",json.dumps({
      "release_only_components":z["digital_content_graph_component_count"],
      "compared":z["comparators"],"frozen":z["retrospective_erasure_tolerant_rule"]},
      sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--census",type=Path,required=True)
    p.add_argument("--topology",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    run(args.census,args.topology,args.out)
