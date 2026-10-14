#!/usr/bin/env python3
"""Liu2024 EDF overlapping DIGITAL-BLOCK topology on 3 prereported pairs.

Separate contiguous record spans from dispersed matching channel blocks.
Source: version-pinned NEMAR public 50-EDF data-product manifest.
Six public full EDFs are downloaded and their full binary SHA256
recomputed and required to match publisher's digests BEFORE analysis.
No raw waveforms, per-patient metadata or assumed MAT lineage published.
One native EDF storage record is NOT an independent labelled EEG trial.
"""
import argparse,hashlib,json
from pathlib import Path
from p2_nemar_real_edf_full_file_and_digital_channel_audit import (
    ROOT,MANIFEST,COMMIT,fetch,parse_edf
)
from p2_nemar_complete_fifty_pairwise_digital_scan import (
    EEG_LABELS,digital_eeg_blocks
)

PAIRS=((4,5),(11,15),(47,49))
IDS=tuple(sorted({x for pair in PAIRS for x in pair}))

def runs(vals,predicate):
    begins=[];start=None
    for i,z in enumerate(vals):
        if predicate(z):
            if start is None:start=i
        elif start is not None:
            begins.append({"first_zero_based_EDF_record":start,
               "last_zero_based_EDF_record_inclusive":i-1,
               "n_native_records":i-start})
            start=None
    if start is not None:
        begins.append({"first_zero_based_EDF_record":start,
            "last_zero_based_EDF_record_inclusive":len(vals)-1,
            "n_native_records":len(vals)-start})
    return begins

def analyze(a,b):
    assert a["records"]==b["records"]
    R=a["records"]
    matched_per_record=[
      sum(a["eeg_block_digest"][ch][i]==b["eeg_block_digest"][ch][i]
          for ch in EEG_LABELS)
      for i in range(R)]
    counts={str(k):matched_per_record.count(k) for k in range(len(EEG_LABELS)+1)
            if matched_per_record.count(k)}
    assert sum(matched_per_record)==sum(k*val for k,val in
              ((int(k),v) for k,v in counts.items()))
    positive=runs(matched_per_record,lambda z:z>0)
    full=runs(matched_per_record,lambda z:z==len(EEG_LABELS))
    near=runs(matched_per_record,lambda z:z>=25)
    first_last=lambda val: [i for i,n in enumerate(matched_per_record) if n==val]
    return {"n_native_EDF_records":R,"n_named_EEG_channels":len(EEG_LABELS),
      "total_aligned_exact_EEG_storage_blocks":sum(matched_per_record),
      "per_record_matched_channel_histogram":counts,
      "num_records_with_any_matching_EEG_channel":
          sum(n>0 for n in matched_per_record),
      "num_records_with_all_29_EEG_channels_matched":
          sum(n==29 for n in matched_per_record),
      "num_records_with_at_least_25_EEG_channels_matched":
          sum(n>=25 for n in matched_per_record),
      "num_records_with_zero_EEG_channels_matched":
          sum(n==0 for n in matched_per_record),
      "all_29_equal_contiguous_record_runs":full,
      "at_least_25_equal_contiguous_record_runs":near,
      "any_equal_contiguous_record_runs":positive,
      "longest_full29_contiguous_run":max([z["n_native_records"] for z in full],default=0),
      "longest_any_equal_contiguous_run":max([z["n_native_records"] for z in positive],default=0),
      "first_ten_per_record_equal_channel_counts":matched_per_record[:10],
      "last_ten_per_record_equal_channel_counts":matched_per_record[-10:],
      "not_physically_independent_EEG_trials":True}

def synthetic():
    r=runs([0,0,3,3,0,1,1,1],lambda v:v>0)
    assert [x["n_native_records"] for x in r]==[2,3]
    assert r[0]["first_zero_based_EDF_record"]==2
    assert runs([0,0],lambda v:v>0)==[]
    return {"status":"SYNTHETIC_RECORD_SPAN_BOUNDARY_PASS"}

def live():
    meta=json.loads(fetch(MANIFEST,150000))
    byid={int(x["subject"]):x for x in meta["files"]}
    assert set(IDS).issubset(byid) and meta["moabb_version_used"]=="1.5.0"
    prepared={}
    for i in IDS:
        row=byid[i]
        blob=fetch(f'{ROOT}/sourcedata/{row["file"]}',int(row["bytes"])+4096)
        if hashlib.sha256(blob).hexdigest()!=row["sha256"]:
            raise RuntimeError("source distributor pinned full EDF SHA changed")
        p=parse_edf(blob)
        p["eeg_block_digest"]=digital_eeg_blocks(blob,p)
        prepared[i]=p
        print("P2_TOPOLOGY_REAL_EDF_VERIFIED",i,flush=True)
    out=[]
    for i,j in PAIRS:
        x=analyze(prepared[i],prepared[j])
        out.append({"released_public_ID_pair":[i,j],**x})
        print("P2_EXACT_EDF_TEMPORAL_OVERLAP_TOPOLOGY",
              json.dumps({"pair":[i,j],
                          "hist":x["per_record_matched_channel_histogram"],
                          "full_runs":x["all_29_equal_contiguous_record_runs"],
                          "near_runs":x["at_least_25_equal_contiguous_record_runs"],
                          "any_runs":x["any_equal_contiguous_record_runs"]},
                         sort_keys=True),flush=True)
    assert [x["total_aligned_exact_EEG_storage_blocks"] for x in out]==[
        9280,171,6957]
    return {"status":"PUBLIC_EDF_DIGITAL_BLOCK_OVERLAP_CONTIGUOUS_SPANS_REAL",
      "versioned_provenance_commit":COMMIT,
      "full_binary_SHA256_recomputed_and_confirmed_6_files":True,
      "same_underlying_Liu2024_data_collection":True,
      "no_patient_or_raw_MAT_lineage_identity_inference":True,
      "without_raw_EEG_waveforms_in_output":True,
      "n_analysed_prenominated_pairs":len(PAIRS),
      "independent_full_50_census_receipt":38388066957,
      "published_EDF_native_storage_block_not_trial":True,
      "pairs":out}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if not args.self_test and not args.out: parser.error("--out required")
    z=synthetic() if args.self_test else live()
    data=json.dumps(z,sort_keys=True,indent=2)+"\n"
    print("P2_DIGITAL_TOPOLOGY_REPRO_STATUS",z["status"])
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data)
