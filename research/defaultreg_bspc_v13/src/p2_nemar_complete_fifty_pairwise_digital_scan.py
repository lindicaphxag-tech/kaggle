#!/usr/bin/env python3
"""All-50 public NEMAR EDF data-product audit, no released waveforms stored.

Distribution version: NEMAR nm000158 v1.0.3
Git provenance: 11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83
All 50 full EDF SHA256 hashes are independently recomputed and checked.
Compare ALL unordered public ID pairs (50 choose 2 = 1225).
Two endpoints: entire time-varying digital channels; aligned physical
EEG-labelled native EDF storage blocks. These blocks are NOT independent
trials, recordings, patients or acquisition sessions. This is the SAME
underlying Liu2024 dataset, not external clinical validation.
"""
import argparse, collections, hashlib, itertools, json, time
from pathlib import Path

from p2_nemar_real_edf_full_file_and_digital_channel_audit import (
    COMMIT, ROOT, MANIFEST, fetch, parse_edf, field, number, selftest,
)

EEG_LABELS=(
    "FP1","FP2","Fz","F3","F4","F7","F8","FCz","FC3","FC4",
    "FT7","FT8","Cz","C3","C4","T3","T4","CP3","CP4","TP7",
    "TP8","Pz","P3","P4","T5","T6","Oz","O1","O2",
)
EXPECTED_IDS=tuple(range(1,51))

def digital_eeg_blocks(blob, meta):
    labels=meta["channel_labels"]
    if len(labels)!=len(set(labels)):
        raise ValueError("ambiguous repeated EDF channel labels")
    idx={name:i for i,name in enumerate(labels)}
    if not set(EEG_LABELS).issubset(idx):
        raise ValueError("29 EEG label definitions not found")
    n=meta["channels"]
    records=meta["records"]
    hd=number(field(blob,184,192))
    sizes=meta["samples_per_record"]
    # Build in an unambiguous, label-first shape.
    blocks={name:[] for name in EEG_LABELS}
    start=hd
    for rec in range(records):
        for channel in range(n):
            length=2*sizes[channel]
            piece=blob[start:start+length]
            name=labels[channel]
            if name in blocks:
                blocks[name].append(hashlib.sha256(piece).hexdigest())
            start+=length
    if start!=len(blob) or not all(len(v)==records for v in blocks.values()):
        raise AssertionError("malformed per-record EEG block layout")
    return blocks

def analyze_pair(x,y):
    if x["records"]!=y["records"]:
        return dict(comparable=False,reason="different EDF record count")
    for name in EEG_LABELS:
        i=x["channel_labels"].index(name);j=y["channel_labels"].index(name)
        if x["samples_per_record"][i]!=y["samples_per_record"][j]:
            return dict(comparable=False,reason="EEG samples-per-record differ")
    common=[]
    for name in EEG_LABELS:
        i=x["channel_labels"].index(name);j=y["channel_labels"].index(name)
        if (x["channel_has_digital_variation"][i] and
            y["channel_has_digital_variation"][j] and
            x["channel_digest"][i]==y["channel_digest"][j]):
            common.append(name)
    exact=sum(
        a==b
        for name in EEG_LABELS
        for a,b in zip(x["eeg_block_digest"][name],y["eeg_block_digest"][name])
    )
    n_blocks=len(EEG_LABELS)*x["records"]
    assert exact>=len(common)*x["records"]
    return dict(comparable=True,exact_time_varying_EEG_full_channels=len(common),
                exact_time_varying_EEG_channel_labels=common,
                aligned_equal_native_EEG_storage_blocks=exact,
                labelled_native_EEG_storage_blocks=n_blocks,
                aligned_block_match_fraction=exact/n_blocks)

def summarize_rows(rows):
    if len(rows)!=1225 or len({tuple(z["public_ids"]) for z in rows})!=1225:
        raise ValueError("not a full 50 choose 2 scan")
    if any(not z["comparable"] for z in rows):
        raise ValueError("cannot benchmark different EDF geometry without mapping")
    top=sorted(rows,key=lambda z:(-z["exact_time_varying_EEG_full_channels"],
                                  -z["aligned_equal_native_EEG_storage_blocks"],
                                  z["public_ids"]))
    hist=collections.Counter(z["exact_time_varying_EEG_full_channels"] for z in rows)
    matches=collections.Counter(z["aligned_equal_native_EEG_storage_blocks"]
                                for z in rows)
    return dict(pair_count=len(rows),full_dynamic_channel_histogram=dict(sorted(hist.items())),
                most_common_block_counts=matches.most_common(12),
                pairs_with_any_full_time_varying_EEG_channel=sum(
                    z["exact_time_varying_EEG_full_channels"]>0 for z in rows),
                pairs_with_at_least_one_fifth_equal_EEG_blocks=sum(
                    z["aligned_block_match_fraction"]>=.2 for z in rows),
                pairs_with_at_least_half_equal_EEG_blocks=sum(
                    z["aligned_block_match_fraction"]>=.5 for z in rows),
                max_non_4_5_EEG_full_channels=max(
                    z["exact_time_varying_EEG_full_channels"]
                    for z in rows if z["public_ids"]!=[4,5]),
                max_non_4_5_equal_EEG_blocks=max(
                    z["aligned_equal_native_EEG_storage_blocks"]
                    for z in rows if z["public_ids"]!=[4,5]),
                top_20=[{"public_ids":z["public_ids"],
                         "exact_time_varying_EEG_full_channels":
                            z["exact_time_varying_EEG_full_channels"],
                         "aligned_equal_native_EEG_storage_blocks":
                            z["aligned_equal_native_EEG_storage_blocks"],
                         "aligned_block_match_fraction":
                            z["aligned_block_match_fraction"]}
                        for z in top[:20]])

def synthetic_controls():
    # The standard library EDF parser is exercised independently in its
    # synthetic fixture; this test also verifies complete-pair enumeration,
    # group accounting, monotonicity of full-channel vs per-record blocks.
    assert selftest()["status"]=="SYNTHETIC_EDF_PARSER_PASS"
    a=list(itertools.combinations(EXPECTED_IDS,2))
    assert len(a)==1225 and len(set(a))==1225
    assert len(EEG_LABELS)==29 and "CPz" not in EEG_LABELS
    fake=[]
    for i,j in a:
        full=int((i,j)==(4,5))
        fake.append({"public_ids":[i,j],"comparable":True,
            "exact_time_varying_EEG_full_channels":full,
            "aligned_equal_native_EEG_storage_blocks":320*full,
            "labelled_native_EEG_storage_blocks":9280,
            "aligned_block_match_fraction":320*full/9280})
    stats=summarize_rows(fake)
    assert stats["pair_count"]==1225
    assert stats["pairs_with_any_full_time_varying_EEG_channel"]==1
    assert stats["max_non_4_5_EEG_full_channels"]==0
    return {"status":"PASS","synthetic_pairs":len(a),
            "no_real_public_EEG_read":True}

def live():
    provenance=fetch(MANIFEST,150000)
    manifest=json.loads(provenance)
    files=manifest["files"]
    byid={int(item["subject"]):item for item in files}
    if (len(files)!=50 or set(byid)!=set(EXPECTED_IDS) or
        manifest.get("moabb_version_used")!="1.5.0" or
        manifest.get("n_files")!=50):
        raise ValueError("unexpected versioned public EDF manifest")
    scans={};each=[]
    for uid in EXPECTED_IDS:
        row=byid[uid]
        data=fetch(f'{ROOT}/sourcedata/{row["file"]}',int(row["bytes"])+4096)
        checksum=hashlib.sha256(data).hexdigest()
        if checksum!=row["sha256"]:
            raise RuntimeError(f"EDF public ID {uid} does not match pinned provenance")
        meta=parse_edf(data)
        meta["eeg_block_digest"]=digital_eeg_blocks(data,meta)
        scans[uid]=meta
        each.append({"released_public_id":uid,
                     "bytes":len(data),
                     "whole_file_manifest_sha256_verified":True,
                     "records":meta["records"],
                     "channel_count":meta["channels"],
                     "n_labelled_EEG_channels":len(EEG_LABELS)})
        print("P2_PUBLIC_EDF_REHASH_VERIFIED",uid,len(data),flush=True)
    pairs=[]
    for i,j in itertools.combinations(EXPECTED_IDS,2):
        cmp=analyze_pair(scans[i],scans[j])
        pairs.append({"public_ids":[i,j],**cmp})
    summary=summarize_rows(pairs)
    return {
        "status":"PUBLIC_NEMAR_FIFTY_EDF_WHOLE_COHORT_SCALAR_AUDIT",
        "source":"NEMAR nm000158 v1.0.3 public preprocessed EDF",
        "underlying_acquisition_dataset":"Liu2024, same study not independent collection",
        "source_provenance_commit":COMMIT,
        "manifest_sha256":hashlib.sha256(provenance).hexdigest(),
        "n_public_ids_rehashed":len(scans),
        "n_unordered_comparisons":len(pairs),
        "all_full_file_sha256_match_versioned_external_manifest":True,
        "full_EDF_sha256_unique_ids":
            len(set(row["sha256"] for row in files)),
        "EEG_labels":list(EEG_LABELS),
        "CPz_constant_reference_excluded_from_overlap_count":True,
        "no_raw_EEG_waveforms_or_patient_data_in_output":True,
        "MAT_to_EDF_transform_and_clinical_acquisition_identity_unknown":True,
        "pairs_are_not_independent_patient_observations":True,
        "detection_thresholds_are_descriptive_not_prospective_FPR":True,
        "summary":summary,"pairwise_scalar_records":pairs,
        "per_public_file_nonidentifying_counts":each}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    if args.self_test: result=synthetic_controls()
    else:
        if args.out is None:parser.error("--out required for public 50 EDF scan")
        result=live()
    out=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(out)
    print("P2_FIFTY_EDF_SCALAR_RECEIPT",json.dumps({
        "status":result["status"],
        "summary":result.get("summary"),
        "n_public_ids_rehashed":result.get("n_public_ids_rehashed")},
        sort_keys=True),flush=True)
