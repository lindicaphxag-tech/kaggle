#!/usr/bin/env python3
"""Cross-ID native EDF digital RECORD-BLOCK provenance diagnostics.

This test is distinct from whole-record waveform correlation and complete
digital-channel byte equality. For each public-release ID pair it asks:
(1) do *nonconstant* corresponding-channel EDF blocks match byte-for-byte
    at the original record index?
(2) do any such nonconstant blocks match at a DIFFERENT EDF record index?
(3) how many aligned native blocks correlate strongly in UNFILTERED
    integer digital samples, as opposed to only after preprocessing?

The test does NOT reconstruct clinical identity, recognise acquisition
sessions, infer a causal EEG leakage penalty, or estimate how often
unselected population recordings contain duplicates. Digital blocks and
patient-related EDF header fields are never written to result artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

from p2_edf_digital_payload_audit import header
from p2_raw_edf_channel_affine_fingerprint import _make_small_edf

PAIR_IDS = ((4,5),(47,49),(11,15),(2,3),(6,7))
DISCOVERY_PAIRS = {(4,5),(47,49),(11,15)}

def _records(path, h):
    """Read exactly the native EDF digital int16 per-channel record blocks."""
    with Path(path).open("rb") as f:
        f.seek(h["header"])
        for record_idx in range(h["records"]):
            chunks=[]
            for count in h["samples"]:
                raw=f.read(2*count)
                if len(raw)!=2*count:
                    raise EOFError("native EDF digital record truncated")
                chunks.append(raw)
            yield record_idx, chunks

def _variable(raw):
    a=np.frombuffer(raw,dtype="<i2")
    return a.size>=3 and int(a.min())!=int(a.max())

def _pearson_nonconstant(raw_a,raw_b):
    a=np.frombuffer(raw_a,dtype="<i2").astype(np.float64)
    b=np.frombuffer(raw_b,dtype="<i2").astype(np.float64)
    if len(a)!=len(b) or len(a)<3:
        return None
    a-=np.mean(a)
    b-=np.mean(b)
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    if den<=0:
        return None
    return float(np.clip(np.dot(a,b)/den,-1,1))

def one_pair(path_a,path_b):
    A=header(path_a);B=header(path_b)
    same_schema=all(A[key]==B[key] for key in ("n","names","samples","records"))
    if not same_schema:
        raise ValueError("nonmatching EDF channel/record schema: report as unsupported, not negative")
    if A["records"]<1:
        raise ValueError("no native EDF records")
    # Per-channel SHA256 indexed by EDF record number. Skip constant blocks:
    # identical zero blocks are not evidence of meaningful EEG signal reuse.
    map_a=[defaultdict(set) for _ in range(A["n"])]
    for i,chunks in _records(path_a,A):
        for ch,raw in enumerate(chunks):
            if _variable(raw):
                map_a[ch][hashlib.sha256(raw).digest()].add(i)
    result={
        "aligned_nonconstant_channel_blocks":0,
        "aligned_exact_nonconstant_blocks":0,
        "any_position_exact_nonconstant_blocks":0,
        "relocated_exact_nonconstant_blocks":0,
        "aligned_raw_nonconstant_corr_ge_0p95":0,
        "aligned_raw_nonconstant_corr_ge_0p9999":0,
        "constant_or_unmatched_blocks_not_positive_evidence":True,
        "n_edf_records":A["records"],
        "n_edf_channels":A["n"],
        "not_physically_calibrated":True,
        "does_not_inspect_EDF_patient_header":True,
    }
    # Pair the corresponding native blocks from the original files.
    corrs=[]
    source=_records(path_a,A)
    destination=_records(path_b,B)
    for (idx,aa),(check_idx,bb) in zip(source,destination):
        if idx!=check_idx:
            raise AssertionError("native record order drift")
        for ch,(x,y) in enumerate(zip(aa,bb)):
            dx=hashlib.sha256(y).digest()
            if _variable(y):
                positions=map_a[ch].get(dx,set())
                if positions:
                    result["any_position_exact_nonconstant_blocks"]+=1
                    if idx in positions:
                        result["aligned_exact_nonconstant_blocks"]+=1
                    if any(pos!=idx for pos in positions):
                        result["relocated_exact_nonconstant_blocks"]+=1
            rho=_pearson_nonconstant(x,y)
            if rho is None:
                continue
            result["aligned_nonconstant_channel_blocks"]+=1
            result["aligned_raw_nonconstant_corr_ge_0p95"]+=int(rho>=.95)
            result["aligned_raw_nonconstant_corr_ge_0p9999"]+=int(rho>=.9999)
            corrs.append(rho)
    if corrs:
        result["native_block_correlation_median"]=float(np.median(corrs))
        result["native_block_correlation_p10"]=float(np.quantile(corrs,.1))
        result["native_block_correlation_p90"]=float(np.quantile(corrs,.9))
    else:
        result["native_block_correlation_median"]=None
        result["native_block_correlation_p10"]=None
        result["native_block_correlation_p90"]=None
    result["comparison"]="Same EDF channel and record index for Pearson; SHA256 of original channel/block digital bytes also matched across record indices"
    return result

def self_test():
    with tempfile.TemporaryDirectory() as td:
        base=Path(td)
        a=base/"a.edf"; b=base/"b.edf";c=base/"c.edf";d=base/"d.edf"
        X=[[1,2,7,8,3,4,9,10],[-2,0,3,6,7,9,13,14]]
        Y=[[2*q+17 for q in x] for x in X]
        Permuted=[x[4:]+x[:4] for x in X]
        Different=[[1,6,2,7,4,9,5,13],[11,3,7,0,4,14,2,10]]
        for p,v in ((a,X),(b,Y),(c,Permuted),(d,Different)):
            _make_small_edf(p,v)
        same=one_pair(a,a);affine=one_pair(a,b)
        shift=one_pair(a,c);diff=one_pair(a,d)
        assert same["aligned_exact_nonconstant_blocks"]==4
        assert same["aligned_raw_nonconstant_corr_ge_0p9999"]==4
        assert affine["aligned_exact_nonconstant_blocks"]==0
        assert affine["aligned_raw_nonconstant_corr_ge_0p9999"]==4
        assert shift["aligned_exact_nonconstant_blocks"]==0
        assert shift["relocated_exact_nonconstant_blocks"]==4
        assert shift["any_position_exact_nonconstant_blocks"]==4
        assert diff["aligned_exact_nonconstant_blocks"]==0
        assert diff["relocated_exact_nonconstant_blocks"]==0
        assert all(0<=x["aligned_raw_nonconstant_corr_ge_0p95"]<=
                   x["aligned_nonconstant_channel_blocks"]
                   for x in (same,affine,shift,diff))
    print("P2_NATIVE_EDF_BLOCK_EXACT_ALIGNED_AFFINE_RELOCATED_NEGATIVE_SELFTEST_PASS",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    if args.out is None:
        ap.error("--out required for public digital EEG audit")
    from moabb.datasets import Liu2024
    dataset=Liu2024()
    pairs=[]
    for a,b in PAIR_IDS:
        result=one_pair(dataset.data_path(a)[0],dataset.data_path(b)[0])
        result["released_public_ids"]=[a,b]
        result["selected_after_prior_processed_1225_pair_screen"]=(a,b) in DISCOVERY_PAIRS
        pairs.append(result)
        print("P2_NATIVE_EDF_BLOCK_PAIR_DONE",a,b,
              result["aligned_exact_nonconstant_blocks"],
              result["relocated_exact_nonconstant_blocks"],
              result["aligned_raw_nonconstant_corr_ge_0p9999"],
              result["aligned_nonconstant_channel_blocks"],flush=True)
    report={
        "study":"Liu2024 original public EDF native unfiltered digital block-content and correlation diagnostic",
        "status":"RETROSPECTIVE_PUBLIC_DIGITAL_BLOCK_AUDIT",
        "exploratory_fixed_pairs_before_current_algorithm_run":True,
        "not_independently_registered_before_prior_data_inspection":True,
        "no_waveforms_or_patient_EDF_header_in_output":True,
        "not_patient_or_acquisition_identity_evidence_by_itself":True,
        "no_classifier_target_outcomes_used":True,
        "native_blocks_are_not_epoch_or_patient_counts":True,
        "pairs":pairs,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.write_text(text,encoding="utf-8")
    print("P2_NATIVE_BLOCK_REPORT_SHA256",hashlib.sha256(text.encode()).hexdigest(),flush=True)

if __name__=="__main__":
    main()
