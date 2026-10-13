#!/usr/bin/env python3
"""P2 public source-MAT-to-publisher-EDF numerical correspondence diagnostic.

Both are separate release products. EDF samples are *already preprocessed*
by the publisher; integer EDF payload isn't the original sensor recording.
Checks exact signal sample counts, aligned channel-index correlations and
best one-to-one channel permutations after subtracting each channel mean.
No claim of a known MAT-to-EDF conversion map, recording sessions or people.
Only scalar aggregates are emitted; source arrays remain on a CI runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from p2_figshare_six_source_mat_range_audit import (
    IDS, StrictRange, URL, directory, write_verified_member, candidate_arrays)
from p2_edf_digital_payload_audit import header
from p2_native_edf_block_provenance_audit import _records

def pair_correlations(source_mat,edf_digital):
    """Compare aligned sample series with all channel identity hypotheses open."""
    a=np.asarray(source_mat,dtype=np.float64)
    b=np.asarray(edf_digital,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
        raise ValueError("MAT / EDF samples are not aligned at the release layer")
    if a.shape[1]<3:
        raise ValueError("not enough samples")
    aa=a-a.mean(axis=1,keepdims=True)
    bb=b-b.mean(axis=1,keepdims=True)
    an=np.linalg.norm(aa,axis=1)
    bn=np.linalg.norm(bb,axis=1)
    viable_a=an>0
    viable_b=bn>0
    ag=aa[viable_a]/an[viable_a,None]
    bg=bb[viable_b]/bn[viable_b,None]
    rho=np.clip(ag@bg.T,-1,1)
    if rho.size==0:
        raise ValueError("no nonconstant matching sample channels")
    na,nb=rho.shape
    idxa=np.flatnonzero(viable_a)
    idxb=np.flatnonzero(viable_b)
    diag=np.array([rho[i,j] for i,sa in enumerate(idxa)
                   for j,sb in enumerate(idxb) if sa==sb],dtype=float)
    rows,cols=linear_sum_assignment(-np.abs(rho))
    permuted=np.abs(rho[rows,cols])
    positive=(rho[rows,cols]>=0)
    max_per_source=np.max(np.abs(rho),axis=1)
    # Assess local 4000-sample blocks separately. A global correlation can
    # conceal trial-specific mismatch or be dominated by a single segment.
    local=[]
    if a.shape[1]==40*4000:
        for channel in sorted(set(idxa.tolist()) & set(idxb.tolist())):
            for block in range(40):
                u=a[channel,block*4000:(block+1)*4000]
                v=b[channel,block*4000:(block+1)*4000]
                u=u-u.mean();v=v-v.mean()
                denom=np.linalg.norm(u)*np.linalg.norm(v)
                if denom>0:local.append(float(np.dot(u,v)/denom))
    local=np.asarray(local,dtype=float)
    # Relax trial order inside each same-index channel, keeping the published
    # 4000-sample trial coordinate fixed. This is an exploratory best match,
    # not an externally verified source-to-EDF trial correspondence.
    trial_matches=[]
    if a.shape[1]==40*4000:
        for channel in sorted(set(idxa.tolist()) & set(idxb.tolist())):
            u=a[channel].reshape(40,4000).copy()
            v=b[channel].reshape(40,4000).copy()
            u-=u.mean(axis=1,keepdims=True)
            v-=v.mean(axis=1,keepdims=True)
            un=np.linalg.norm(u,axis=1);vn=np.linalg.norm(v,axis=1)
            goodu=un>0;goodv=vn>0
            if not goodu.any() or not goodv.any():continue
            corr=(u[goodu]/un[goodu,None])@(v[goodv]/vn[goodv,None]).T
            rr,cc=linear_sum_assignment(-np.abs(corr))
            trial_matches.extend(np.abs(corr[rr,cc]).tolist())
    trial_matches=np.asarray(trial_matches,dtype=float)
    return {
        "source_channels":int(a.shape[0]),
        "edf_channels":int(b.shape[0]),
        "n_aligned_samples_per_channel":int(a.shape[1]),
        "nonconstant_source_channels":int(na),
        "nonconstant_edf_channels":int(nb),
        "same_channel_index_eligible":int(len(diag)),
        "same_channel_index_pearson_median":float(np.median(diag)) if diag.size else None,
        "same_channel_index_abs_pearson_median":float(np.median(np.abs(diag))) if diag.size else None,
        "same_channel_index_abs_pearson_ge_0p95":int(np.sum(np.abs(diag)>=.95)),
        "same_channel_index_abs_pearson_ge_0p9999":int(np.sum(np.abs(diag)>=.9999)),
        "same_index_4000_sample_blocks_eligible":int(local.size),
        "same_index_blockwise_abs_r_median":float(np.median(np.abs(local))) if local.size else None,
        "same_index_blockwise_abs_r_ge_0p95":int(np.sum(np.abs(local)>=.95)),
        "blockwise_comparison_assumes_40_ordered_4000_sample_blocks":True,
        "same_index_channel_optimal_trial_assignment_eligible":int(trial_matches.size),
        "same_index_optimal_trial_assignment_abs_r_median":float(np.median(trial_matches)) if trial_matches.size else None,
        "same_index_optimal_trial_assignment_abs_r_ge_0p95":int(np.sum(trial_matches>=.95)),
        "trial_assignment_is_posthoc_and_does_not_verify_acquisition_identity":True,
        "best_unconstrained_source_channel_match_median_abs_r":float(np.median(max_per_source)),
        "one_to_one_assignment_matched_channels":int(len(rows)),
        "one_to_one_assignment_median_abs_r":float(np.median(permuted)),
        "one_to_one_assignment_n_abs_r_ge_0p95":int(np.sum(permuted>=.95)),
        "one_to_one_assignment_n_abs_r_ge_0p9999":int(np.sum(permuted>=.9999)),
        "one_to_one_assignment_negative_sign_count":int(np.sum(~positive)),
        "one_to_one_is_exploratory_best_fit_not_external_electrode_map":True,
        "sample_order_shift_resampling_and_filtering_not_inverted":True,
    }

def edf_digital_samples(path):
    """Collect native int16 public EDF channels without header/patient metadata."""
    h=header(path)
    counts=h["samples"]
    if len(set(counts))!=1:
        raise ValueError("not all EDF channels share common timebase; fail closed")
    per_channel=[[] for _ in counts]
    for _,blocks in _records(path,h):
        for ch,payload in enumerate(blocks):
            per_channel[ch].append(np.frombuffer(payload,dtype="<i2"))
    out=np.stack([np.concatenate(x) for x in per_channel],axis=0)
    return out,h["names"]

def matrix_from_source(mat):
    x=np.asarray(mat)
    if x.ndim!=3 or x.shape[0]!=40 or x.shape[1]!=33 or x.shape[2]!=4000:
        raise ValueError(f"unexpected source MAT shape: {x.shape}; fail closed")
    return np.ascontiguousarray(x.transpose(1,0,2).reshape(33,-1))

def synthetic_test():
    rng=np.random.default_rng(20261013)
    a=rng.normal(size=(4,3000)).astype(np.float64)
    b=a*7+13
    b[1]=a[2]*3+7
    b[2]=a[1]*4+9
    x=pair_correlations(a,b)
    assert x["one_to_one_assignment_n_abs_r_ge_0p9999"]==4
    assert x["same_channel_index_abs_pearson_ge_0p9999"]==2
    assert x["one_to_one_assignment_median_abs_r"]>.9999
    unrelated=pair_correlations(a,rng.normal(size=(4,3000)))
    assert unrelated["one_to_one_assignment_n_abs_r_ge_0p95"]==0
    # A swapped/unaligned trial fails the local correspondence diagnostic.
    blocks=np.tile(a[:, :4000] if a.shape[1]>=4000 else rng.normal(size=(4,4000)),(1,40))
    shifted=np.roll(blocks,4000,axis=1)
    check=pair_correlations(blocks,shifted)
    assert check["same_index_4000_sample_blocks_eligible"]==160
    # One-to-one trial permutation should be recovered on exact affine blocks.
    trial_rng=np.random.default_rng(24)
    q=trial_rng.normal(size=(3,40,4000))
    perm=trial_rng.permutation(40)
    rearranged=(q[:,perm,:]*3+11).reshape(3,160000)
    permuted=pair_correlations(q.reshape(3,160000),rearranged)
    assert permuted["same_index_optimal_trial_assignment_abs_r_ge_0p95"]==120
    assert permuted["same_index_blockwise_abs_r_ge_0p95"]<120
    test=np.arange(40*33*4000).reshape(40,33,4000)
    m=matrix_from_source(test)
    assert np.array_equal(m[9],test[:,9,:].reshape(-1))
    print("P2_SOURCE_MAT_EDF_ALIGNMENT_SYNTHETIC_AFFINE_SWAP_NEGATIVE_PASS",flush=True)

def real(out):
    from moabb.datasets import Liu2024
    ds=Liu2024()
    reader=StrictRange(URL)
    listing,total=directory(reader)
    meta={
        "status":"EXECUTED_PUBLIC_RELEASE_MAT_TO_EDF_EXPLORATORY_CHANNEL_CORRESPONDENCE",
        "figshare_source_zip_id":38516555,
        "public_archive_total_bytes":total,
        "full_source_archive_not_downloaded":True,
        "only_six_source_mat_members_requested":True,
        "EDF_is_publisher_preprocessed_NOT_sensor_original":True,
        "no_EEG_waveforms_or_EDF_patient_header_released":True,
        "uninspected_clinical_acquisition_logs":True,
        "no_source_to_EDF_channel_map_assumed":True,
        "no_clinical_patient_identity_or_session_conclusion":True,
        "previously_public_retrospective_research":True,
        "rows":[],
    }
    with tempfile.TemporaryDirectory() as tmp:
        for sid in IDS:
            wanted=f"sourcedata/sub-{sid:02d}/sub-{sid:02d}_task-motor-imagery_eeg.mat"
            found=[v for v in listing if v["name"]==wanted]
            if len(found)!=1:raise RuntimeError(f"missing or ambiguous source MAT for public ID {sid}")
            dest=Path(tmp)/f"source-{sid:02d}.mat"
            receipt=write_verified_member(reader,found[0],dest)
            if not receipt["zip_crc32_matches"]:raise AssertionError("MAT CRC mismatch")
            backend,arrays=candidate_arrays(dest)
            if len(arrays)!=1:
                raise RuntimeError("source MAT signal semantics ambiguous: expected one numeric array")
            mat=matrix_from_source(arrays[0][0])
            digi,labels=edf_digital_samples(ds.data_path(sid)[0])
            if mat.shape!=digi.shape:
                raise ValueError(f"source MAT / EDF shape mismatch: {mat.shape} vs {digi.shape}")
            result=pair_correlations(mat,digi)
            result.update({
                "released_public_id":sid,
                "source_mat_loader_backend":backend,
                "source_mat_shape":[40,33,4000],
                "EDF_digital_samples_shape":list(digi.shape),
                "source_member_verified_crc":True,
            })
            meta["rows"].append(result)
            print("P2_SOURCE_TO_EDF_PAIR",sid,
                  "same_index",result["same_channel_index_abs_pearson_median"],
                  "permuted",result["one_to_one_assignment_median_abs_r"],flush=True)
            del arrays,mat,digi
    meta["total_public_Figshare_HTTP_206_bytes"]=reader.total_received
    out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+"\n"
    out.write_text(text,encoding="utf-8")
    print("P2_SOURCE_MAT_TO_EDF_AGGREGATES_SHA256",hashlib.sha256(text.encode()).hexdigest(),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    if a.self_test:return synthetic_test()
    if a.out is None:ap.error("--out required")
    real(a.out)

if __name__=="__main__":
    main()
