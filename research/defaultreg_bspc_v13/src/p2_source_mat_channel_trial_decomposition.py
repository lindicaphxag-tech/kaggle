#!/usr/bin/env python3
"""Liu2024 original source MAT EEG trial x channel decomposition (public release).

Use six exact-name MAT ZIP members through strict, range-only public reads.
The prior overall 40x33x4000 array correlations are insufficient: this
independent per-channel/per-trial analysis checks whether one high-energy
channel dominates global similarity. No source MAT arrays, clinical
metadata, or subject identifiers beyond released public ID integers are
published; only aggregated diagnostics.

The source MAT arrays have not been independently mapped to the EDF's
channel metadata. Until mapping is verified, index positions are NOT
labelled clinical EEG channels. High similarity does not identify people.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from p2_figshare_six_source_mat_range_audit import (
    IDS, PAIRS, URL, StrictRange, directory, write_verified_member, candidate_arrays
)

def corr(x,y):
    a=np.asarray(x,dtype=np.float64).reshape(-1)
    b=np.asarray(y,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or len(a)<3: return None
    a=a-a.mean()
    b=b-b.mean()
    na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b))
    if na<=1e-16 or nb<=1e-16:return None
    return float(np.clip(np.dot(a,b)/na/nb,-1.0,1.0))

def compare(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[0]!=40 or a.shape[1]!=33 or a.shape[2]!=4000:
        raise ValueError("source original MAT candidate shape drift; stop without fabricated alignment")
    if not(np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("source MAT signal array contains nonfinite values")
    ntrial,nchan,ntime=a.shape
    channel=[]
    epoch_exact=[]
    epoch_90_corr=[]
    epoch_90_exact=[]
    channel_block_exact=0;channel_block_near=0;nonconstant=0
    sse_a=[];sse_b=[];means_a=[];means_b=[];within_covs=[]
    for j in range(nchan):
        x=a[:,j,:].astype(np.float64)
        y=b[:,j,:].astype(np.float64)
        v_a=float(np.mean((x-x.mean())**2))
        v_b=float(np.mean((y-y.mean())**2))
        r=corr(x,y)
        exact=bool(np.array_equal(x,y))
        channel.append({
            "source_mat_channel_index":j,
            "aligned_numeric_channel_full_exact":exact,
            "channel_full_pearson":r,
            "rms_signal_a":float(np.sqrt(np.mean(x**2))),
            "rms_signal_b":float(np.sqrt(np.mean(y**2))),
        })
        sse_a.append(v_a);sse_b.append(v_b)
        means_a.append(float(x.mean()));means_b.append(float(y.mean()))
        within_covs.append(float(np.mean((x-x.mean())*(y-y.mean()))))
    for trial in range(ntrial):
        exact=0;near=0;nonconst=0
        for j in range(nchan):
            x=a[trial,j,:];y=b[trial,j,:]
            r=corr(x,y)
            if r is None:continue
            nonconst+=1
            e=bool(np.array_equal(x,y))
            exact+=int(e);near+=int(r>=.9999)
        channel_block_exact+=exact;channel_block_near+=near;nonconstant+=nonconst
        epoch_exact.append(exact)
        epoch_90_exact.append(nonconst>0 and exact/nonconst>=.9)
        epoch_90_corr.append(nonconst>0 and near/nonconst>=.9)
    cc=[v["channel_full_pearson"] for v in channel if v["channel_full_pearson"] is not None]
    def longest(flags):
        out=run=0
        for x in flags:
            run=(run+1) if x else 0
            out=max(out,run)
        return out
    variances=np.asarray(sse_a,dtype=np.float64)
    ma=np.asarray(means_a,dtype=np.float64)
    mb=np.asarray(means_b,dtype=np.float64)
    within_cov=float(np.mean(within_covs))
    within_variance_a=float(np.mean(sse_a))
    within_variance_b=float(np.mean(sse_b))
    between_variance_a=float(np.var(ma))
    between_variance_b=float(np.var(mb))
    between_covariance=float(np.mean((ma-ma.mean())*(mb-mb.mean())))
    within_denom=np.sqrt(within_variance_a*within_variance_b)
    within_r=within_cov/within_denom if within_denom>0 else None
    between_frac_a=between_variance_a/(between_variance_a+within_variance_a)
    between_frac_b=between_variance_b/(between_variance_b+within_variance_b)
    out={
        "source_mat_candidate_shape":list(a.shape),
        "globally_aligned_numeric_pearson":corr(a,b),
        "within_channel_centered_pooled_pearson":float(np.clip(within_r,-1,1)) if within_r is not None else None,
        "between_channel_mean_variance_fraction_source_a":float(between_frac_a),
        "between_channel_mean_variance_fraction_source_b":float(between_frac_b),
        "between_channel_mean_covariance":between_covariance,
        "within_channel_mean_covariance":within_cov,
        "number_constant_source_mat_channel_indices":sum(z["channel_full_pearson"] is None for z in channel),
        "exact_full_array":bool(np.array_equal(a,b)),
        "n_nonconstant_trial_channel_blocks":nonconstant,
        "exact_trial_channel_blocks":channel_block_exact,
        "corr_ge_0p9999_trial_channel_blocks":channel_block_near,
        "n_source_mat_channels":nchan,
        "channel_full_pearson_median":float(np.median(cc)) if cc else None,
        "channel_full_pearson_min":float(np.min(cc)) if cc else None,
        "channel_full_pearson_p10":float(np.quantile(cc,.10)) if cc else None,
        "channel_full_pearson_ge_0p9999":sum(r is not None and r>=.9999 for r in (x["channel_full_pearson"] for x in channel)),
        "channel_full_exact_count":sum(v["aligned_numeric_channel_full_exact"] for v in channel),
        "fraction_channel_centered_variance_from_single_largest":float(np.max(variances)/np.sum(variances)),
        "n_source_mat_trials_90percent_channels_corr_ge_0p9999":sum(epoch_90_corr),
        "n_source_mat_trials_90percent_channels_exact":sum(epoch_90_exact),
        "longest_consecutive_trials_90percent_channels_corr_ge_0p9999":longest(epoch_90_corr),
        "trial_median_exact_channels":float(np.median(epoch_exact)),
        "per_source_mat_channel_aggregates":channel,
        "no_EDF_mat_channel_map_claim":True,
        "no_patient_identification_or_independent_acquisition_claim":True,
        "provenance_scope":"original released source MAT numerical arrays, not original sensor acquisition logs"
    }
    return out

def self_test():
    rng=np.random.default_rng(7)
    a=rng.standard_normal((40,33,4000)).astype(np.float64)
    assert compare(a,a)["exact_trial_channel_blocks"]==40*33
    b=a.copy();b[:,0,:]+=10
    x=compare(a,b)
    assert not x["exact_full_array"]
    assert x["channel_full_exact_count"]==32
    assert x["n_source_mat_trials_90percent_channels_exact"]==40
    c=rng.standard_normal((40,33,4000))
    z=compare(a,c)
    assert z["channel_full_pearson_ge_0p9999"]==0
    # Global covariance can be dominated by one extreme channel,
    # but the per-channel median resists this confound.
    huge=rng.standard_normal((40,33,4000))
    v=rng.standard_normal((40,33,4000))
    huge[:,0,:]*=1e7;v[:,0,:]=huge[:,0,:]
    q=compare(huge,v)
    assert q["globally_aligned_numeric_pearson"]>.99
    assert q["fraction_channel_centered_variance_from_single_largest"]>.99
    assert q["channel_full_pearson_median"]<.25
    offset1=rng.standard_normal((40,33,4000))
    offset2=rng.standard_normal((40,33,4000))
    offset1[:,17,:]+=187500.0
    offset2[:,17,:]+=187500.0
    fake=compare(offset1,offset2)
    assert fake["globally_aligned_numeric_pearson"]>.9999
    assert abs(fake["within_channel_centered_pooled_pearson"])<.1
    assert fake["between_channel_mean_variance_fraction_source_a"]>.999
    print("P2_SOURCE_MAT_CHANNEL_TRIAL_FALSE_GLOBAL_CORRELATION_SELFTEST_PASS",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test();return
    if args.out is None:p.error("--out required")
    reader=StrictRange(URL)
    entries,total=directory(reader)
    report={
        "status":"EXECUTED_SOURCE_MAT_ORIGINAL_RELEASE_CHANNEL_TRIAL_DIAGNOSTICS",
        "official_figshare_version_file_id":38516555,
        "full_original_release_source_zip_bytes":total,
        "full_archive_not_downloaded":True,
        "six_public_files_only":True,
        "source_mat_to_edf_channel_map_unverified":True,
        "pre_selected_pairs_from_prior_public_source_audit":True,
        "previously_seen_public_data_not_prospective_validation":True,
        "no_EEG_data_or_patient_metadata_in_artifacts":True,
        "pair_comparisons":[],
    }
    with tempfile.TemporaryDirectory() as td:
        files={}
        for i in IDS:
            filename=f"sourcedata/sub-{i:02d}/sub-{i:02d}_task-motor-imagery_eeg.mat"
            matching=[m for m in entries if m["name"]==filename]
            if len(matching)!=1:raise ValueError("ambiguous official MAT public filename")
            path=Path(td)/f"source-{i}.mat"
            receipt=write_verified_member(reader,matching[0],path)
            assert receipt["zip_crc32_matches"]
            files[i]=path
        for ia,ib in PAIRS:
            _,a=candidate_arrays(files[ia])
            _,b=candidate_arrays(files[ib])
            if len(a)!=1 or len(b)!=1:
                raise ValueError("expected one large signal-shaped numerical source MAT candidate per file")
            row=compare(a[0][0],b[0][0])
            row["released_public_ids"]=[ia,ib]
            report["pair_comparisons"].append(row)
            print("P2_MAT_CHANNEL_TRIAL",ia,ib,
                  "glob",row["globally_aligned_numeric_pearson"],
                  "medianch",row["channel_full_pearson_median"],
                  "dominance",row["fraction_channel_centered_variance_from_single_largest"],
                  "within_channel",row["within_channel_centered_pooled_pearson"],
                  "between_mean_variance_frac",row["between_channel_mean_variance_fraction_source_a"],
                  "near",row["channel_full_pearson_ge_0p9999"],flush=True)
    report["total_http_range_body_bytes"]=reader.total_received
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.write_text(body,encoding="utf-8")
    print("P2_MAT_CHANNEL_TRIAL_AGGREGATES_SHA256",hashlib.sha256(body.encode()).hexdigest(),flush=True)

if __name__=="__main__":main()
