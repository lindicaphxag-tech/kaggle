#!/usr/bin/env python3
"""Held-out temporal test of numeric MAT/EDF correspondence.

Matches trial order using the FIRST 2000 samples only, then evaluates
precommitted matches on the disjoint LAST 2000 samples. Same-array
channel indices are exploratory; no electrode, participant, session or
acquisition identity is inferred. No raw waveforms are exported.
"""
from __future__ import annotations
import argparse,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from p2_figshare_six_source_mat_range_audit import IDS,StrictRange,URL,directory,write_verified_member,candidate_arrays


def standardized(blocks):
    x=np.asarray(blocks,dtype=np.float64)
    x=x-x.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(x,axis=1)
    if np.any(norm==0):return x,norm
    return x/norm[:,None],norm


def heldout_pair(mat,edf):
    a=np.asarray(mat,dtype=np.float64);b=np.asarray(edf,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=40*4000:
        raise ValueError('expected same-shape 40x4000 sampled release products')
    train=[];test=[];valid=0
    for ch in range(a.shape[0]):
        u=a[ch].reshape(40,4000);v=b[ch].reshape(40,4000)
        A,na=standardized(u[:,:2000]);B,nb=standardized(v[:,:2000])
        if np.any(na==0) or np.any(nb==0):continue
        corr=A@B.T
        rr,cc=linear_sum_assignment(-np.abs(corr))
        C,nc=standardized(u[:,2000:]);D,nd=standardized(v[:,2000:])
        good=(nc[rr]>0)&(nd[cc]>0)
        if not good.any():continue
        train.extend(np.abs(corr[rr[good],cc[good]]).tolist())
        test.extend(np.abs(np.sum(C[rr[good]]*D[cc[good]],axis=1)).tolist())
        valid+=int(good.sum())
    tr=np.asarray(train);te=np.asarray(test)
    if valid==0:raise RuntimeError('no nonconstant channel trial comparisons')
    return {'eligible_channel_trial_matches':valid,
            'train_first_2000_median_abs_r':float(np.median(tr)),
            'heldout_last_2000_median_abs_r':float(np.median(te)),
            'heldout_last_2000_abs_r_ge_0p95':int(np.sum(te>=.95)),
            'heldout_last_2000_abs_r_ge_0p9999':int(np.sum(te>=.9999)),
            'trial_permutation_fitted_only_on_first_half':True,
            'channel_index_assumed_same_for_exploratory_screen':True}


def selftest():
    rng=np.random.default_rng(20261013)
    q=rng.normal(size=(3,40,4000))
    perm=rng.permutation(40)
    transformed=(q[:,perm,:]*2.5+13).reshape(3,-1)
    x=heldout_pair(q.reshape(3,-1),transformed)
    assert x['heldout_last_2000_abs_r_ge_0p9999']==120
    unrelated=heldout_pair(q.reshape(3,-1),rng.normal(size=(3,40*4000)))
    assert unrelated['heldout_last_2000_abs_r_ge_0p95']==0
    print('P2_HELDOUT_TEMPORAL_MATCH_SYNTHETIC_PASS',flush=True)


def real(out):
    from moabb.datasets import Liu2024
    from p2_source_mat_to_edf_crosslayer_alignment import matrix_from_source,edf_digital_samples
    ds=Liu2024();reader=StrictRange(URL);listing,total=directory(reader)
    mat={};edf={};receipts=[]
    with tempfile.TemporaryDirectory() as tmp:
        for sid in IDS:
            name=f'sourcedata/sub-{sid:02d}/sub-{sid:02d}_task-motor-imagery_eeg.mat'
            found=[x for x in listing if x['name']==name]
            if len(found)!=1:raise RuntimeError('missing/ambiguous source member')
            dest=Path(tmp)/f'{sid}.mat'
            receipt=write_verified_member(reader,found[0],dest)
            if not receipt['zip_crc32_matches']:raise AssertionError('CRC failure')
            backend,arrays=candidate_arrays(dest)
            if len(arrays)!=1:raise RuntimeError('ambiguous MAT signal candidate')
            mat[sid]=matrix_from_source(arrays[0][0])
            edf[sid],_=edf_digital_samples(ds.data_path(sid)[0])
            if mat[sid].shape!=edf[sid].shape:raise RuntimeError('shape mismatch')
            receipts.append({'public_id':sid,'source_member_crc_verified':True,
                             'source_backend':backend,'shape':list(mat[sid].shape)})
            print('P2_HELDOUT_LAYER_PRODUCTS_LOADED',sid,flush=True)
    rows=[]
    for src in IDS:
        for dst in IDS:
            result=heldout_pair(mat[src],edf[dst])
            rows.append({'source_mat_public_id':src,'processed_edf_public_id':dst,**result})
            print('P2_HELDOUT_LAYER_PAIR',src,dst,
                  'train_median',result['train_first_2000_median_abs_r'],
                  'heldout_median',result['heldout_last_2000_median_abs_r'],
                  'heldout_ge95',result['heldout_last_2000_abs_r_ge_0p95'],
                  'heldout_ge9999',result['heldout_last_2000_abs_r_ge_0p9999'],
                  'eligible',result['eligible_channel_trial_matches'],flush=True)
    payload={'status':'PUBLIC_RELEASE_CROSS_LAYER_HELDOUT_TEMPORAL_SCREEN',
             'source_figshare_file_id':38516555,
             'source_archive_size_bytes':total,
             'requested_source_ids':list(IDS),'all_six_by_six_release_id_pairs':True,
             'source_member_receipts':receipts,'rows':rows,
             'reader_http_bytes_received':reader.total_received,
             'no_signal_arrays_or_patient_metadata_exported':True,
             'first_half_fit_second_half_evaluation_only':True,
             'heldout_halves_not_independent_acquisitions':True,
             'unknown_channel_map_reference_filter_and_processing_order':True,
             'cannot_establish_raw_acquisition_or_clinical_patient_identity':True,
             'screen_thresholds_descriptive_not_external_error_calibrated':True}
    out.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();out.write_bytes(data)
    print('P2_HELDOUT_CROSSLAYER_SHA256',hashlib.sha256(data).hexdigest(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.self_test:selftest()
    else:
        if args.out is None:parser.error('--out required')
        real(args.out)
