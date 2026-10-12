#!/usr/bin/env python3
"""P2 real Liu2024 near-match forensic diagnostics on three screened pairs.

A discovery-follow-up in one already examined public dataset.
Intentionally does NOT classify near-matches as equal clinical patients.
Outputs scalar similarity/affine residual statistics only, no EEG arrays.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery
from p2_edf_digital_payload_audit import compare as compare_edf

PAIRS=((4,5),(47,49),(11,15))
NEGATIVE_CONTROL=((2,3),(6,7))
ALL=sorted({sid for p in PAIRS+NEGATIVE_CONTROL for sid in p})

def scalar_match(a,b):
    a,b=np.asarray(a,dtype=np.float64).reshape(-1),np.asarray(b,dtype=np.float64).reshape(-1)
    if a.shape!=b.shape or a.size<2:raise ValueError("unaligned EEG")
    ca=a-a.mean();cb=b-b.mean()
    ssaa=np.dot(ca,ca);ssbb=np.dot(cb,cb)
    if ssaa<=0 or ssbb<=0:raise ValueError("degenerate EEG")
    cosine=float(np.dot(ca,cb)/np.sqrt(ssaa*ssbb))
    slope=float(np.dot(ca,cb)/ssaa);offset=float(b.mean()-slope*a.mean())
    diff=cb-slope*ca
    return {"pearson":cosine,"least_squares_affine_slope_B_from_A":slope,
      "least_squares_affine_offset":offset,
      "relative_affine_residual_l2":float(np.linalg.norm(diff)/np.sqrt(ssbb)),
      "n_aligned_samples":int(len(a)),
      "exact_numeric_float64":bool(np.array_equal(a,b))}

def main():
    pa=argparse.ArgumentParser();pa.add_argument("--out",required=True,type=Path)
    args=pa.parse_args()
    ds=Liu2024()
    paradigm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                          fmin=8,fmax=30,resample=160,baseline=None,
                          reject_by_annotation=False)
    data={}
    for sid in ALL:
        X,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);y=np.asarray(labels=="right_hand",dtype=np.uint8)
        if X.ndim!=3 or X.shape[0]!=40 or X.shape[1]!=29 or int(y.sum())!=20 or not np.isfinite(X).all():
            raise RuntimeError("pinned public EEG schema drift")
        data[sid]=(X,y)
        print("P2_NEAR_ID_DIAGNOSTIC_READ",sid,flush=True)
    pairs=[]
    for ia,ib in PAIRS+NEGATIVE_CONTROL:
        A,ya=data[ia];B,yb=data[ib]
        if A.shape!=B.shape:raise ValueError("shape mismatch")
        all_=scalar_match(A,B)
        channel=[]
        for i in range(A.shape[1]):
            channel.append(scalar_match(A[:,i,:],B[:,i,:]))
        epochs=[scalar_match(A[i],B[i]) for i in range(len(A))]
        edf=compare_edf(ds.data_path(ia)[0],ds.data_path(ib)[0])
        pairs.append({
          "released_ids":[ia,ib],
          "selected_after_global_1225_pair_screening":(ia,ib) in PAIRS,
          "raw_edf_digital":edf,
          "n_original_trials":40,
          "n_original_channels":29,
          "labels_exact_sequence_match":bool(np.array_equal(ya,yb)),
          "all_processed_samples":all_,
          "channels_all_29_exact_numeric":bool(np.array_equal(A,B)),
          "n_processed_channels_exact_numeric":int(sum(np.array_equal(A[:,i,:],B[:,i,:]) for i in range(29))),
          "n_channels_pearson_ge_0p9999":int(sum(c["pearson"]>=.9999 for c in channel)),
          "min_channel_correlation":float(min(c["pearson"] for c in channel)),
          "median_channel_correlation":float(np.median([c["pearson"] for c in channel])),
          "max_channel_relative_affine_residual_l2":float(max(c["relative_affine_residual_l2"] for c in channel)),
          "mean_channel_relative_affine_residual_l2":float(np.mean([c["relative_affine_residual_l2"] for c in channel])),
          "min_epoch_correlation":float(min(x["pearson"] for x in epochs)),
          "median_epoch_correlation":float(np.median([x["pearson"] for x in epochs])),
          "max_epoch_relative_affine_residual_l2":float(max(x["relative_affine_residual_l2"] for x in epochs)),
          "not_a_clinical_patient_identity_conclusion":True
        })
        print("P2_NEAR_ID_PAIR_DIAGNOSTIC",json.dumps({
            "pair":[ia,ib],"processed":all_,
            "edf_channels_equal":edf["n_identical_digital_channels"],
            "edf_channels_total":edf["n_raw_channels"],
            "label_sequence_equal":bool(np.array_equal(ya,yb)),
            "min_channel_correlation":pairs[-1]["min_channel_correlation"],
            "max_epoch_residual_l2":pairs[-1]["max_epoch_relative_affine_residual_l2"]
          },sort_keys=True),flush=True)
    result={"status":"RETROSPECTIVE_LIU2024_SCALAR_AFFINE_AND_RAW_DIGITAL_NEARMATCH_AUDIT",
       "dataset":"Liu2024 MOABB 1.7.2 public EEG 8-30Hz 160Hz",
       "near_pairs_previously_selected_from_1225_pairs":[list(p) for p in PAIRS],
       "fixed_additional_controls":[list(p) for p in NEGATIVE_CONTROL],
       "measurement_scope":"trial-aligned numeric processed EEG and raw EDF digital int16 byte equality",
       "does_not_exhaust_cross_trial_alignments_or_time_shifted_signals":True,
       "not_new_independent_acquisition":True,"not_a_causal_or_patient_identity_claim":True,
       "pairs":pairs}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.write_text(body)
    print("P2_NEARMATCH_FORENSIC_SUMMARY",
          json.dumps({"pairs":[{"ids":p["released_ids"],"corr":p["all_processed_samples"]["pearson"],
                              "residual":p["all_processed_samples"]["relative_affine_residual_l2"],
                              "same_digital_channels":p["raw_edf_digital"]["n_identical_digital_channels"],
                              "same_labels":p["labels_exact_sequence_match"]}
                             for p in pairs]},sort_keys=True),flush=True)
    print("P2_FORENSIC_SHA256",hashlib.sha256(body.encode()).hexdigest(),flush=True)
if __name__=="__main__":main()
