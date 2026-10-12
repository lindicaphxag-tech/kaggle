#!/usr/bin/env python3
"""Liu2024 public EEG near-duplicate screening on 50 digital records.

Robustness DESCRIPTIVE ONLY: aligned trials under a pinned pipeline.
Not patient identity, not all possible time shifts/resampling changes,
not full-epoch pairwise matching under permutation. No classifier AUC.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from moabb.datasets import Liu2024
from moabb.paradigms import MotorImagery

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    ds=Liu2024()
    pm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                    fmin=8,fmax=30,resample=160,baseline=None,
                    reject_by_annotation=False)
    vec=[];ns=[];receipts=[];label_vectors=[];ref_shape=None;label_pattern=None
    for sid in range(1,51):
        X,lab,_=pm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);lab=np.asarray(lab)
        if X.ndim!=3 or X.shape[0]!=40 or X.shape[1]!=29 or not np.isfinite(X).all():
            raise RuntimeError("public input schema drift")
        this=(lab=="right_hand").astype(np.uint8)
        if int(this.sum())!=20:raise ValueError("label imbalance drift")
        if ref_shape is None:ref_shape=X.shape
        if X.shape!=ref_shape:raise ValueError("cross-ID epoch shape mismatch")
        if label_pattern is None:label_pattern=this
        # If event sequences differ, retain content similarity but DO NOT call
        # it matching labelled trials: separately report label concordance.
        V=X.reshape(-1)
        V=V-np.mean(V)
        norm=float(np.linalg.norm(V))
        if not np.isfinite(norm) or norm==0:raise ValueError("empty digital EEG signal")
        vec.append(V/norm);ns.append(norm)
        label_vectors.append(this.copy())
        receipts.append({"id":sid,"label_vector_equals_id1":bool(np.array_equal(this,label_pattern)),
                         "n_trials":int(len(this))})
        print("P2_NEAR_DUPLICATE_PUBLIC_SIGNAL_LOADED",sid,flush=True)
    arr=np.asarray(vec,dtype=np.float64)
    cos=arr@arr.T
    norms=np.asarray(ns)
    pairs=[]
    for i in range(50):
        for j in range(i+1,50):
            c=float(np.clip(cos[i,j],-1.,1.))
            # Scale-invariant correlation of trial-aligned whole-record vectors:
            # does not establish equality of raw digital samples.
            sim=c
            if c>=0.95:
                pairs.append({"ids":[i+1,j+1],"whole_record_centered_cosine":sim,
                   "matches_nominal_trial_label_sequence":
                       bool(np.array_equal(label_vectors[i],label_vectors[j]))})
    pairs.sort(key=lambda z:(-z["whole_record_centered_cosine"],z["ids"]))
    high=[x for x in pairs if x["whole_record_centered_cosine"]>=0.9999]
    # Exact released ID4/5 is already established with strict array equality
    # elsewhere; cosine of 1 is only a diagnostic here, not the exact proof.
    d={"status":"P2_PUBLIC_50_ALIGNED_WHOLE_RECORD_NEAR_DUPLICATE_SCREEN",
       "dataset":"MOABB 1.7.2 Liu2024 public EEG",
       "processing":"8-30 Hz,160 Hz,29 EEG channels,40 epochs per ID",
       "threshold_initiated_before_result_view":"centered whole-record cosine >= 0.9999",
       "screened_pairs":1225,
       "n_pairs_ge_0p95":len(pairs),"n_pairs_ge_0p9999":len(high),
       "high_similarity_candidate_pairs":high,
       "top_five_pairs_ge_0p95":pairs[:5],
       "no_cross_record_epoch_reordering_or_time_shift_test":True,
       "no_personal_human_identity_inference":True,
       "target_auc_never_read":True,
       "not_independent_new_EEG_acquisition":True,
       "cannot_evaluate_underlying_original_patient_identity":True}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n"
    a.out.write_text(body)
    print("P2_PUBLIC_NEAR_DUPLICATE_SCREEN_SUMMARY",
          json.dumps({"high_similarity_candidate_pairs":high,
              "n_pairs_ge_0p95":len(pairs),"n_pairs_ge_0p9999":len(high)},sort_keys=True),flush=True)
    print("RECEIPT_SHA",hashlib.sha256(body.encode()).hexdigest())
if __name__=="__main__":main()
