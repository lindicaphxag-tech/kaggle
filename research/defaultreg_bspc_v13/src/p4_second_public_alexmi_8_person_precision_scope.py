#!/usr/bin/env python3
"""Cross-dataset REAL EEG final-tangent binary64 quantization-sign audit.

Frozen protocol: AlexMI released 1..8 (same already project-examined
dataset), right-hand vs feet, 2 electrodes C3/C4, 8-30Hz, 160Hz.
Take first 20 / class and fixed 8 positive-negative pairs per person.
Exactly 4 final-feature quantization resolutions (18,10,6,4 bits)
are PRE-FIXED IN CODE from earlier BNCI2014_001 precision screening.
Evaluate whole-lambda conditional rank-sign on all 8 x 8 x 4 = 256
person/pair/precision conditions. No target selection or AUC claim.
The EEG preprocessing is NOT included in the verified uncertainty box.
"""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from moabb.datasets import AlexMI
from moabb.paradigms import MotorImagery
import p4_real_eeg_final_feature_precision_budget_curve as source
from p4_final_feature_entrywise_error_to_covariance_certificate import (
    fit_pooled,error_bound
)
from p4_covariance_coordinate_resolvent_bound import certify

SUBJECTS=tuple(range(1,9))
N=20
PAIR_N=8
BITS=(18,10,6,4)
SEED=20261014
TASKS=("right_hand","feet")
ELECTRODES=("C3","C4")

def run():
    ds=AlexMI()
    paradigm=MotorImagery(n_classes=2,events=list(TASKS),
        channels=list(ELECTRODES),fmin=8,fmax=30,resample=160,
        baseline=None,reject_by_annotation=False)
    out=[]
    for sid in SUBJECTS:
        epochs,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
        epochs=np.asarray(epochs,dtype=np.float64)
        labels=np.asarray(labels)
        if epochs.ndim!=3 or epochs.shape[1]!=2 or len(epochs)!=40:
            raise ValueError("Unexpected public AlexMI 40-epoch geometry")
        if set(labels.tolist())!=set(TASKS) or not np.isfinite(epochs).all():
            raise ValueError("Unexpected binary public-task labels")
        y=np.asarray(labels=="right_hand",dtype=int)
        if int(y.sum())!=20:
            raise ValueError("Unexpected 20-per-class release")
        take=np.r_[np.flatnonzero(y==0)[:N],np.flatnonzero(y==1)[:N]]
        C=source.base.old.covariances(epochs[take])
        ref=source.base.old.reference(C)
        features=np.asarray(source.base.old.tangent(C,ref),dtype=np.float64)
        if features.shape!=(40,3) or not np.isfinite(features).all():
            raise ValueError("Need valid three tangent final features")
        rng=np.random.default_rng(SEED+sid)
        pairs=list(zip(rng.choice(np.arange(N,2*N),PAIR_N,replace=False),
                       rng.choice(np.arange(N),PAIR_N,replace=False)))
        subject={
          "released_public_id":sid,"n_original_epochs":int(len(epochs)),
          "n_used_balanced_trials":2*N,"n_final_tangent_features":3,
          "pair_count":PAIR_N,"fixed_predecision_pair_indices":[
             [int(a),int(b)] for a,b in pairs],
          "fixed_input_feature_binary64_sha256":hashlib.sha256(
             np.ascontiguousarray(features).tobytes()).hexdigest(),
          "precision_checks":[]}
        for bits in BITS:
            q,eta,observed=source.quantize(features,bits)
            assert observed<=eta
            S,d,_=fit_pooled(q,[0]*N+[1]*N,F(1,5))
            epsS,epsD,epsV=error_bound(q,[0]*N+[1]*N,eta)
            counts={"POS":0,"NEG":0,"UNKNOWN":0}
            details=[]
            for idx,(p,n) in enumerate(pairs):
                v=(q.row(int(p))-q.row(int(n))).T
                answer=certify(S,v,d,epsS,epsV,epsD,"0","1")
                status=answer["status"]
                counts[status]+=1
                details.append({
                  "frozen_pair_index":idx,"status":status,
                  "whole_lambda_lower_margin":answer["whole_interval_base_margin_lower"],
                  "all_input_box_score_error_upper":answer["whole_interval_score_error_upper"]})
            row={"fractional_bits":bits,"eta":str(eta),
                 "observed_max_exact_input_roundoff":str(observed),
                 "exact_final_binary64_entrywise_error_verified":True,
                 "counts":counts,
                 "certified_positive_or_negative":counts["POS"]+counts["NEG"],
                 "sampled_pairs":details}
            if sum(counts.values())!=PAIR_N:raise AssertionError("pair accounting")
            subject["precision_checks"].append(row)
        out.append(subject)
        print("P4_ALEXMI_EXTERNAL_DATASET_FINAL_FEATURE_EIGHT_PERSON",sid,
              json.dumps([{"bits":x["fractional_bits"],"counts":x["counts"]}
                           for x in subject["precision_checks"]],sort_keys=True),flush=True)
    pooled={}
    for bits in BITS:
        cc={k:sum(person["precision_checks"][BITS.index(bits)]["counts"][k]
                 for person in out)
            for k in ("POS","NEG","UNKNOWN")}
        pooled[str(bits)]={"counts":cc,"certified":cc["POS"]+cc["NEG"],
                          "person_count":len(out),"pairs_per_person":PAIR_N}
    assert all(sum(pooled[str(b)]["counts"].values())==8*PAIR_N for b in BITS)
    return {
     "status":"SECOND_PUBLIC_EEG_ACQUISITION_REAL_FINAL_FEATURE_NUMERIC_SCOPE",
     "dataset":"AlexMI (previously project-examined, not unseen cohort)",
     "source_dataset_of_prior_precision_curve":"BNCI2014_001",
     "n_released_people":8,"all_in_same_AlexMI_public_acquisition":True,
     "n_fixed_positive_negative_pair_per_person":PAIR_N,
     "total_pair_precision_conditions":len(SUBJECTS)*PAIR_N*len(BITS),
     "frozen_bit_grid":list(BITS),
     "only_final_binary64_tangent_features_certified":True,
     "all_roundoff_exact_rational_after_final_features":True,
     "no_preprocessing_MNE_covariance_or_tangent_machine_roundoff_guarantee":True,
     "not_independent_trial_groups_or_personality":True,
     "no_AUC_or_cross_subject_prediction_evaluated":True,
     "no_external_preregistered_protocol":True,
     "person_and_precision_results":out,
     "aggregated_descriptive_precision_counts":pooled
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args();z=run()
    raw=json.dumps(z,sort_keys=True,indent=2,allow_nan=False)+"\n"
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(raw)
    print("P4_ALEXMI_PUBLIC_FINAL_FEATURE_ALL_8_REAL_PERSON_SCOPE",
          json.dumps(z["aggregated_descriptive_precision_counts"],sort_keys=True),flush=True)
    print("P4_ALEXMI_REAL_FEATURE_SHA256",hashlib.sha256(raw.encode()).hexdigest())
