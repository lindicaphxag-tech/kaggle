#!/usr/bin/env python3
"""P2 REAL previously exposed public EEG near-overlap negative controls.

Screen source uses the same ALIGNED whole-record centered Pearson cutoff as
the Liu2024 exploratory screen. Weibo2014 (10 IDs), BNCI2014_002 (14 IDs).
No AUC, no target label outcome, no claim about patients or unseen cohort.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from moabb.datasets import Weibo2014,BNCI2014_002
from moabb.paradigms import MotorImagery
DATASETS={
 "Weibo2014":{"ds":Weibo2014,"ids":tuple(range(1,11)),
               "events":["left_hand","right_hand"],"positive":"right_hand",
               "channels":["C3","Cz","C4"],"expected_channels":3},
 "BNCI2014_002":{"ds":BNCI2014_002,"ids":tuple(range(1,15)),
               "events":["right_hand","feet"],"positive":"feet",
               "channels":None,"expected_channels":15}
}

def one(name,config):
    c=config
    kw=dict(n_classes=2,events=c["events"],fmin=8,fmax=30,
            resample=160,baseline=None,reject_by_annotation=False)
    if c["channels"] is not None:kw["channels"]=c["channels"]
    pr=MotorImagery(**kw);ds=c["ds"]()
    records={}
    for sid in c["ids"]:
        X,label,_=pr.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);lab=np.asarray(label)
        y=(lab==c["positive"]).astype(np.uint8)
        if X.ndim!=3 or X.shape[1]!=c["expected_channels"] or set(lab.tolist())!=set(c["events"]) or not np.isfinite(X).all():
            raise RuntimeError(f"{name} release/schema drift on {sid}")
        V=np.asarray(X,dtype=np.float64).ravel()
        V=V-V.mean();norm=np.linalg.norm(V)
        if norm<=0:raise RuntimeError("empty EEG")
        records[sid]={"vector":V/norm,"shape":X.shape,"labels":y}
        print("P2_NEGATIVE_COHORT_NEAR_RECORD",name,sid,len(y),flush=True)
    near=[];skipped=0;highest=None;npairs=0
    ids=list(c["ids"])
    for ii,a in enumerate(ids):
        for b in ids[ii+1:]:
            A,B=records[a],records[b]
            if A["shape"]!=B["shape"]:
                skipped+=1;continue
            npairs+=1
            correlation=float(np.clip(np.dot(A["vector"],B["vector"]),-1.,1.))
            candidate={"released_ids":[a,b],"full_record_centered_pearson":correlation,
                  "same_trial_label_sequence":bool(np.array_equal(A["labels"],B["labels"]))}
            if highest is None or correlation>highest["full_record_centered_pearson"]:highest=candidate
            if correlation>=.95:near.append(candidate)
    near.sort(key=lambda x:-x["full_record_centered_pearson"])
    return {"dataset":name,"n_released_ids":len(ids),"pairs_possible":len(ids)*(len(ids)-1)//2,
         "pairs_scanned_aligned":npairs,"pairs_shape_incompatible":skipped,
         "thresholds":[.95,.9999],
         "n_pairs_ge_0p95":len(near),
         "n_pairs_ge_0p9999":sum(x["full_record_centered_pearson"]>=.9999 for x in near),
         "candidates_ge_0p95":near[:30],
         "maximum_aligned_pair":highest,
         "not_independent_first_time_EEG_acquisition":True,
         "does_not_rule_out_unaligned_or_reprocessed_duplicates":True,
         "no_target_AUC_or_decoding_outcome_used":True}

def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    results={k:one(k,v) for k,v in DATASETS.items()}
    report={"status":"PUBLIC_EEG_PREVIOUSLY_EXPOSED_ALIGNED_NEAR_RECORD_CONTROL_AUDIT",
       "processing":"MOABB1.7.2; 8–30Hz; 160Hz; frozen native/channel selection",
       "threshold_same_as_Liu2024_screen":True,
       "no_physical_patient_identity_inference":True,
       "dataset_control_results":results}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    args.out.write_text(body)
    print("P2_TWO_COHORT_REAL_NEAR_OVERLAP_SCREEN_RESULT",json.dumps({
       n:{k:v[k] for k in ("pairs_possible","pairs_scanned_aligned","pairs_shape_incompatible",
            "n_pairs_ge_0p95","n_pairs_ge_0p9999","maximum_aligned_pair")} for n,v in results.items()
    },sort_keys=True),flush=True)
    print("P2_TWO_CONTROLS_SHA256",hashlib.sha256(body.encode()).hexdigest(),flush=True)
if __name__=="__main__":main()
