#!/usr/bin/env python3
"""Two *previously exposed* EEG acquisition exact-epoch negative controls.

No classifier score, model fit, AUC, target info rule, new selection
or clinical result is calculated. Source is pinned public MOABB 1.7.2.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from moabb.datasets import Weibo2014
from moabb.paradigms import MotorImagery
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
sys.path.insert(0,str(ROOT/"research"/"defaultreg_bspc_v13"/"src"))
import original_frozen_runner as old
from exact_epoch_identity_v27 import audit_exact_epoch_overlap

PROTOCOL=ROOT/"research"/"defaultreg_bspc_v13"/"V28_TWO_EXTERNAL_EPOCH_CONTROLS_PREOUTCOME_PROTOCOL.md"

def cohort(name):
    if name=="Weibo2014":
        subjects=list(range(1,11))
        dataset=Weibo2014()
        events=["left_hand","right_hand"]
        opts=dict(channels=["C3","Cz","C4"])
        expected_channels=3
        positive="right_hand"
    elif name=="BNCI2014_002":
        s=old.dataset_spec("BNCI2014_002")
        subjects=list(s["subjects"])
        dataset=s["dataset"]
        events=list(s["events"])
        opts={}
        expected_channels=15
        positive=s["positive_label"]
    else:raise ValueError("unregistered dataset not allowed")
    paradigm=MotorImagery(n_classes=2,events=events,fmin=8,fmax=30,
                          resample=160,baseline=None,reject_by_annotation=False,**opts)
    records=[]
    trial_total=0
    for sid in subjects:
        X,label,meta=paradigm.get_data(dataset=dataset,subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        label=np.asarray(label)
        y=(label==positive).astype(int)
        if (X.ndim!=3 or X.shape[1]!=expected_channels or
            len(label)!=len(X) or set(label.tolist())!=set(events) or
            not np.isfinite(X).all() or min(np.bincount(y,minlength=2))<20):
            raise RuntimeError(f"fixed {name} EEG acquisition contract failed for {sid}")
        records.append({"subject":sid,"signal":X,"labels":y})
        trial_total+=len(y)
        print("V28_NEGATIVE_CONTROL_INPUT",name,sid,len(y),flush=True)
    audit=audit_exact_epoch_overlap(records)
    if audit["n_original_subject_ids"]!=len(subjects) or audit["n_total_trial_instances"]!=trial_total:
        raise AssertionError("dataset denominator drift")
    # Output is exclusively aggregate graph metadata, no EEG samples.
    return {key:value for key,value in audit.items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    if not PROTOCOL.exists():raise RuntimeError("precommitted protocol missing")
    result={
      "status":"V28_TWO_PREVIOUSLY_EXPOSED_ACQUISITIONS_EPOCH_INTEGRITY",
      "no_classifier_training_or_target_outcome_scoring":True,
      "global_recording_signal_inspection_not_strict_Z0":True,
      "not_new_independent_acquisition_validation":True,
      "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
      "datasets":{}
    }
    for name in ("Weibo2014","BNCI2014_002"):
        result["datasets"][name]=cohort(name)
    body=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(body)
    print(json.dumps({"status":result["status"],"summary":{
       n:{"n_subjects":v["n_original_subject_ids"],
          "n_epoch_instances":v["n_total_trial_instances"],
          "n_exact_unique_epochs":v["n_unique_exact_epoch_contents"],
          "n_cross_ID_exact_overlap_edges":v["n_subject_pairs_with_exact_overlap"],
          "n_connected_record_groups":v["n_exact_epoch_connected_groups"]}
       for n,v in result["datasets"].items()},
       "receipt_sha256":hashlib.sha256(body.encode()).hexdigest()},indent=2),flush=True)

if __name__=="__main__":main()
