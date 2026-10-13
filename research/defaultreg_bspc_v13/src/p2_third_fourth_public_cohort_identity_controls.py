#!/usr/bin/env python3
"""New publication-reliability negative controls: exact and aligned near matches.

No classifier fitting, patient-identity inference or diagnostic false-
positive-rate estimation. Two PREVIOUSLY EXPOSED public EEG acquisitions
are checked using transparent morphology and label-specific hashing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2014_001, BNCI2015_001
from moabb.paradigms import MotorImagery

COHORTS={
  "BNCI2014_001":{
    "cls":BNCI2014_001,"ids":range(1,10),
    "events":["left_hand","right_hand"],"positive":"right_hand",
    "channels":["C3","Cz","C4"],"n_expected_channels":3
  },
  "BNCI2015_001":{
    "cls":BNCI2015_001,"ids":range(1,13),
    "events":["right_hand","feet"],"positive":"feet",
    "channels":None,"n_expected_channels":13
  }
}
def epoch_digest(x,y):
    x=np.ascontiguousarray(np.asarray(x,dtype="<f8"))
    h=hashlib.sha256()
    h.update(bytes([int(y)]))
    h.update(np.asarray(x.shape,dtype="<i8").tobytes())
    h.update(x.tobytes())
    return h.hexdigest()

def one(name,cfg):
    kwargs={
        "n_classes":2,"events":cfg["events"],
        "fmin":8,"fmax":30,"resample":160,
        "baseline":None,"reject_by_annotation":False
    }
    if cfg["channels"] is not None:kwargs["channels"]=cfg["channels"]
    paradigm=MotorImagery(**kwargs)
    ds=cfg["cls"]()
    history={}
    entries=[]
    total_trials=0
    exact_overlap=set()
    for sid in cfg["ids"]:
        X,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64)
        lab=np.asarray(labels)
        y=(lab==cfg["positive"]).astype(np.uint8)
        if X.ndim!=3 or X.shape[1]!=cfg["n_expected_channels"] or not np.isfinite(X).all():
            raise RuntimeError(f"Unexpected frozen public EEG structure: {name} ID {sid} {X.shape}")
        if set(lab.tolist())!=set(cfg["events"]):raise RuntimeError("public event taxonomy drift")
        total_trials+=len(y)
        for ix in range(len(y)):
            key=epoch_digest(X[ix],y[ix])
            for previous in history.get(key,()):
                if previous!=sid:exact_overlap.add(tuple(sorted((previous,sid))))
            history.setdefault(key,set()).add(sid)
        full=X.reshape(-1)
        centered=full-full.mean()
        norm=float(np.linalg.norm(centered))
        if not np.isfinite(norm) or norm<=0:raise ValueError("invalid full EEG center/norm")
        entries.append({"sid":sid,"shape":list(X.shape),"n":len(y),
                        "vector":centered/norm,
                        "label_hash":hashlib.sha256(y.tobytes()).hexdigest()})
        print("P2_ADDITIONAL_COHORT_PUBLIC_EEG_SCANNED",name,sid,len(y),flush=True)
    near=[]
    incompatible=0;scanned=0;largest=None
    for i,a in enumerate(entries):
        for b in entries[i+1:]:
            if a["shape"]!=b["shape"]:
                incompatible+=1
                continue
            scanned+=1
            rho=float(np.clip(np.dot(a["vector"],b["vector"]),-1.,1.))
            cand={"ids":[a["sid"],b["sid"]],
                  "aligned_full_record_pearson":rho,
                  "same_trial_label_hash":a["label_hash"]==b["label_hash"]}
            if largest is None or rho>largest["aligned_full_record_pearson"]:
                largest=cand
            if rho>=.95:near.append(cand)
    n=len(entries)
    result={
        "public_acquisition":name,"released_ids":n,
        "trial_instances":total_trials,"different_epoch_digest_count":len(history),
        "cross_id_exact_epoch_overlap_edges":[list(v) for v in sorted(exact_overlap)],
        "n_cross_id_exact_epoch_overlap_edges":len(exact_overlap),
        "pairs_total":n*(n-1)//2,"pairs_same_shape_scanned":scanned,
        "pairs_unscannable_shape_mismatch":incompatible,
        "pairs_aligned_full_record_corr_ge_0p95":len(near),
        "pairs_aligned_full_record_corr_ge_0p9999":sum(v["aligned_full_record_pearson"]>=.9999 for v in near),
        "high_similarity_pairs":sorted(near,key=lambda z:-z["aligned_full_record_pearson"]),
        "max_compatible_pair":largest,
        "label_and_signal_included_in_exact_hash":True,
        "cannot_discover_reordered_time_shifted_or_remontaged_duplicates":True,
        "no_classification_results_or_patient_identity_read":True
    }
    if scanned+incompatible!=result["pairs_total"]:raise AssertionError("bad pair accounting")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    data={
        "status":"EXECUTED_ADDITIONAL_PUBLIC_EEG_IDENTITY_CONTROLS",
        "controls_were_seen_public_datasets_not_external_prospective_holdouts":True,
        "preprocessing":"MOABB 1.7.2 / 8-30 Hz / 160 Hz",
        "is_not_prospective_Z0_or_population_alias_rate_estimate":True,
        "protocol_checks":"Epoch SHA includes trial array shape, float64 samples, event class; near screen uses aligned whole record correlation",
        "results":{name:one(name,v) for name,v in COHORTS.items()}
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    body=json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.write_text(body,encoding="utf-8")
    print("P2_EXTERNAL_PUBLIC_IDENTITY_CONTROLS",json.dumps({
        k:{x:v[x] for x in ("pairs_total","n_cross_id_exact_epoch_overlap_edges",
                            "pairs_aligned_full_record_corr_ge_0p95","pairs_same_shape_scanned")}
        for k,v in data["results"].items()},sort_keys=True),flush=True)
    print("P2_EXTERNAL_PUBLIC_IDENTITY_SHA256",hashlib.sha256(body.encode()).hexdigest())

if __name__=="__main__":main()
