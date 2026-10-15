#!/usr/bin/env python3
"""One additional previously untracked external-acquisition P3 control.

BNCI2015_004 is a MENTAL-TASK, not motor-imagery, dataset. Binary
'math' vs 'letter' is the native MOABB event pair. Public source IDs
1..8 and all five source-only selectors are fully fixed and JSON
SHA-256 committed before public target ID 9 is requested. Source
choices use matched preprocessing C3/Cz/C4 and grouped folds.
A novel experiment workflow is not an independently preregistered
study. One participant is NOT an independent population estimate.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from moabb.datasets import BNCI2015_004
from moabb.paradigms import MotorImagery
from p3_bnci004_four_operator_same_source import old,select,evaluate

SOURCE=tuple(range(1,9))
TARGET=9
EVENTS=("math","letter")
CHANNELS=("C3","Cz","C4")

def dump(path,record):
    b=(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b)
    return hashlib.sha256(b).hexdigest()

def load(paradigm,ds,ids):
    c=[];y=[];groups=[];log=[];sha=hashlib.sha256()
    for sid in ids:
        epochs,labels,_=paradigm.get_data(dataset=ds,subjects=[sid])
        epochs=np.asarray(epochs,dtype=np.float64)
        labels=np.asarray(labels)
        if (epochs.ndim!=3 or epochs.shape[1]!=3 or len(epochs)<25
            or not np.isfinite(epochs).all()
            or set(labels.tolist())!=set(EVENTS)):
            raise ValueError(f"Unexpected BNCI2015_004 math/letter schema for {sid} "
                             f"{epochs.shape} {set(labels.tolist())}")
        yy=np.asarray(labels==EVENTS[1],dtype=int)
        if not 0.1<yy.mean()<0.9: raise ValueError("unbalanced task")
        z=old.covariances(epochs)
        sha.update(np.ascontiguousarray(z).tobytes())
        c.append(z);y.append(yy);groups.extend([sid]*len(yy))
        log.append({"public_id":sid,"n_binary_trials":int(len(yy)),
                    "n_letter":int(yy.sum()),"eeg_channels":3})
        print("P3_BNCI2015_004_SOURCE_OR_HELD_LOADED",sid,len(yy),flush=True)
    return np.concatenate(c),np.concatenate(y),np.asarray(groups),log,sha.hexdigest()

def run(out):
    ds=BNCI2015_004()
    paradigm=MotorImagery(n_classes=2,events=list(EVENTS),
                   channels=list(CHANNELS),fmin=8,fmax=30,
                   resample=160,baseline=None,reject_by_annotation=False)
    C,y,g,receipts,source_digest=load(paradigm,ds,SOURCE)
    decision=select(C,y,g)
    lock={"status":"BNCI2015_004_8_SOURCE_PRE_HELD_ID9_FROZEN",
          "source_ids":list(SOURCE),"target_id9_not_yet_requested":True,
          "dataset":"BNCI2015_004","dataset_DOI":"10.1371/journal.pone.0123727",
          "paradigm":"mental tasks, NOT motor imagery",
          "events":list(EVENTS),"channels":list(CHANNELS),
          "processing":"8-30Hz, 160Hz, source-fitted tangent geometry",
          "source_covariance_SHA256":source_digest,
          "source_receipts":receipts,
          "source_chooses_all_five_policies":decision,
          "pipeline_commit":os.environ.get("GITHUB_SHA","LOCAL_NOT_GITHUB"),
          "same_common_operator_matrix":True,
          "no_target_labels_or_features_for_choices":True,
          "prior_repository_search_had_no_BNCI2015_004":True,
          "not_externally_preregistered":True,
          "source_persons_are_NOT_eight_independent_datasets":True}
    locksha=dump(out/"P3_BNCI2015_004_SOURCE_BEFORE_ID9.json",lock)
    print("P3_BNCI2015_004_FROZEN_CHOICES_BEFORE_TARGET9_SHA",locksha,flush=True)
    Ct,yt,_,target_receipt,targetsha=load(paradigm,ds,[TARGET])
    vals=evaluate(C,y,g,Ct,yt,decision)
    deltas={key:float(v-vals["source_macro"]) for key,v in vals.items()}
    result={"status":"FIRST_REPO_EXTERNAL_MENTAL_TASK_SOURCE_ONLY_FIVE_POLICY_TARGET",
            "source_lock_sha":locksha,
            "source_ids":list(SOURCE),"target_id":TARGET,
            "target_receipt":target_receipt,
            "target_feature_sha256":targetsha,
            "auc":vals,"paired_delta_vs_source_macro":deltas,
            "source_decision_is_identical_for_any_future_held_subject":True,
            "one_held_person_in_one_acquisition":True,
            "not_proof_of_EEG_motor_imagery_generalisation":True,
            "not_externally_independently_preregistered":True,
            "no_prospective_population_p_or_CI":True,
            "no_target_specific_source_policy_tuning":True}
    digest=dump(out/"P3_BNCI2015_004_FIRST_REAL_HELD_TARGET_RESULT.json",result)
    print("P3_BNCI2015_004_FIRST_NEW_REAL_TARGET_RESULT",
          json.dumps(result,sort_keys=True),flush=True)
    print("P3_BNCI2015_004_RESULT_SHA256",digest,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();run(args.out)
