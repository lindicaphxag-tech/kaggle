#!/usr/bin/env python3
"""P3 first-repo Zhou2016 held acquisition four-operator source-only study.

Source public IDs 1,2,3 then lock BEFORE requesting held public ID 4.
All 5 policy choices frozen jointly, exact same 3-channel source-fitted
tangent geometry and target signal/labels. A new repo-targeted dataset
is not an externally preregistered experiment; only one held participant
belongs to this independent named acquisition. No universal null claim.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from moabb.datasets import Zhou2016
from moabb.paradigms import MotorImagery
from p3_bnci004_four_operator_same_source import old,select,evaluate

SOURCE=(1,2,3)
TARGET=4

def save(path,obj):
    s=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(s)
    return hashlib.sha256(s).hexdigest()

def acquire(p,ds,ids):
    cs=[];ys=[];groups=[];receipts=[];sha=hashlib.sha256()
    for sid in ids:
        X,y,metadata=p.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);y=np.asarray(y)
        if (X.ndim!=3 or X.shape[1]!=3 or len(X)<50
                or set(y.tolist())!={"left_hand","right_hand"} or
                not np.isfinite(X).all()):
            raise RuntimeError(f"Zhou2016 ID {sid} unexpected shape/labels")
        binary=np.asarray(y=="right_hand",dtype=int)
        if not 0.15 < binary.mean() < 0.85:
            raise RuntimeError("bad class balance")
        cv=old.covariances(X)
        sha.update(np.ascontiguousarray(cv).tobytes())
        cs.append(cv);ys.append(binary);groups.extend([sid]*len(binary))
        receipts.append({"public_id":sid,"n":len(binary),
            "n_positive":int(binary.sum()),"n_channels":3})
        print("P3_ZHOU_PUBLIC_ID_IMPORTED",sid,len(binary),flush=True)
    return np.concatenate(cs),np.concatenate(ys),np.asarray(groups),receipts,sha.hexdigest()

def run(out):
    ds=Zhou2016()
    p=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                    channels=["C3","Cz","C4"],fmin=8,fmax=30,
                    resample=160,baseline=None,reject_by_annotation=False)
    Cs,ys,gs,receipts,features_sha=acquire(p,ds,SOURCE)
    decisions=select(Cs,ys,gs)
    lock_data={"status":"ZHOUMI_PRE_HELD_ID4_SOURCE_CHOICE_LOCK",
      "dataset":"Zhou2016","dataset_DOI":"10.1371/journal.pone.0162657",
      "source_ids":list(SOURCE),"held_id_UNREAD":TARGET,
      "source_receipts":receipts,"source_covariance_feature_SHA256":features_sha,
      "pipeline_commit":os.environ.get("GITHUB_SHA","LOCAL_NON_REGISTRY"),
      "source_choices":decisions,
      "algorithms":["source_macro","continuous","SCE","anisotropic","group_centered"],
      "hyperparameters":{"SCE_radius":1,"anisotropic_alpha_grid":[0,.5,2],
        "anisotropic_beta":.25,"continuous_gap_tolerance":.02},
      "processing":"C3 Cz C4, left vs right, 8-30Hz, 160Hz, all source sessions",
      "target_ID4_not_read_until_after_source_SHA":True,
      "known_external_preregistration":False,
      "one_named_acquisition_not_multiple_independent_acquisitions":True,
      "no_target_EEG_or_labels_for_source_decisions":True}
    sha=save(out/"P3_ZHOU_FROZEN_BEFORE_PUBLIC_HELD4.json",lock_data)
    print("P3_ZHOU_HELD_ID4_SOURCE_LOCK_SHA256",sha,flush=True)
    Ct,yt,gt,t_receipt,targetfeatures_sha=acquire(p,ds,[TARGET])
    auc=evaluate(Cs,ys,gs,Ct,yt,decisions)
    deltas={k:float(v-auc["source_macro"]) for k,v in auc.items()}
    report={"status":"NEW_REPO_FIRST_ZHOU2016_FOUR_OPERATOR_COMPARISON",
        "source_lock_SHA256":sha,"source_IDs":list(SOURCE),"held_ID":TARGET,
        "held_receipt":t_receipt,"target_covariance_input_SHA256":targetfeatures_sha,
        "methods_auc":auc,"paired_delta_AUC_vs_macro":deltas,
        "material_harm_threshold_AUC":-.02,
        "at_least_one_material_harm":any(x<-.02 for x in deltas.values()),
        "max_per_held_target_gain_AUC":max(deltas.values()),
        "not_externally_preregistered":True,
        "not_confirmatory_acquisition_level_meta_analysis":True,
        "not_independent_of_P1_shared_general_method_family":True,
        "one_held_person_one_acquisition":True}
    result_sha=save(out/"P3_ZHOU_FIRST_HELD_REAL_EEG_RESULTS.json",report)
    print("P3_ZHOU_PRE_LOCKED_HELD_REAL_RESULT",
          json.dumps(report,sort_keys=True),flush=True)
    print("P3_ZHOU_RESULT_SHA256",result_sha,flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.out)
