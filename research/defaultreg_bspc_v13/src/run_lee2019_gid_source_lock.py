#!/usr/bin/env python3
"""Lee2019_MI source-only freeze. Never reads target IDs 13+.

Inputs: fixed first session, labelled offline train run, 3 channels.
Outputs SHA256 of immutable protocol/choices before any target EEG IO.
Held stage is separate and will require this exact SHA.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from moabb.datasets import Lee2019_MI
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from source_group_invariant_lda import fit_group_state,vector,source_group_cv

SRC=tuple(range(1,13))
TARGET_A=tuple(range(13,25))
TARGET_B=tuple(range(25,37))
CHANNELS=("C3","Cz","C4")
EVENTS=("left_hand","right_hand")
PROTO=ROOT/"research"/"defaultreg_bspc_v13"/"LEE2019_GID_FROZEN_FIRST_EXPERIMENT_PROTOCOL.md"
FILE="LEE2019_S1_SOURCE12_IMMUTABLE_CHOICES.json"

def sha(x):return hashlib.sha256(x).hexdigest()
def write(obj,path):
    b=(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b)
    return sha(b)

def source_data():
    ds=Lee2019_MI(train_run=True,test_run=False,
                  resting_state=False,sessions=[1])
    para=old.MotorImagery(n_classes=2,events=list(EVENTS),
          channels=list(CHANNELS),fmin=8,fmax=30,
          resample=160,baseline=None,reject_by_annotation=False)
    cov=[];y=[];g=[];rec=[]
    for sid in SRC:
        X,labels,_=para.get_data(dataset=ds,subjects=[sid])
        labels=np.asarray(labels)
        yy=(labels==EVENTS[1]).astype(int)
        if (X.ndim!=3 or X.shape[1]!=3 or set(labels.tolist())!=set(EVENTS)
            or not np.isfinite(X).all() or
            min(np.bincount(yy,minlength=2))<20 or
            X.shape[0]<80 or X.shape[-1]<150):
            raise ValueError("LEE2019_FROZEN_SOURCE_SCHEMA_FAILED_"+str(sid))
        C=old.covariances(X)
        cov.append(C);y.append(yy);g.extend([sid]*len(yy))
        rec.append({"id":sid,"trials":len(yy),
                    "class0":int((yy==0).sum()),"class1":int((yy==1).sum()),
                    "covariance_sha256":sha(C.tobytes()),
                    "epoch_shape":list(X.shape)})
        print("LEE2019_SOURCE_ONLY_READ",sid,len(yy),flush=True)
    return np.concatenate(cov),np.concatenate(y),np.asarray(g),rec

def frozen_choices(C,y,g,rec):
    original=old.source_path(C,y,g)
    path=source_group_cv(C,y,g,original["feasible_full_lambdas"])
    a=path["decisions"]
    if (a["original"]["lambda"]!=original["lambda_source_macro"] or
        abs(a["original"]["source_macro_auc"]-
        original["criteria_by_lambda"][str(a["original"]["lambda"])]["source_macro_auc"])>1e-12):
        raise AssertionError("ORIGINAL_POOLED_LDA_PARITY_BROKEN")
    inv=old.reference(C)
    state=fit_group_state(old.tangent(C,inv),y,g)
    w=vector(state,a["original"]["lambda"],"original")
    v=vector(state,a["gid"]["lambda"],"gid")
    cosine=float(w@v/(np.linalg.norm(w)*np.linalg.norm(v)))
    import moabb,sklearn
    return {"source_protocol_sha256":sha(PROTO.read_bytes()),
       "source_ids":list(SRC),"target_A_UNREAD":list(TARGET_A),
       "target_B_UNREAD":list(TARGET_B),
       "session":1,"train_run":True,"test_run":False,
       "channels":list(CHANNELS),"class_labels":list(EVENTS),
       "source_trial_receipts":rec,"n_source_trials":int(len(y)),
       "fixed_original_grid":list(map(float,old.FULL_GRID)),
       "chosen_source_only":path,"final_operator_cosine":cosine,
       "nuisance_cov_between_fraction":state.trace_between_fraction,
       "positive_transfer_pre_prediction":
           a["gid"]["source_only_forecast_positive"],
       "original_lambdas_exactly_reproduced":True,
       "moabb_version":str(moabb.__version__),
       "sklearn_version":str(sklearn.__version__),
       "numpy_version":str(np.__version__),
       "source_choices_created_before_any_held_target_IO":True,
       "not_independently_preregistered":True}

def main(out):
    if not PROTO.exists():raise RuntimeError("missing frozen Lee protocol")
    C,y,g,rec=source_data()
    x=frozen_choices(C,y,g,rec)
    digest=write(x,out/FILE)
    print("LEE2019_SOURCE_MANIFEST_SHA_BEFORE_TARGET",digest,flush=True)
    print("LEE2019_SOURCE_ONLY_LOCKED_FORECAST",json.dumps({
      "digest":digest,
      "predicted_positive":x["positive_transfer_pre_prediction"],
      "nonparallel_cosine":x["final_operator_cosine"],
      "between_cov_fraction":x["nuisance_cov_between_fraction"],
      "original_lambda":x["chosen_source_only"]["decisions"]["original"]["lambda"],
      "gid_lambda":x["chosen_source_only"]["decisions"]["gid"]["lambda"],
      "source_delta":x["chosen_source_only"]["decisions"]["gid"]["source_paired_mean_delta"],
    },sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    main(p.parse_args().out)
