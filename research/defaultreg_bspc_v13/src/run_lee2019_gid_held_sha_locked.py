#!/usr/bin/env python3
"""Lee2019 dual held blocks. Source SHA must reproduce BEFORE target IO.

NO model or data-adaptive processing is performed on held trial features,
except passing them through the source-fitted preprocessing/classifiers.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from pyriemann.classification import MDM
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"research"/"defaultreg_proper_predictive"))
import original_frozen_runner as old
from source_group_invariant_lda import fit_group_state,predict
import run_lee2019_gid_source_lock as lock

def person(para,ds,sid):
    X,labels,_=para.get_data(dataset=ds,subjects=[sid])
    labels=np.asarray(labels)
    y=(labels==lock.EVENTS[1]).astype(int)
    if (X.ndim!=3 or X.shape[1]!=3
        or set(labels.tolist())!=set(lock.EVENTS)
        or not np.isfinite(X).all() or
        min(np.bincount(y,minlength=2))<20
        or len(X)<80 or X.shape[-1]<150):
        raise RuntimeError("LEE2019_FROZEN_TARGET_SCHEMA_"+str(sid))
    C=old.covariances(X)
    return X,C,y,{"id":int(sid),"n":int(len(y)),
                 "covariance_sha256":lock.sha(C.tobytes())}

def load_sources():
    ds=lock.Lee2019_MI(train_run=True,test_run=False,
                       resting_state=False,sessions=[1])
    para=old.MotorImagery(n_classes=2,events=list(lock.EVENTS),
         channels=list(lock.CHANNELS),fmin=8,fmax=30,
         resample=160,baseline=None,reject_by_annotation=False)
    xs=[];cs=[];ys=[];gs=[];rec=[]
    for sid in lock.SRC:
        X,C,y,r=person(para,ds,sid)
        xs.append(X);cs.append(C);ys.append(y);gs.extend([sid]*len(y))
        rec.append({"id":sid,"trials":len(y),
                    "class0":int((y==0).sum()),"class1":int((y==1).sum()),
                    "covariance_sha256":lock.sha(C.tobytes()),
                    "epoch_shape":list(X.shape)})
        print("LEE2019_HELD_RUN_SOURCE_ONLY_READ",sid,flush=True)
    return para,ds,np.concatenate(xs),np.concatenate(cs),np.concatenate(ys),np.array(gs),rec

def source_models(X,C,y,g):
    inv=old.reference(C);Xt=old.tangent(C,inv)
    gid=fit_group_state(Xt,y,g)
    classical=old.spectral_state(Xt,y)
    LR=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000).fit(Xt,y)
    mdm=MDM(metric="riemann").fit(C,y)
    if list(mdm.classes_)!=[0,1]:raise ValueError("MDM classes drift")
    from mne.decoding import CSP
    csp=CSP(n_components=3,reg=.1,log=True,
            norm_trace=False,transform_into="average_power")
    transformed=csp.fit_transform(X,y)
    LRcsp=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000).fit(transformed,y)
    return (inv,Xt,gid,classical,LR,mdm,csp,LRcsp)

def held_scores(model,X,C,y_source,choices):
    inv,Xtrain,gid,classical,LR,mdm,csp,LRcsp=model
    Xtan=old.tangent(C,inv)
    result={}
    for mode in ("original","gid","within_only","balanced_only"):
        result[mode]=predict(gid,Xtan,choices["decisions"][mode]["lambda"],mode)
    reference=old.scalar_scores(classical,Xtan,choices["decisions"]["original"]["lambda"])
    if not np.allclose(result["original"],reference,atol=1e-9,rtol=1e-9):
        raise AssertionError("original scalar LDA not reproduced on target")
    for kind in ("LW","OAS"):
        s,_=old.analytic_scores(Xtrain,y_source,Xtan,kind)
        result[kind]=s
    result["fixed_full_shrink"]=old.scalar_scores(classical,Xtan,1.)
    result["tangent_LogReg"]=LR.decision_function(Xtan)
    distances=mdm.transform(C)
    result["riemannian_MDM"]=distances[:,0]-distances[:,1]
    result["source_CSP3_LogReg"]=LRcsp.decision_function(csp.transform(X))
    return result

def signflip(x):
    x=np.asarray(x,dtype=float)
    if len(x)!=12:raise AssertionError("required 12 independent group IDs")
    bits=np.arange(4096,dtype=np.uint32)
    signs=1-2*((bits[:,None]>>np.arange(12,dtype=np.uint32))&1).astype(np.int8)
    return float(np.mean(np.abs(signs@x)>=abs(float(x.sum()))-1e-12))

def result_block(label,persons,ids):
    assert tuple(z["person_id"] for z in persons)==ids
    methods=list(persons[0]["auc"])
    assert all(set(z["auc"])==set(methods) for z in persons)
    means={mode:float(np.mean([z["auc"][mode] for z in persons])) for mode in methods}
    d=np.array([q["auc"]["gid"]-q["auc"]["original"] for q in persons])
    bootstrap=d[np.random.default_rng(20261013).integers(0,12,(20000,12))].mean(axis=1)
    return {
      "block":label,"n_persons":12,"n_trials":sum(q["trials"] for q in persons),
      "macro_auc_per_method":means,
      "gid_minus_original_mean_delta":float(d.mean()),
      "gid_minus_original_exact_two_sided_signflip_p":signflip(d),
      "gid_minus_original_descriptive_bootstrap95":np.quantile(bootstrap,[.025,.975]).tolist(),
      "gain_positive_persons":int((d>1e-12).sum()),
      "gain_negative_persons":int((d< -1e-12).sum()),
      "gid_losses_gt_002":int((d< -.020000000001).sum()),
      "all_person_results":persons
    }

def main(locked_path,locked_sha,out,block_to_run):
    if not lock.PROTO.exists():raise RuntimeError("original protocol missing")
    original=locked_path.read_bytes()
    if lock.sha(original)!=locked_sha:raise AssertionError("ORIGINAL_SHA_MISMATCH_STOP")
    source_original=json.loads(original)
    para,ds,X,C,y,g,rec=load_sources()
    source_recomputed=lock.frozen_choices(C,y,g,rec)
    out.mkdir(parents=True,exist_ok=True)
    rehash=lock.write(source_recomputed,out/lock.FILE)
    if rehash!=locked_sha or source_recomputed!=source_original:
        raise AssertionError("SOURCE_CHOICE_SHA_NOT_IDENTICAL_STOP_BEFORE_TARGET")
    print("LEE2019_PRE_HELD_SOURCE_SHA_VERIFIED",locked_sha,flush=True)
    mdl=source_models(X,C,y,g)
    summary=[]
    for block,ids in [("A",lock.TARGET_A),("B",lock.TARGET_B)]:
        if block!=block_to_run:continue
        persons=[]
        for sid in ids:
            Xv,Cv,yv,receipt=person(para,ds,sid)
            scores=held_scores(mdl,Xv,Cv,y,source_original["chosen_source_only"])
            auc={k:float(roc_auc_score(yv,score)) for k,score in scores.items()}
            persons.append({"person_id":int(sid),"trials":len(yv),
                            "auc":auc,"receipt":receipt,
                            "gid_minus_original":auc["gid"]-auc["original"]})
            print("LEE2019_HELD_TARGET_RECORDED",sid,len(yv),flush=True)
        data=result_block(block,persons,ids)
        summary.append(data)
        lock.write({"original_source_sha256":locked_sha,**data},
                   out/("LEE2019_HELD_"+block+"_COMPLETE.json"))
        print("LEE2019_BLOCK_COMPLETE",json.dumps({
          "block":block,"mean_delta":data["gid_minus_original_mean_delta"],
          "exact_p":data["gid_minus_original_exact_two_sided_signflip_p"],
          "macro_auc":data["macro_auc_per_method"],
          "n_loss_gt002":data["gid_losses_gt_002"]},sort_keys=True),flush=True)
    b=summary[0]
    promoted_block=bool(source_original["positive_transfer_pre_prediction"]
       and source_original["final_operator_cosine"]<1-1e-5
       and b["gid_minus_original_mean_delta"]>=.01
       and b["gid_minus_original_exact_two_sided_signflip_p"]<.05
       and b["gid_losses_gt_002"]==0)
    output={
      "status":"ONE_HELD_BLOCK_COMPLETED_WITHOUT_RETUNE",
      "block":block_to_run,
      "original_source_sha256":locked_sha,
      "source_only_forecast_positive":source_original["positive_transfer_pre_prediction"],
      "nonparallel_original_vs_gid_cosine":source_original["final_operator_cosine"],
      "n_source_people":12,"n_held_people":12,
      "fixed_source_model_no_target_retrain":True,
      "block_result":b,"block_gate_passed":promoted_block,
      "two_block_overall_promotion_NOT_YET_ESTABLISHED":True,
      "not_independent_acquisition_replicated":True
    }
    lock.write(output,out/("LEE2019_BLOCK_"+block_to_run+"_HELD_RESULTS.json"))
    print("LEE2019_LOCKED_HELD_BLOCK_FINAL",json.dumps({
      "block":block_to_run,"source_sha":locked_sha,
      "block_gate_passed":promoted_block,
      "mean_delta":b["gid_minus_original_mean_delta"],
      "exact_p":b["gid_minus_original_exact_two_sided_signflip_p"]
    },sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original-manifest",type=Path,required=True)
    p.add_argument("--original-sha",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--block",choices=["A","B"],required=True)
    a=p.parse_args()
    main(a.original_manifest,a.original_sha,a.out,a.block)
