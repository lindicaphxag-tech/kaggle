#!/usr/bin/env python3
"""P1 corrected proper predictive NLL fair comparison BNCI2015_001.

NEW computation on an ALREADY EXPOSED original 12-person acquisition:
pure retrospective; never classify as prospective confirmation.
All shrinkage choices are source-participant-group CV only.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2015_001
from moabb.paradigms import MotorImagery

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"defaultreg_proper_predictive"))
import original_frozen_runner as old

METHODS=("source_macro","true_source_train_mean_predictive","historical_val_class_centred","LedoitWolf","OAS")
GRID=old.FULL_GRID

def dump(obj,p):
    body=(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    return hashlib.sha256(body).hexdigest()

def load():
    ds=BNCI2015_001()
    pm=MotorImagery(n_classes=2,events=["right_hand","feet"],
                    fmin=8,fmax=30,resample=160,baseline=None,
                    reject_by_annotation=False)
    c=[];ys=[];gs=[];rec=[]
    for sid in range(1,13):
        X,lab,_=pm.get_data(dataset=ds,subjects=[sid])
        X=np.asarray(X,dtype=np.float64);lab=np.asarray(lab)
        y=(lab=="feet").astype(int)
        if X.ndim!=3 or X.shape[1]!=13 or set(lab.tolist())!={"right_hand","feet"} or not np.isfinite(X).all():
            raise RuntimeError("BNCI2015_001 fixed public EEG drift")
        c.append(old.covariances(X));ys.append(y);gs.extend([sid]*len(y))
        rec.append({"participant":sid,"trials":int(len(y)),"n_feet":int(y.sum()),"channels":13})
        print("P1_CORRECT_NLL_RAW_COHORT",sid,len(y),flush=True)
    return np.concatenate(c),np.concatenate(ys),np.asarray(gs),rec

def choose(c,y,g):
    ids=sorted(set(map(int,g)));folds=old._held_groups(ids)
    store={float(l):{"auc":[],"proper":[],"historical":[]} for l in GRID}
    viable=set(map(float,GRID))
    for held in folds:
        v=np.isin(g,held);t=~v
        ref=old.reference(c[t])
        Xtr=old.tangent(c[t],ref);Xv=old.tangent(c[v],ref)
        st=old.spectral_state(Xtr,y[t])
        _,_,difference,mid,_=st
        source_means=np.stack([mid-difference/2,mid+difference/2])
        for sid in held:
            m=(g[v]==sid)
            xx=Xv[m];yy=y[v][m]
            if set(yy.tolist())!={0,1}:raise ValueError("missing class on held source person")
            mu=np.stack([xx[yy==i].mean(0) for i in (0,1)])
            Rproper=xx-source_means[yy]
            Rhistorical=xx-mu[yy]
            for lam in GRID:
                lam=float(lam)
                if lam not in viable:continue
                try:
                    s=old.scalar_scores(st,xx,lam)
                    au=float(roc_auc_score(yy,s))
                    pr=old.predictive_nll(st,Rproper,lam)
                    hi=old.predictive_nll(st,Rhistorical,lam)
                except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                    viable.discard(lam)
                    continue
                store[lam]["auc"].append(au)
                store[lam]["proper"].append(pr)
                store[lam]["historical"].append(hi)
    viable=sorted(l for l in viable if all(len(store[l][k])==len(ids) for k in ("auc","proper","historical")))
    if len(viable)<2:raise RuntimeError("no common source-lambda support")
    def mean(l,k):return float(np.mean(store[l][k]))
    macro=max(viable,key=lambda l:(mean(l,"auc"),l))
    proper=min(viable,key=lambda l:(mean(l,"proper"),l))
    historical=min(viable,key=lambda l:(mean(l,"historical"),l))
    predicted=float(mean(macro,"auc")-mean(proper,"auc"))
    return {"source_macro":macro,"true_source_train_mean_predictive":proper,
            "source_predicted_macro_minus_proper_auc":predicted,
            "source_predicted_direction":("macro" if predicted>0 else "proper" if predicted<0 else "tie"),
            "historical_val_class_centred":historical,
            "n_source_participants":len(ids),"feasible_grid_count":len(viable),
            "proper_not_target_val_class_centred":True,
            "target_held_features_labels_unused_in_inner_selection":True}

def one(c,y,g,target,policy):
    tr=g!=target;v=~tr
    ref=old.reference(c[tr]);Xtr=old.tangent(c[tr],ref);Xv=old.tangent(c[v],ref)
    st=old.spectral_state(Xtr,y[tr])
    pred={k:old.scalar_scores(st,Xv,float(policy[k])) for k in (
          "source_macro","true_source_train_mean_predictive","historical_val_class_centred")}
    for k in ("LedoitWolf","OAS"):
        pred[k]=old.analytic_scores(Xtr,y[tr],Xv,"LW" if k=="LedoitWolf" else "OAS")[0]
    return {k:float(roc_auc_score(y[v],s)) for k,s in pred.items()}

def main():
    pa=argparse.ArgumentParser();pa.add_argument("--out",type=Path,required=True)
    args=pa.parse_args();out=args.out
    c,y,g,rec=load()
    selected=[]
    for target in range(1,13):
        m=(g!=target)
        p=choose(c[m],y[m],g[m])
        selected.append({"held_participant":target,"source_only_selectors":p})
        print("P1_SOURCE_ONLY_CORRECTED_SELECTORS",target,p,flush=True)
    lock=dump({"dataset":"BNCI2015_001","participants":12,"moabb":"1.7.2",
               "processing":"8-30Hz/160Hz/13 channels","grid":list(GRID),
               "previously_exposed_target_data":True,
               "source_only_fit_per_fold":True,
               "not_new_prospective_independent_test":True,
               "participant_receipts":rec,"per_target_source_choices":selected},
              out/"BNCI2015_001_SOURCE_CHOICES_BEFORE_NEW_TARGET_AUC.json")
    rows=[]
    for x in selected:
        t=x["held_participant"];metrics=one(c,y,g,t,x["source_only_selectors"])
        predicted=x["source_only_selectors"]["source_predicted_macro_minus_proper_auc"]
        observed=metrics["source_macro"]-metrics["true_source_train_mean_predictive"]
        rows.append({"held_participant":t,"auc":metrics,
                     "source_predicted_macro_minus_proper_auc":predicted,
                     "held_observed_macro_minus_proper_auc":observed,
                     "direction_correct_excluding_ties":bool(predicted*observed>0)})
        print("P1_BNCI2015_CORRECT_PROPER_HELD",t,metrics,flush=True)
    summary={k:float(np.mean([r["auc"][k] for r in rows])) for k in METHODS}
    for alternative in ("historical_val_class_centred","true_source_train_mean_predictive","LedoitWolf","OAS"):
        x=np.array([r["auc"]["source_macro"]-r["auc"][alternative] for r in rows])
        summary["macro_minus_"+alternative+"_paired_mean"]=float(x.mean())
        summary["macro_vs_"+alternative+"_harmed_gt002"]=int(np.sum(x<-.0200000001))
        summary["macro_vs_"+alternative+"_helped_gt002"]=int(np.sum(x>.0200000001))
    # Post-outcome descriptive/exploratory exact paired sign-flip
    # versus corrected proper source-train mean predictive comparator.
    paired=np.asarray([r["auc"]["source_macro"]-
                       r["auc"]["true_source_train_mean_predictive"] for r in rows])
    observed=float(paired.mean())
    n=len(paired)
    signs=1-2*((np.arange(1<<n)[:,None] >> np.arange(n)[None,:])&1)
    exact=float(np.mean(np.abs((signs*paired).mean(axis=1))>=abs(observed)-1e-12))
    rng=np.random.default_rng(20261013)
    draw=rng.integers(0,n,size=(20000,n))
    percentile=np.quantile(paired[draw].mean(axis=1),[.025,.975])
    summary["retrospective_macro_minus_proper_exact_signflip_two_sided_p"]=exact
    summary["retrospective_macro_minus_proper_bootstrap_95"]=[float(x) for x in percentile]
    summary["paired_nonzero"]=int(np.sum(np.abs(paired)>1e-12))
    predictions=np.asarray([r["source_predicted_macro_minus_proper_auc"] for r in rows])
    non_ties=(np.abs(predictions)>1e-12)&(np.abs(paired)>1e-12)
    summary["source_predicted_direction_evaluable"]=int(non_ties.sum())
    summary["source_predicted_direction_correct"]=int(np.sum(predictions[non_ties]*paired[non_ties]>0))
    summary["source_predicted_direction_accuracy"]=float(np.mean(predictions[non_ties]*paired[non_ties]>0)) if non_ties.any() else None
    summary["source_prediction_mean_absolute_error_auc"]=float(np.mean(np.abs(predictions-paired)))
    summary["prediction_evaluation_exploratory_previously_exposed_dataset"]=True
    summary["inference_not_original_prospective"]=True
    print("P1_BNCI2015_CORRECT_PROPER_EXPLORATORY_STATISTICS",
          json.dumps({"mean":observed,"exact_two_sided_p":exact,
                      "descriptive_bootstrap_95":percentile.tolist()},sort_keys=True),flush=True)
    outdata={"status":"RETROSPECTIVE_CORRECT_SOURCE_NLL_BNCI2015_001",
       "source_lock_sha":lock,"n_subjects":12,
       "new_model_family_improvement_claimed":False,
       "previously_exposed_target_dataset":True,
       "no_new_confirmatory_p_values":True,
       "no_independent_replicate_of_JNE_primary":True,
       "source_choices_frozen_before_new_outcome_arithmetic":True,
       "summary":summary,"per_subject":rows}
    sha=dump(outdata,out/"P1_BNCI2015_001_PROPER_CORRECT_COMPARISON_REAL.json")
    print("P1_REAL_BNCI2015_PROPER_CORRECT_COMPARISON_FINISHED",json.dumps(summary,sort_keys=True),flush=True)
    print("P1_RESULT_SHA",sha)
if __name__=="__main__":main()
