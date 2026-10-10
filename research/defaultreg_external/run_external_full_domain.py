"""Untouched Cho2017 / Lee2019_MI full-domain Default Regularisation replication.

Frozen design:
- all available subjects, outer LOPO;
- exact common 21 sensorimotor channels;
- 8-30 Hz, dataset task window, resampled to 160 Hz;
- Z0: no outer-target labels/features/moments choose geometry, lambda, or model;
- deterministic five-fold participant-grouped source selection;
- full [0,1] scalar shrinkage path;
- comparators: LedoitWolf, OAS, source-discriminability, source-cov-NLL,
  tangent logistic regression, Riemannian MDM.

Target labels are used only after all target score vectors are frozen.
"""
from __future__ import annotations
import argparse,json,math,platform,sys,time
from pathlib import Path
import numpy as np
import moabb,sklearn,pyriemann
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from moabb.datasets import Cho2017,Lee2019_MI
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM

COMMON21=("FC5","FC3","FC1","FCz","FC2","FC4","FC6",
          "C5","C3","C1","Cz","C2","C4","C6",
          "CP5","CP3","CP1","CPz","CP2","CP4","CP6")
FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)
SEED=20261010

def build_dataset(name):
    if name=="Cho2017":
        return Cho2017(),tuple(range(1,53)),3.0
    if name=="Lee2019_MI":
        # Test/online MI runs do not have trial labels in MOABB; use both
        # sessions of the labelled offline/train run only.
        return Lee2019_MI(train_run=True,test_run=False,sessions=[1,2]),tuple(range(1,55)),4.0
    raise ValueError("Unsupported frozen dataset "+name)

def covariances(X):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=3 or X.shape[1]!=len(COMMON21) or not np.isfinite(X).all():
        raise ValueError("Expected finite [trial,21,time] EEG")
    X=X-X.mean(axis=-1,keepdims=True)
    C=X@X.transpose(0,2,1)/X.shape[-1]
    scale=np.trace(C,axis1=1,axis2=2)/X.shape[1]
    C+=np.eye(X.shape[1])[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
    return (C+C.transpose(0,2,1))/2

def reference(C):
    M=np.mean(C,axis=0)
    e,U=np.linalg.eigh((M+M.T)/2)
    if np.any(~np.isfinite(e)) or np.any(e<=0):
        raise ValueError("Source tangent reference not SPD")
    return (U*(e**-.5))@U.T

def tangent(C,inv):
    Z=np.einsum("ab,nbc,cd->nad",inv,C,inv)
    Z=(Z+Z.transpose(0,2,1))/2
    e,U=np.linalg.eigh(Z)
    if np.any(~np.isfinite(e)) or np.any(e<=0):
        raise ValueError("Trial covariance not SPD after source whitening")
    L=(U*np.log(e)[:,None,:])@U.transpose(0,2,1)
    i,j=np.triu_indices(L.shape[1])
    out=L[:,i,j]
    out[:,i!=j]*=np.sqrt(2.)
    return out

def moments(X,y):
    a=X[y==0];b=X[y==1]
    if len(a)<2 or len(b)<2:raise ValueError("Both classes need >=2 rows")
    m0=a.mean(0);m1=b.mean(0)
    R=np.concatenate((a-m0,b-m1),axis=0)
    S=(R.T@R)/(len(R)-2)
    return (S+S.T)/2,m1-m0,(m0+m1)/2,R

def spectral_state(X,y):
    S,d,mid,R=moments(X,y)
    e,U=np.linalg.eigh(S)
    nu=max(float(np.mean(e)),np.finfo(float).eps)
    return e,U,d,mid,nu

def ridge_scores(state,X,lam):
    e,U,d,mid,nu=state
    r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):
        raise np.linalg.LinAlgError("Infeasible scalar shrinkage candidate")
    w=U@((U.T@d)/r)
    return (X-mid)@w

def predictive_nll(state,residual,lam):
    e,U,d,mid,nu=state
    r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):
        raise np.linalg.LinAlgError("Infeasible covariance candidate")
    Z=residual@U
    return float((np.sum(np.log(r))+np.mean(np.sum((Z*Z)/r,axis=1)))/len(r))

def analytic_scores(Xtr,ytr,Xte,kind):
    _,d,mid,R=moments(Xtr,ytr)
    est=(LedoitWolf(store_precision=False) if kind=="LW" else OAS(store_precision=False)).fit(R)
    C=(est.covariance_+est.covariance_.T)/2
    C+=np.eye(C.shape[0])*max(float(np.trace(C)),1.)*1e-12
    w=np.linalg.solve(C,d)
    return (Xte-mid)@w,float(est.shrinkage_)

def source_path(C,y,g):
    people=sorted(set(map(int,g)))
    if len(people)<20:raise ValueError("External source pool unexpectedly small")
    store={lam:{"auc":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID}
    fold_receipts=[]
    for fold in range(5):
        held_people=[s for s in people if s%5==fold]
        va=np.isin(g,held_people);tr=~va
        if not va.any() or not tr.any():raise ValueError("Empty deterministic source fold")
        inv=reference(C[tr])
        Xtr=tangent(C[tr],inv);Xva=tangent(C[va],inv)
        state=spectral_state(Xtr,y[tr])
        by_person={}
        for sid in held_people:
            m=g[va]==sid
            yy=y[va][m];XX=Xva[m]
            if set(np.unique(yy))!={0,1}:raise ValueError(f"source validation subject {sid} lacks class")
            _,_,_,R=moments(XX,yy)
            by_person[sid]=(yy,XX,R)
        fold_receipts.append({"fold":fold,"train_subjects":[int(s) for s in people if s not in held_people],
                              "validation_subjects":list(map(int,held_people)),
                              "train_trials":int(tr.sum()),"validation_trials":int(va.sum())})
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            aucs=[];nlls=[]
            try:
                for sid in held_people:
                    yy,XX,R=by_person[sid]
                    aucs.append(float(roc_auc_score(yy,ridge_scores(state,XX,lam))))
                    nlls.append(predictive_nll(state,R,lam))
            except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                feasible[lam]=False
                continue
            if not np.isfinite(aucs).all() or not np.isfinite(nlls).all():
                feasible[lam]=False
                continue
            store[lam]["auc"].extend(aucs);store[lam]["nll"].extend(nlls)
    good=[lam for lam in FULL_GRID if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("No full-domain candidate feasible on every source participant")
    crit={}
    for lam in good:
        a=np.asarray(store[lam]["auc"]);n=np.asarray(store[lam]["nll"])
        crit[lam]={"source_macro_auc":float(np.mean(a)),
                   "source_auc_se":float(np.std(a,ddof=1)/np.sqrt(len(a))),
                   "covariance_mean_nll":float(np.mean(n)),
                   "covariance_nll_se":float(np.std(n,ddof=1)/np.sqrt(len(n))),
                   "per_source_auc":list(map(float,a)),
                   "per_source_cov_nll":list(map(float,n))}
    ld=max(good,key=lambda x:(crit[x]["source_macro_auc"],x))
    lc=min(good,key=lambda x:(crit[x]["covariance_mean_nll"],x))
    best_a=crit[ld]["source_macro_auc"];se_a=crit[ld]["source_auc_se"]
    best_c=crit[lc]["covariance_mean_nll"];se_c=crit[lc]["covariance_nll_se"]
    D=[l for l in good if crit[l]["source_macro_auc"]>=best_a-se_a-1e-15]
    C1=[l for l in good if crit[l]["covariance_mean_nll"]<=best_c+se_c+1e-15]
    inter=sorted(set(D)&set(C1));union=set(D)|set(C1)
    return {"lambda_discriminability":float(ld),"lambda_covariance_nll":float(lc),
            "discriminability_boundary_hit":bool(ld in (0.,1.)),
            "covariance_boundary_hit":bool(lc in (0.,1.)),
            "discriminability_1se_lambdas":list(map(float,D)),
            "covariance_1se_lambdas":list(map(float,C1)),
            "one_se_intersection":list(map(float,inter)),
            "one_se_jaccard":float(len(inter)/len(union)) if union else 1.,
            "source_auc_regret_at_covariance_lambda":
                float(crit[ld]["source_macro_auc"]-crit[lc]["source_macro_auc"]),
            "covariance_nll_regret_at_discriminability_lambda":
                float(crit[ld]["covariance_mean_nll"]-crit[lc]["covariance_mean_nll"]),
            "feasible_lambdas":list(map(float,good)),
            "infeasible_lambdas":[float(x) for x in FULL_GRID if x not in good],
            "criteria_by_lambda":{str(k):v for k,v in crit.items()},
            "inner_fold_receipts":fold_receipts}

def load_all(name):
    dataset,subjects,tmax=build_dataset(name)
    paradigm=MotorImagery(n_classes=2,events=["left_hand","right_hand"],
                          fmin=8,fmax=30,tmin=0.0,tmax=tmax,baseline=None,
                          channels=list(COMMON21),resample=160,
                          reject_by_annotation=False)
    covs=[];ys=[];groups=[];meta_receipt=[]
    for sid in subjects:
        X,labels,meta=paradigm.get_data(dataset=dataset,subjects=[sid])
        labels=np.asarray(labels)
        if set(labels)!={"left_hand","right_hand"}:
            raise ValueError(f"Unexpected labels for subject {sid}: {set(labels)}")
        y=(labels=="right_hand").astype(np.int64)
        if set(np.unique(y))!={0,1}:raise ValueError("Missing one class")
        C=covariances(X)
        covs.append(C);ys.append(y);groups.extend([sid]*len(y))
        sessions=sorted(set(map(str,meta["session"]))) if "session" in meta else []
        runs=sorted(set(map(str,meta["run"]))) if "run" in meta else []
        meta_receipt.append({"subject":sid,"trials":int(len(y)),
                             "left_hand":int(np.sum(y==0)),"right_hand":int(np.sum(y==1)),
                             "sessions":sessions,"runs":runs,
                             "samples_per_trial":int(X.shape[-1])})
        del X
    return np.concatenate(covs),np.concatenate(ys),np.asarray(groups),meta_receipt,subjects,tmax

def score_outer(C,y,g,target):
    tr=g!=target;te=~tr
    if target in set(g[tr]):raise AssertionError("Outer target leakage")
    t0=time.perf_counter()
    path=source_path(C[tr],y[tr],g[tr])
    inv=reference(C[tr])
    Xtr=tangent(C[tr],inv);Xte=tangent(C[te],inv)
    scores={}
    scores["LedoitWolf_LDA"],lw=analytic_scores(Xtr,y[tr],Xte,"LW")
    scores["OAS_LDA"],oas=analytic_scores(Xtr,y[tr],Xte,"OAS")
    state=spectral_state(Xtr,y[tr])
    scores["source_macro_full_domain_LDA"]=ridge_scores(
        state,Xte,path["lambda_discriminability"])
    scores["source_cov_nll_full_domain_LDA"]=ridge_scores(
        state,Xte,path["lambda_covariance_nll"])
    lr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000)
    lr.fit(Xtr,y[tr]);scores["TS_LogisticRegression"]=lr.decision_function(Xte)
    mdm=MDM(metric="riemann",n_jobs=1)
    mdm.fit(C[tr],y[tr]);dist=mdm.transform(C[te])
    if dist.shape[1]!=2:raise ValueError("MDM did not return two class distances")
    scores["Riemannian_MDM"]=dist[:,0]-dist[:,1]
    # Only now evaluate target labels.
    yy=y[te]
    auc={name:float(roc_auc_score(yy,score)) for name,score in scores.items()}
    return {"target_subject":int(target),
            "source_subjects":[int(x) for x in sorted(set(g[tr]))],
            "target_trials":int(te.sum()),
            "target_label_counts":{"left_hand":int(np.sum(yy==0)),
                                   "right_hand":int(np.sum(yy==1))},
            "target_information_track":"Z0",
            "target_data_used_for_selection":False,
            "target_batch_moments_used":False,
            "analytic_shrinkage":{"LedoitWolf":lw,"OAS":oas},
            "source_path":path,"auc":auc,
            "fold_wall_seconds":float(time.perf_counter()-t0)}

def paired_summary(folds,candidate,baseline):
    d=np.asarray([f["auc"][candidate]-f["auc"][baseline] for f in folds])
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(d),size=(20000,len(d)))
    lo,hi=np.quantile(np.mean(d[idx],axis=1),[.025,.975])
    signs=rng.choice((-1.,1.),size=(200000,len(d)))
    obs=abs(float(np.mean(d)))
    stats=np.abs(np.mean(signs*d[None,:],axis=1))
    p=float((np.sum(stats>=obs-1e-15)+1)/(len(stats)+1))
    return {"candidate":candidate,"baseline":baseline,"n_participants":len(d),
            "mean_delta_auc":float(np.mean(d)),"median_delta_auc":float(np.median(d)),
            "q25_delta_auc":float(np.quantile(d,.25)),
            "min_delta_auc":float(np.min(d)),"max_delta_auc":float(np.max(d)),
            "improved":int(np.sum(d>0)),"tied":int(np.sum(d==0)),
            "harmed_gt_0_02":int(np.sum(d<-.02)),
            "participant_bootstrap_ci95":[float(lo),float(hi)],
            "mc_signflip_p_two_sided":p,"mc_signflip_draws":200000}

def holm(ps):
    p=np.asarray(ps);order=np.argsort(p);out=np.empty(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*float(p[i])));out[i]=running
    return list(map(float,out))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True,choices=["Cho2017","Lee2019_MI"])
    ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    C,y,g,receipt,subjects,tmax=load_all(a.dataset)
    if set(np.unique(g))!=set(subjects):raise ValueError("Dataset participant denominator drift")
    folds=[score_outer(C,y,g,sid) for sid in subjects]
    methods=list(folds[0]["auc"])
    if any(set(f["auc"])!=set(methods) for f in folds):raise ValueError("Method-set drift")
    macro={m:float(np.mean([f["auc"][m] for f in folds])) for m in methods}
    comparisons=[
      paired_summary(folds,"source_macro_full_domain_LDA","LedoitWolf_LDA"),
      paired_summary(folds,"source_macro_full_domain_LDA","source_cov_nll_full_domain_LDA")]
    adj=holm([x["mc_signflip_p_two_sided"] for x in comparisons])
    for x,padj in zip(comparisons,adj):x["holm_p_H2_H3"]=padj
    result={
      "status":"REAL_EXTERNAL_FULL_DOMAIN_PROSPECTIVE",
      "dataset":a.dataset,"subjects":list(subjects),"n_subjects":len(subjects),
      "task":"left_hand_vs_right_hand","channels":list(COMMON21),
      "feature_dimension":231,"band_hz":[8,30],"epoch_seconds":[0,tmax],
      "resample_hz":160,"reject_by_annotation":False,
      "target_information_track":"Z0",
      "target_labels_used_before_scoring":False,
      "target_batch_statistics_used":False,
      "full_domain_grid":list(FULL_GRID),
      "source_inner_split":"deterministic participant_id modulo 5, geometry refit without held source fold",
      "participant_receipt":receipt,
      "mean_auc_by_method":macro,"primary_comparisons":comparisons,
      "folds":folds,
      "environment":{"python":sys.version,"platform":platform.platform(),
                     "moabb":moabb.__version__,"sklearn":sklearn.__version__,
                     "pyriemann":pyriemann.__version__},
      "wall_seconds":round(time.monotonic()-start,2),
      "interpretation":"Untouched external cohort; participant is inferential unit; no cross-paper metric comparison."}
    p=out/(a.dataset+"_full_domain_lopo.json")
    p.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"status":result["status"],"dataset":a.dataset,
                      "n_subjects":len(subjects),"mean_auc_by_method":macro,
                      "primary_comparisons":comparisons,
                      "wall_seconds":result["wall_seconds"]},indent=2))
    return result

if __name__=="__main__":main()
