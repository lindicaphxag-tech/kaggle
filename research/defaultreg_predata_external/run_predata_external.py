"""PRE-DATA Default Regularisation external confirmation.

Datasets: Shin2017A and BNCI2014_002.
Frozen before target outcome access on this branch.

Primary question: does source participant-discriminability-optimal scalar
regularisation differ from source covariance-predictive-NLL-optimal
regularisation, and does that choice matter for unseen participants under
strict zero-target (Z0) deployment?

No target feature moments, labels, batch statistics, references or target-
dependent hyperparameters enter model selection. Target labels are accessed
only after every method's score vector (including the entire scalar path) has
been frozen in memory for that outer participant.
"""
from __future__ import annotations
import argparse,itertools,json,math,platform,sys,time
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh as generalized_eigh
import moabb,sklearn,pyriemann
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from moabb.datasets import Shin2017A,BNCI2014_002
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM

FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)
SEED=20261010
TMIN,TMAX=0.0,4.0
RESAMPLE=160
N_CSP=6

def build_dataset(name):
    if name=="Shin2017A":
        return (Shin2017A(accept=True,motor_imagery=True,mental_arithmetic=False),
                tuple(range(1,30)),("left_hand","right_hand"))
    if name=="BNCI2014_002":
        return BNCI2014_002(),tuple(range(1,15)),("right_hand","feet")
    raise ValueError("unsupported frozen dataset")

def covariances(X):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=3 or not np.isfinite(X).all():raise ValueError("finite [trial,channel,time] EEG required")
    X=X-X.mean(axis=-1,keepdims=True)
    C=X@X.transpose(0,2,1)/X.shape[-1]
    s=np.trace(C,axis1=1,axis2=2)/X.shape[1]
    C+=np.eye(X.shape[1])[None,:,:]*np.maximum(s[:,None,None]*1e-6,1e-20)
    return (C+C.transpose(0,2,1))/2

def reference(C):
    M=np.mean(C,axis=0);e,U=np.linalg.eigh((M+M.T)/2)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("source reference not SPD")
    return (U*(e**-.5))@U.T

def tangent(C,inv):
    Z=np.einsum("ab,nbc,cd->nad",inv,C,inv);Z=(Z+Z.transpose(0,2,1))/2
    e,U=np.linalg.eigh(Z)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("trial covariance not SPD")
    L=(U*np.log(e)[:,None,:])@U.transpose(0,2,1)
    i,j=np.triu_indices(L.shape[1]);out=L[:,i,j]
    out[:,i!=j]*=np.sqrt(2.)
    return out

def moments(X,y):
    a=X[y==0];b=X[y==1]
    if len(a)<2 or len(b)<2:raise ValueError("both classes need >=2 trials")
    m0=a.mean(0);m1=b.mean(0)
    R=np.concatenate((a-m0,b-m1),axis=0)
    S=(R.T@R)/(len(R)-2)
    return (S+S.T)/2,m1-m0,(m0+m1)/2,R

def spectral_state(X,y):
    S,d,mid,R=moments(X,y);e,U=np.linalg.eigh(S)
    return e,U,d,mid,max(float(np.mean(e)),np.finfo(float).eps),R

def ridge_scores(state,X,lam):
    e,U,d,mid,nu,_=state
    r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):raise np.linalg.LinAlgError("infeasible lambda")
    w=U@((U.T@d)/r)
    return (X-mid)@w

def predictive_nll(state,residual,lam):
    e,U,_,_,nu,_=state;r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):raise np.linalg.LinAlgError("infeasible lambda")
    Z=residual@U
    return float((np.sum(np.log(r))+np.mean(np.sum((Z*Z)/r,axis=1)))/len(r))

def analytic_scores(Xtr,ytr,Xte,kind):
    _,d,mid,R=moments(Xtr,ytr)
    est=(LedoitWolf(store_precision=False) if kind=="LW" else OAS(store_precision=False)).fit(R)
    C=(est.covariance_+est.covariance_.T)/2
    C+=np.eye(C.shape[0])*max(float(np.trace(C)),1.)*1e-12
    w=np.linalg.solve(C,d)
    return (Xte-mid)@w,float(est.shrinkage_)

def csp_fit(C,y,n_components=N_CSP):
    C0=np.mean(C[y==0],axis=0);C1=np.mean(C[y==1],axis=0)
    total=(C0+C1)/2
    eps=max(float(np.trace(total))/len(total),1e-20)*1e-8
    vals,W=generalized_eigh(C1,C0+C1+eps*np.eye(len(C0)))
    idx=np.argsort(np.abs(vals-.5))[::-1][:min(n_components,len(vals))]
    W=W[:,idx]
    return W,vals[idx]

def csp_features(C,W):
    proj=np.einsum("pi,nij,jq->npq",W,C,W)
    v=np.diagonal(proj,axis1=1,axis2=2)
    v=np.maximum(v,np.finfo(float).tiny)
    v=v/np.sum(v,axis=1,keepdims=True)
    return np.log(v)

def source_path(C,y,g):
    people=sorted(set(map(int,g)))
    if len(people)<8:raise ValueError("too few source participants")
    nfold=min(5,len(people))
    store={lam:{"auc":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID};fold_receipts=[]
    for fold in range(nfold):
        held=[s for s in people if (s-1)%nfold==fold]
        va=np.isin(g,held);tr=~va
        inv=reference(C[tr]);Xtr=tangent(C[tr],inv);Xva=tangent(C[va],inv)
        state=spectral_state(Xtr,y[tr])
        by_person={}
        for sid in held:
            m=g[va]==sid;yy=y[va][m];XX=Xva[m]
            if set(np.unique(yy))!={0,1}:raise ValueError("source validation participant lacks class")
            _,_,_,R=moments(XX,yy);by_person[sid]=(yy,XX,R)
        fold_receipts.append({"fold":fold,"train_subjects":[int(s) for s in people if s not in held],
                              "validation_subjects":list(map(int,held))})
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            aa=[];nn=[]
            try:
                for sid in held:
                    yy,XX,R=by_person[sid]
                    aa.append(float(roc_auc_score(yy,ridge_scores(state,XX,lam))))
                    nn.append(predictive_nll(state,R,lam))
            except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                feasible[lam]=False;continue
            if not np.isfinite(aa).all() or not np.isfinite(nn).all():
                feasible[lam]=False;continue
            store[lam]["auc"].extend(aa);store[lam]["nll"].extend(nn)
    good=[lam for lam in FULL_GRID if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("no shrinkage point feasible in every source fold")
    crit={}
    for lam in good:
        a=np.asarray(store[lam]["auc"]);n=np.asarray(store[lam]["nll"])
        crit[lam]={"source_macro_auc":float(np.mean(a)),
                   "source_auc_se":float(np.std(a,ddof=1)/np.sqrt(len(a))),
                   "covariance_mean_nll":float(np.mean(n)),
                   "covariance_nll_se":float(np.std(n,ddof=1)/np.sqrt(len(n)))}
    ld=max(good,key=lambda z:(crit[z]["source_macro_auc"],z))
    lc=min(good,key=lambda z:(crit[z]["covariance_mean_nll"],z))
    threshold=crit[lc]["covariance_mean_nll"]+crit[lc]["covariance_nll_se"]
    one_se=[z for z in good if crit[z]["covariance_mean_nll"]<=threshold+1e-15]
    return {"lambda_discriminability":float(ld),"lambda_covariance_nll":float(lc),
            "discriminability_boundary_hit":bool(ld in (0.,1.)),
            "covariance_boundary_hit":bool(lc in (0.,1.)),
            "feasible_lambdas":list(map(float,good)),
            "infeasible_lambdas":[float(z) for z in FULL_GRID if z not in good],
            "criteria_by_lambda":{str(z):crit[z] for z in good},
            "covariance_1se_lambdas":list(map(float,one_se)),
            "inner_fold_receipts":fold_receipts}

def load_all(name):
    dataset,subjects,events=build_dataset(name)
    paradigm=MotorImagery(n_classes=2,events=list(events),fmin=8,fmax=30,
                          tmin=TMIN,tmax=TMAX,baseline=None,resample=RESAMPLE,
                          reject_by_annotation=False)
    covs=[];ys=[];groups=[];receipts=[];nchan=None
    for sid in subjects:
        X,labels,meta=paradigm.get_data(dataset=dataset,subjects=[sid])
        labels=np.asarray(labels)
        if set(labels)!=set(events):raise ValueError(f"unexpected labels subject {sid}: {set(labels)}")
        y=(labels==events[1]).astype(np.int64)
        if set(np.unique(y))!={0,1}:raise ValueError("missing class")
        if nchan is None:nchan=X.shape[1]
        if X.shape[1]!=nchan:raise ValueError("channel denominator drift")
        C=covariances(X);covs.append(C);ys.append(y);groups.extend([sid]*len(y))
        sessions=sorted(set(map(str,meta["session"]))) if "session" in meta else []
        runs=sorted(set(map(str,meta["run"]))) if "run" in meta else []
        receipts.append({"subject":sid,"trials":int(len(y)),
                         "class0":int(np.sum(y==0)),"class1":int(np.sum(y==1)),
                         "sessions":sessions,"runs":runs,
                         "channels":int(X.shape[1]),"samples_per_trial":int(X.shape[-1])})
        del X
    return np.concatenate(covs),np.concatenate(ys),np.asarray(groups),receipts,subjects,events,int(nchan)

def score_outer(C,y,g,target):
    tr=g!=target;te=~tr
    if target in set(g[tr]):raise AssertionError("target leakage")
    fold_start=time.perf_counter()
    t=time.perf_counter();path=source_path(C[tr],y[tr],g[tr]);timings={"source_path":time.perf_counter()-t}
    t=time.perf_counter();inv=reference(C[tr]);Xtr=tangent(C[tr],inv);Xte=tangent(C[te],inv)
    timings["tangent_transform"]=time.perf_counter()-t
    scores={}
    t=time.perf_counter();scores["LedoitWolf_LDA"],lw=analytic_scores(Xtr,y[tr],Xte,"LW")
    timings["LedoitWolf_LDA"]=time.perf_counter()-t
    t=time.perf_counter();scores["OAS_LDA"],oas=analytic_scores(Xtr,y[tr],Xte,"OAS")
    timings["OAS_LDA"]=time.perf_counter()-t
    state=spectral_state(Xtr,y[tr])
    t=time.perf_counter()
    path_scores={}
    for lam in path["feasible_lambdas"]:
        path_scores[str(float(lam))]=ridge_scores(state,Xte,float(lam))
    timings["complete_target_scalar_path"]=time.perf_counter()-t
    scores["source_macro_full_domain_LDA"]=path_scores[str(path["lambda_discriminability"])]
    scores["source_cov_nll_full_domain_LDA"]=path_scores[str(path["lambda_covariance_nll"])]
    t=time.perf_counter();lr=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000)
    lr.fit(Xtr,y[tr]);scores["TS_LogisticRegression"]=lr.decision_function(Xte)
    timings["TS_LogisticRegression"]=time.perf_counter()-t
    t=time.perf_counter();mdm=MDM(metric="riemann",n_jobs=1);mdm.fit(C[tr],y[tr]);dist=mdm.transform(C[te])
    if dist.shape[1]!=2:raise ValueError("MDM must return two distances")
    scores["Riemannian_MDM"]=dist[:,0]-dist[:,1];timings["Riemannian_MDM"]=time.perf_counter()-t
    t=time.perf_counter();W,csp_eval=csp_fit(C[tr],y[tr]);Ftr=csp_features(C[tr],W);Fte=csp_features(C[te],W)
    csp_lr=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000);csp_lr.fit(Ftr,y[tr])
    scores["CSP6_LogisticRegression"]=csp_lr.decision_function(Fte)
    timings["CSP6_LogisticRegression"]=time.perf_counter()-t

    # Target labels are first accessed here, AFTER all score vectors exist.
    yy=y[te]
    auc={k:float(roc_auc_score(yy,v)) for k,v in scores.items()}
    path_auc={k:float(roc_auc_score(yy,v)) for k,v in path_scores.items()}
    one=[str(float(l)) for l in path["covariance_1se_lambdas"]]
    vals=[path_auc[k] for k in one]
    timings["outer_total"]=time.perf_counter()-fold_start
    return {"target_subject":int(target),"source_subjects":[int(x) for x in sorted(set(g[tr]))],
            "target_trials":int(te.sum()),"target_information_track":"Z0",
            "target_data_used_for_selection":False,"target_batch_moments_used":False,
            "target_labels_accessed_only_after_score_freeze":True,
            "target_label_counts":{"class0":int(np.sum(yy==0)),"class1":int(np.sum(yy==1))},
            "analytic_shrinkage":{"LedoitWolf":lw,"OAS":oas,
                                  "LedoitWolf_nearest_grid":float(min(FULL_GRID,key=lambda z:abs(z-lw))),
                                  "OAS_nearest_grid":float(min(FULL_GRID,key=lambda z:abs(z-oas)))},
            "source_path":path,"auc":auc,"target_auc_path_by_lambda":path_auc,
            "target_path_scores_frozen_before_labels":True,
            "source_covariance_1se_target_auc_span":float(max(vals)-min(vals)),
            "csp_generalized_eigenvalues":list(map(float,csp_eval)),
            "runtime_seconds":{k:float(v) for k,v in timings.items()}}

def paired_summary(folds,candidate,baseline):
    d=np.asarray([f["auc"][candidate]-f["auc"][baseline] for f in folds],dtype=float)
    rng=np.random.default_rng(SEED)
    idx=rng.integers(0,len(d),size=(20000,len(d)))
    lo,hi=np.quantile(np.mean(d[idx],axis=1),[.025,.975])
    obs=abs(float(np.mean(d)))
    if len(d)<=20:
        signs=np.asarray(list(itertools.product((-1.,1.),repeat=len(d))),dtype=float)
        pv=float(np.mean(np.abs(np.mean(signs*d[None,:],axis=1))>=obs-1e-15))
        engine="exact";draws=len(signs)
    else:
        signs=rng.choice((-1.,1.),size=(200000,len(d)))
        stats=np.abs(np.mean(signs*d[None,:],axis=1))
        pv=float((np.sum(stats>=obs-1e-15)+1)/(len(stats)+1));engine="MC";draws=len(signs)
    k=max(1,int(math.ceil(.25*len(d))))
    return {"candidate":candidate,"baseline":baseline,"n_participants":len(d),
            "mean_delta_auc":float(np.mean(d)),"median_delta_auc":float(np.median(d)),
            "q25_delta_auc":float(np.quantile(d,.25)),"q10_delta_auc":float(np.quantile(d,.10)),
            "min_delta_auc":float(np.min(d)),"max_delta_auc":float(np.max(d)),
            "lower_quartile_mean_delta_auc":float(np.mean(np.sort(d)[:k])),
            "improved":int(np.sum(d>0)),"tied":int(np.sum(d==0)),
            "harmed_gt_0_01":int(np.sum(d<-.01)),"harmed_gt_0_02":int(np.sum(d<-.02)),
            "participant_bootstrap_ci95":[float(lo),float(hi)],
            "signflip_p_two_sided":pv,"signflip_engine":engine,"signflip_draws":int(draws)}

def holm(ps):
    p=np.asarray(ps);order=np.argsort(p);out=np.empty(len(p));run=0.
    for rank,i in enumerate(order):
        run=max(run,min(1.,(len(p)-rank)*float(p[i])));out[i]=run
    return list(map(float,out))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",choices=["Shin2017A","BNCI2014_002"],required=True)
    ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    C,y,g,receipt,subjects,events,nchan=load_all(a.dataset)
    if set(np.unique(g))!=set(subjects):raise ValueError("participant denominator drift")
    folds=[score_outer(C,y,g,sid) for sid in subjects]
    methods=list(folds[0]["auc"])
    if any(set(f["auc"])!=set(methods) for f in folds):raise ValueError("method-set drift")
    macro={m:float(np.mean([f["auc"][m] for f in folds])) for m in methods}
    comps=[paired_summary(folds,"source_macro_full_domain_LDA","source_cov_nll_full_domain_LDA"),
           paired_summary(folds,"source_macro_full_domain_LDA","LedoitWolf_LDA")]
    for row,padj in zip(comps,holm([z["signflip_p_two_sided"] for z in comps])):
        row["holm_p_H2_H3"]=padj
    ld=np.asarray([f["source_path"]["lambda_discriminability"] for f in folds])
    lc=np.asarray([f["source_path"]["lambda_covariance_nll"] for f in folds])
    lw=np.asarray([f["analytic_shrinkage"]["LedoitWolf"] for f in folds])
    result={
      "status":"REAL_PRE_DATA_EXTERNAL_FULL_DOMAIN_COMPLETE",
      "dataset":a.dataset,"subjects":list(subjects),"n_subjects":len(subjects),
      "task":f"{events[0]}_vs_{events[1]}","channel_count":nchan,
      "feature_dimension":int(nchan*(nchan+1)//2),"band_hz":[8,30],
      "epoch_seconds":[TMIN,TMAX],"resample_hz":RESAMPLE,
      "target_information_track":"Z0","target_labels_used_before_scoring":False,
      "target_batch_statistics_used":False,"full_domain_grid":list(FULL_GRID),
      "participant_receipt":receipt,"mean_auc_by_method":macro,
      "primary_comparisons":comps,
      "objective_separation":{
        "folds_lambda_D_not_lambda_C":int(np.sum(ld!=lc)),
        "fraction_lambda_D_not_lambda_C":float(np.mean(ld!=lc)),
        "median_abs_lambda_D_minus_lambda_C":float(np.median(np.abs(ld-lc))),
        "lambda_D_boundary_hits":int(np.sum(np.isin(ld,[0.,1.]))),
        "lambda_C_boundary_hits":int(np.sum(np.isin(lc,[0.,1.]))),
        "median_analytic_LW_shrinkage":float(np.median(lw))},
      "validity_frontier_1se_summary":{
        "mean_target_auc_span":float(np.mean([f["source_covariance_1se_target_auc_span"] for f in folds])),
        "median_target_auc_span":float(np.median([f["source_covariance_1se_target_auc_span"] for f in folds])),
        "max_target_auc_span":float(np.max([f["source_covariance_1se_target_auc_span"] for f in folds])),
        "target_path_used_for_selection":False},
      "runtime_seconds":{
        "total":round(time.monotonic()-start,3),
        "median_outer_fold":float(np.median([f["runtime_seconds"]["outer_total"] for f in folds])),
        "median_source_path":float(np.median([f["runtime_seconds"]["source_path"] for f in folds])),
        "median_CSP":float(np.median([f["runtime_seconds"]["CSP6_LogisticRegression"] for f in folds])),
        "median_MDM":float(np.median([f["runtime_seconds"]["Riemannian_MDM"] for f in folds]))},
      "folds":folds,
      "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
                     "scipy":scipy.__version__,"moabb":moabb.__version__,
                     "sklearn":sklearn.__version__,"pyriemann":pyriemann.__version__},
      "interpretation":"PRE-DATA external replication stratum; no cross-paper metric/SOTA inference."}
    p=out/(a.dataset+"_predata_full_domain_lopo.json")
    p.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"status":result["status"],"dataset":a.dataset,
                      "n_subjects":len(subjects),"mean_auc_by_method":macro,
                      "primary_comparisons":comps,
                      "objective_separation":result["objective_separation"],
                      "runtime_seconds":result["runtime_seconds"]},indent=2))
if __name__=="__main__":main()
