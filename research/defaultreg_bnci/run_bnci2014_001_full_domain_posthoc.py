"""BNCI2014_001 full-domain reviewer stress audit.

IMPORTANT: post-hoc after the first BNCI 9-subject outcomes were exposed.
Purpose: test the structural search-grid criticism, not provide independent
confirmation. The full scalar shrinkage domain includes zero, the analytic
LW/OAS neighbourhood, and one.
"""
from __future__ import annotations
import argparse,json,time,itertools
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from run_bnci2014_001_loso import covariances,reference,tangent,within

SUBJECTS=tuple(range(1,10))
FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)

def rank_auc(y,s):
    y=np.asarray(y);s=np.asarray(s,dtype=float)
    n1=int(np.sum(y==1));n0=int(np.sum(y==0))
    if n1==0 or n0==0:raise ValueError("AUC needs both classes")
    order=np.argsort(s,kind="mergesort");sv=s[order]
    rank=np.empty(len(s),dtype=float);i=0
    while i<len(s):
        j=i+1
        while j<len(s) and sv[j]==sv[i]:j+=1
        rank[order[i:j]]=(i+j+1)/2;i=j
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def spectral_state(X,y):
    S,d,mid=within(X,y)
    eig,U=np.linalg.eigh((S+S.T)/2)
    nu=max(float(np.mean(eig)),np.finfo(float).eps)
    return eig,U,d,mid,nu

def fixed_scores(state,X,lam):
    eig,U,d,mid,nu=state
    r=(1-lam)*eig+lam*nu
    tol=64*np.finfo(float).eps*max(nu,float(np.max(np.abs(eig))),1.)
    if np.any(r<=tol):raise np.linalg.LinAlgError("singular candidate")
    w=U@((U.T@d)/r)
    return (X-mid)@w

def cov_nll(state,resid_valid,lam):
    eig,U,d,mid,nu=state
    r=(1-lam)*eig+lam*nu
    tol=64*np.finfo(float).eps*max(nu,float(np.max(np.abs(eig))),1.)
    if np.any(r<=tol):raise np.linalg.LinAlgError("singular candidate")
    Z=resid_valid@U
    return float((np.sum(np.log(r))+np.mean(np.sum((Z*Z)/r,axis=1)))/len(r))

def source_full_path(covs,y,groups):
    people=sorted(set(map(int,groups)))
    store={lam:{"auc":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID}
    for held in people:
        tr=groups!=held;va=~tr
        inv=reference(covs[tr])
        Xtr=tangent(covs[tr],inv);Xva=tangent(covs[va],inv)
        state=spectral_state(Xtr,y[tr])
        Sva,dva,midva=within(Xva,y[va])
        # Residuals for covariance predictive loss only.
        a=Xva[y[va]==0];b=Xva[y[va]==1]
        resid=np.concatenate((a-a.mean(0),b-b.mean(0)),axis=0)
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            try:
                pred=fixed_scores(state,Xva,lam)
                auc=rank_auc(y[va],pred)
                nll=cov_nll(state,resid,lam)
            except (np.linalg.LinAlgError,FloatingPointError,ValueError):
                feasible[lam]=False;continue
            if not np.isfinite(auc) or not np.isfinite(nll):
                feasible[lam]=False;continue
            store[lam]["auc"].append(auc);store[lam]["nll"].append(nll)
    good=[lam for lam in FULL_GRID if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("No lambda feasible on all source folds")
    crit={}
    for lam in good:
        a=np.asarray(store[lam]["auc"]);n=np.asarray(store[lam]["nll"])
        crit[lam]={
          "source_macro_auc":float(np.mean(a)),
          "source_auc_se":float(np.std(a,ddof=1)/np.sqrt(len(a))),
          "covariance_mean_nll":float(np.mean(n)),
          "covariance_nll_se":float(np.std(n,ddof=1)/np.sqrt(len(n))),
          "per_source_auc":list(map(float,a)),
          "per_source_cov_nll":list(map(float,n))}
    ld=max(good,key=lambda x:(crit[x]["source_macro_auc"],x))
    lc=min(good,key=lambda x:(crit[x]["covariance_mean_nll"],x))
    return {
      "lambda_discriminability":float(ld),
      "lambda_covariance_nll":float(lc),
      "feasible_lambdas":list(map(float,good)),
      "infeasible_lambdas":[float(x) for x in FULL_GRID if x not in good],
      "criteria_by_lambda":{str(k):v for k,v in crit.items()},
      "discriminability_boundary_hit":bool(ld in (0.,1.)),
      "covariance_boundary_hit":bool(lc in (0.,1.)),
      "source_auc_regret_at_covariance_lambda":
        float(crit[ld]["source_macro_auc"]-crit[lc]["source_macro_auc"]),
      "cov_nll_regret_at_discriminability_lambda":
        float(crit[ld]["covariance_mean_nll"]-crit[lc]["covariance_mean_nll"])}

def analytic_lambda(X,y,which):
    a=X[y==0];b=X[y==1]
    resid=np.concatenate((a-a.mean(0),b-b.mean(0)),axis=0)
    if which=="LW":return float(LedoitWolf(store_precision=False).fit(resid).shrinkage_)
    if which=="OAS":return float(OAS(store_precision=False).fit(resid).shrinkage_)
    raise ValueError(which)

def exact_signflip(d):
    d=np.asarray(d,dtype=float);n=len(d);obs=abs(float(np.mean(d)))
    exceed=0;total=1<<n
    for mask in range(total):
        s=np.ones(n)
        for i in range(n):
            if mask&(1<<i):s[i]=-1.
        if abs(float(np.mean(s*d)))>=obs-1e-15:exceed+=1
    return float(exceed/total)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="defaultreg_bnci_full_domain_posthoc")
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    dataset=BNCI2014_001(artifact_handling="ignore")
    paradigm=MotorImagery(events=["left_hand","right_hand"],n_classes=2,
                          fmin=8,fmax=30,tmin=0.0,tmax=4.0,
                          baseline=None,reject_by_annotation=False)
    X,labels,metadata=paradigm.get_data(dataset=dataset,subjects=list(SUBJECTS))
    labels=np.asarray(labels);y=(labels=="right_hand").astype(np.int64)
    groups=np.asarray(metadata["subject"],dtype=int)
    if set(groups)!=set(SUBJECTS):raise ValueError("Missing subject")
    C=covariances(X)
    folds=[]
    for target in SUBJECTS:
        tr=groups!=target;te=~tr
        path=source_full_path(C[tr],y[tr],groups[tr])
        inv=reference(C[tr]);Xtr=tangent(C[tr],inv);Xte=tangent(C[te],inv)
        state=spectral_state(Xtr,y[tr])
        lw=analytic_lambda(Xtr,y[tr],"LW");oa=analytic_lambda(Xtr,y[tr],"OAS")
        lambdas={"LedoitWolf_LDA":lw,"OAS_LDA":oa,
                 "source_macro_full_domain_LDA":path["lambda_discriminability"],
                 "source_cov_nll_full_domain_LDA":path["lambda_covariance_nll"]}
        scores={name:fixed_scores(state,Xte,float(lam)) for name,lam in lambdas.items()}
        lr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000)
        lr.fit(Xtr,y[tr]);scores["LogisticRegression"]=lr.decision_function(Xte)
        folds.append({
          "target_subject":int(target),
          "source_subjects":[int(x) for x in sorted(set(groups[tr]))],
          "target_trials":int(te.sum()),
          "target_data_used_for_selection":False,
          "analytic_shrinkage":{"LedoitWolf":lw,"OAS":oa},
          "source_path":path,
          "auc":{k:float(roc_auc_score(y[te],v)) for k,v in scores.items()}})
    methods=list(folds[0]["auc"])
    macro={m:float(np.mean([f["auc"][m] for f in folds])) for m in methods}
    d=np.asarray([f["auc"]["source_macro_full_domain_LDA"]-f["auc"]["LedoitWolf_LDA"]
                  for f in folds])
    result={
      "status":"REAL_BNCI2014_001_FULL_DOMAIN_POSTHOC_AFTER_TARGET_EXPOSURE",
      "confirmation_status":"NOT_INDEPENDENT_CONFIRMATION",
      "reason":"All 9 outer target outcomes were exposed in the earlier truncated-grid run.",
      "dataset":"BNCI2014_001 / BCI Competition IV 2a",
      "subjects":list(SUBJECTS),"full_domain_grid":list(FULL_GRID),
      "target_batch_statistics_used":False,"target_labels_used_before_scoring":False,
      "mean_auc_by_method":macro,"folds":folds,
      "source_macro_full_domain_minus_LW":{
        "mean_delta_auc":float(np.mean(d)),
        "median_delta_auc":float(np.median(d)),
        "improved_subjects":int(np.sum(d>0)),
        "min_delta_auc":float(np.min(d)),
        "max_delta_auc":float(np.max(d)),
        "exact_signflip_p_descriptive":exact_signflip(d)},
      "wall_seconds":round(time.monotonic()-t0,2)}
    (out/"bnci_full_domain_posthoc.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))
    return result

if __name__=="__main__":main()
