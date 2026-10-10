"""Prospective external validation for Default Regularisation on Cho2017/Lee2019_MI.

Frozen before target outcomes in FROZEN_CHO_LEE_Z0_FULL_DOMAIN_20261010.md.
Primary information budget is Z0: target labels/statistics never choose a
representation, shrinkage strength, tangent reference, or model.

The script processes one participant at a time into 21x21 trial covariance
matrices, so raw EEG from all participants is never held in memory together.
"""
from __future__ import annotations
import argparse,json,math,platform,sys,time
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from moabb.datasets import Cho2017,Lee2019_MI
from moabb.paradigms import MotorImagery

COMMON21=("FC5","FC3","FC1","FCz","FC2","FC4","FC6",
          "C5","C3","C1","Cz","C2","C4","C6",
          "CP5","CP3","CP1","CPz","CP2","CP4","CP6")
FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)

CONFIG={
 "cho2017":{
   "dataset_name":"Cho2017",
   "dataset_factory":lambda:Cho2017(),
   "subjects":tuple(range(1,53)),
   "source":tuple(range(1,27)),
   "target":tuple(range(27,53)),
   "tmax":3.0,
   "citation":"Cho et al. 2017, GigaScience; data DOI 10.5524/100295"},
 "lee2019_mi":{
   "dataset_name":"Lee2019_MI",
   "dataset_factory":lambda:Lee2019_MI(train_run=True,test_run=False,resting_state=False),
   "subjects":tuple(range(1,55)),
   "source":tuple(range(1,28)),
   "target":tuple(range(28,55)),
   "tmax":4.0,
   "citation":"Lee et al. 2019, GigaScience/OpenBMI"}}

def covariances(X):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=3 or X.shape[1]!=len(COMMON21) or not np.isfinite(X).all():
        raise ValueError("Expected finite [trial,21,time] EEG")
    X=X-X.mean(axis=-1,keepdims=True)
    C=X@X.transpose(0,2,1)/X.shape[-1]
    scale=np.trace(C,axis1=1,axis2=2)/C.shape[1]
    C+=np.eye(C.shape[1])[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
    return C

def reference(C):
    M=np.mean(C,axis=0);M=(M+M.T)/2
    e,U=np.linalg.eigh(M)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("source reference not SPD")
    return (U*(e**-0.5))@U.T

def tangent(C,inv):
    Z=np.einsum("ab,nbc,cd->nad",inv,C,inv)
    Z=(Z+Z.transpose(0,2,1))/2
    e,U=np.linalg.eigh(Z)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("whitened covariance not SPD")
    L=(U*np.log(e)[:,None,:])@U.transpose(0,2,1)
    p=L.shape[1];i,j=np.triu_indices(p)
    out=L[:,i,j]
    out[:,i!=j]*=np.sqrt(2.0)
    return out

def within(X,y):
    a=X[y==0];b=X[y==1]
    if len(a)<2 or len(b)<2:raise ValueError("both classes need >=2 trials")
    ma=a.mean(0);mb=b.mean(0)
    R=np.concatenate([a-ma,b-mb])
    S=R.T@R/(len(R)-2)
    return (S+S.T)/2,mb-ma,(ma+mb)/2,R

def rank_auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    n1=int(np.sum(y==1));n0=int(np.sum(y==0))
    if n1==0 or n0==0:raise ValueError("AUC requires both classes")
    order=np.argsort(s,kind="mergesort");sv=s[order]
    ranks=np.empty(len(s),float);i=0
    while i<len(s):
        j=i+1
        while j<len(s) and sv[j]==sv[i]:j+=1
        ranks[order[i:j]]=(i+j+1)/2;i=j
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def source_full_path(C,y,g,tail_fraction=.25):
    people=tuple(sorted(map(int,np.unique(g))))
    if len(people)<8:raise ValueError("too few source participants")
    store={lam:{"auc":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID}
    for held in people:
        tr=g!=held;va=~tr
        inv=reference(C[tr])
        Xtr=tangent(C[tr],inv);Xva=tangent(C[va],inv)
        S,d,mid,Rtr=within(Xtr,y[tr])
        _,_,_,Rva=within(Xva,y[va])
        e,U=np.linalg.eigh(S)
        if np.any(~np.isfinite(e)):raise ValueError("nonfinite spectrum")
        nu=max(float(np.mean(e)),np.finfo(float).eps)
        du=U.T@d;V=(Xva-mid)@U;Rv=Rva@U
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            r=(1-lam)*e+lam*nu
            if np.any(~np.isfinite(r)) or float(np.min(r))<=64*np.finfo(float).eps*nu:
                feasible[lam]=False;continue
            w=du/r
            auc=rank_auc(y[va],V@w)
            nll=float((np.sum(np.log(r))+np.mean(np.sum((Rv*Rv)/r,axis=1)))/len(r))
            if not np.isfinite(auc+nll):
                feasible[lam]=False;continue
            store[lam]["auc"].append(auc);store[lam]["nll"].append(nll)
    good=[lam for lam in FULL_GRID if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("no full-domain candidate feasible on all source folds")
    crit={}
    k=max(1,int(math.ceil(tail_fraction*len(people))))
    for lam in good:
        a=np.asarray(store[lam]["auc"],float);n=np.asarray(store[lam]["nll"],float)
        crit[lam]={
          "source_macro_auc":float(a.mean()),
          "source_auc_se":float(a.std(ddof=1)/np.sqrt(len(a))),
          "source_lower_tail_auc":float(np.mean(np.sort(a)[:k])),
          "covariance_mean_nll":float(n.mean()),
          "covariance_nll_se":float(n.std(ddof=1)/np.sqrt(len(n))),
          "per_source_auc":list(map(float,a)),
          "per_source_cov_nll":list(map(float,n))}
    ld=max(good,key=lambda x:(crit[x]["source_macro_auc"],x))
    lc=min(good,key=lambda x:(crit[x]["covariance_mean_nll"],x))
    bestD=crit[ld]["source_macro_auc"];seD=crit[ld]["source_auc_se"]
    bestC=crit[lc]["covariance_mean_nll"];seC=crit[lc]["covariance_nll_se"]
    D1=tuple(l for l in good if crit[l]["source_macro_auc"]>=bestD-seD-1e-15)
    C1=tuple(l for l in good if crit[l]["covariance_mean_nll"]<=bestC+seC+1e-15)
    inter=tuple(sorted(set(D1)&set(C1)));union=set(D1)|set(C1)
    # FCTR frozen in preregistration.
    tails={l:crit[l]["source_lower_tail_auc"] for l in C1}
    cand=max(C1,key=lambda l:(tails[l],crit[l]["source_macro_auc"],-abs(l-lc),-l))
    tail_gain=tails[cand]-tails[lc]
    macro_drop=crit[lc]["source_macro_auc"]-crit[cand]["source_macro_auc"]
    accepted=(tail_gain>=0.01-1e-15 and macro_drop<=0.01+1e-15)
    lf=cand if accepted else lc
    return {
      "lambda_discriminability":float(ld),
      "lambda_covariance_nll":float(lc),
      "lambda_fctr":float(lf),
      "fctr_candidate":float(cand),
      "fctr_intervention_accepted":bool(accepted),
      "fctr_source_tail_gain":float(tail_gain),
      "fctr_source_macro_drop":float(macro_drop),
      "covariance_1se_lambdas":list(map(float,C1)),
      "discriminability_1se_lambdas":list(map(float,D1)),
      "one_se_intersection":list(map(float,inter)),
      "one_se_jaccard":float(len(inter)/len(union)) if union else 1.0,
      "source_auc_regret_at_covariance_lambda":float(
          crit[ld]["source_macro_auc"]-crit[lc]["source_macro_auc"]),
      "covariance_nll_regret_at_discriminability_lambda":float(
          crit[ld]["covariance_mean_nll"]-crit[lc]["covariance_mean_nll"]),
      "discriminability_domain_boundary_hit":bool(ld in (0.,1.)),
      "covariance_domain_boundary_hit":bool(lc in (0.,1.)),
      "feasible_lambdas":list(map(float,good)),
      "infeasible_lambdas":[float(x) for x in FULL_GRID if x not in good],
      "criteria_by_lambda":{str(l):crit[l] for l in good},
      "source_groups":list(people)}

def final_spectral_model(X,y):
    S,d,mid,_=within(X,y)
    e,U=np.linalg.eigh(S);nu=max(float(np.mean(e)),np.finfo(float).eps)
    return e,U,nu,d,mid

def fixed_scores(model,X,lam):
    e,U,nu,d,mid=model
    r=(1-lam)*e+lam*nu
    if np.any(r<=64*np.finfo(float).eps*nu):raise ValueError("selected lambda infeasible")
    w=U@((U.T@d)/r)
    return (X-mid)@w

def analytic_scores(Xs,ys,Xt,est):
    S,d,mid,R=within(Xs,ys)
    fitted=est.fit(R)
    Rcov=(fitted.covariance_+fitted.covariance_.T)/2
    Rcov+=np.eye(Rcov.shape[0])*max(float(np.trace(Rcov)),1.)*1e-12
    w=np.linalg.solve(Rcov,d)
    return (Xt-mid)@w,float(fitted.shrinkage_)

def signflip(d,seed=20261010,draws=200000):
    d=np.asarray(d,float);obs=abs(float(d.mean()));n=len(d)
    tol=64*np.finfo(float).eps*max(1.,obs)
    if n<=20:
        total=1<<n;hit=0
        for mask in range(total):
            s=np.ones(n)
            for i in range(n):
                if mask&(1<<i):s[i]=-1
            hit+=abs(float(np.mean(s*d)))>=obs-tol
        return float(hit/total),"exact",total
    rng=np.random.default_rng(seed);hit=0;done=0
    while done<draws:
        q=min(10000,draws-done)
        s=rng.choice((-1.,1.),size=(q,n))
        hit+=int(np.sum(np.abs(np.mean(s*d[None,:],axis=1))>=obs-tol));done+=q
    return float((hit+1)/(draws+1)),"monte_carlo",draws

def bootstrap_ci(d,seed=20261010,reps=20000):
    d=np.asarray(d,float);rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(reps,len(d)))
    return [float(x) for x in np.quantile(np.mean(d[idx],axis=1),[.025,.975])]

def paired_summary(rows,base,cand):
    d=np.asarray([r["auc"][cand]-r["auc"][base] for r in rows],float)
    p,mode,draws=signflip(d)
    return {
      "baseline":base,"candidate":cand,"n_participants":len(d),
      "mean_delta":float(d.mean()),"median_delta":float(np.median(d)),
      "q25_delta":float(np.quantile(d,.25)),"worst_delta":float(d.min()),
      "best_delta":float(d.max()),"improved":int(np.sum(d>0)),
      "tied":int(np.sum(d==0)),"harmed_gt_0_02":int(np.sum(d<-.02)),
      "bootstrap95":bootstrap_ci(d),"signflip_p":p,
      "signflip_mode":mode,"signflip_draws":draws}

def holm_two(p1,p2):
    vals=[p1,p2];order=np.argsort(vals);adj=[0.,0.];running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.,(2-rank)*vals[idx]));adj[int(idx)]=running
    return adj

def load_dataset(key):
    cfg=CONFIG[key];dataset=cfg["dataset_factory"]()
    paradigm=MotorImagery(events=["left_hand","right_hand"],n_classes=2,
        fmin=8,fmax=30,tmin=0.0,tmax=cfg["tmax"],baseline=None,
        channels=list(COMMON21),resample=160,reject_by_annotation=False)
    covs=[];ys=[];groups=[];metadata_receipt=[]
    for sid in cfg["subjects"]:
        t=time.monotonic()
        X,labels,meta=paradigm.get_data(dataset=dataset,subjects=[sid])
        labels=np.asarray(labels)
        if set(labels)!={"left_hand","right_hand"}:
            raise ValueError(f"subject {sid}: unexpected labels {set(labels)}")
        y=(labels=="right_hand").astype(np.int64)
        if len(np.unique(y))!=2:raise ValueError(f"subject {sid}: one class absent")
        C=covariances(X)
        covs.append(C);ys.append(y);groups.extend([sid]*len(y))
        row={"subject":sid,"trials":int(len(y)),
             "left":int(np.sum(y==0)),"right":int(np.sum(y==1)),
             "array_shape":list(map(int,X.shape)),
             "sessions":sorted({str(x) for x in meta["session"].tolist()}) if "session" in meta else [],
             "runs":sorted({str(x) for x in meta["run"].tolist()}) if "run" in meta else [],
             "processing_seconds":round(time.monotonic()-t,3)}
        metadata_receipt.append(row)
        print("LOADED",json.dumps(row,sort_keys=True),flush=True)
    return np.concatenate(covs),np.concatenate(ys),np.asarray(groups),metadata_receipt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=sorted(CONFIG),required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args();cfg=CONFIG[a.dataset]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    C,y,g,receipt=load_dataset(a.dataset)
    src=np.isin(g,cfg["source"]);tgt=np.isin(g,cfg["target"])
    if set(np.unique(g[src]))!=set(cfg["source"]):raise ValueError("source block incomplete")
    if set(np.unique(g[tgt]))!=set(cfg["target"]):raise ValueError("target block incomplete")
    if np.intersect1d(np.unique(g[src]),np.unique(g[tgt])).size:raise ValueError("source/target overlap")

    # All choices below use SOURCE ONLY.
    path=source_full_path(C[src],y[src],g[src])
    inv=reference(C[src])
    Xs=tangent(C[src],inv);Xt=tangent(C[tgt],inv)
    model=final_spectral_model(Xs,y[src])

    # Generate every target score vector before evaluating target labels.
    path_scores={str(l):fixed_scores(model,Xt,l) for l in path["feasible_lambdas"]}
    scores={}
    lw,lwlam=analytic_scores(Xs,y[src],Xt,LedoitWolf(store_precision=False))
    oa,oalam=analytic_scores(Xs,y[src],Xt,OAS(store_precision=False))
    scores["LedoitWolf_LDA"]=lw;scores["OAS_LDA"]=oa
    scores["source_cov_nll_full_domain_LDA"]=path_scores[str(path["lambda_covariance_nll"])]
    scores["source_macro_full_domain_LDA"]=path_scores[str(path["lambda_discriminability"])]
    scores["FCTR_LDA"]=path_scores[str(path["lambda_fctr"])]
    lr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000)
    lr.fit(Xs,y[src]);scores["TS_LogisticRegression"]=lr.decision_function(Xt)

    # Target labels enter only here, after source choices and target score vectors exist.
    gt=g[tgt];yt=y[tgt];rows=[]
    for sid in cfg["target"]:
        m=gt==sid
        row={"target_subject":int(sid),"target_trials":int(np.sum(m)),
             "left":int(np.sum(yt[m]==0)),"right":int(np.sum(yt[m]==1)),
             "auc":{name:float(roc_auc_score(yt[m],score[m])) for name,score in scores.items()},
             "path_auc_by_lambda":{lam:float(roc_auc_score(yt[m],score[m]))
                                   for lam,score in path_scores.items()}}
        rows.append(row)

    h2=paired_summary(rows,"source_cov_nll_full_domain_LDA","source_macro_full_domain_LDA")
    h3=paired_summary(rows,"LedoitWolf_LDA","source_macro_full_domain_LDA")
    adj=holm_two(h2["signflip_p"],h3["signflip_p"])
    h2["holm_p_H2_H3"]=adj[0];h3["holm_p_H2_H3"]=adj[1]
    fctr=paired_summary(rows,"source_macro_full_domain_LDA","FCTR_LDA")

    result={
      "status":"REAL_PROSPECTIVE_EXTERNAL_Z0_FULL_DOMAIN",
      "confirmation_status":"UNTOUCHED_TARGET_BLOCK_AT_PROTOCOL_FREEZE",
      "dataset":cfg["dataset_name"],"citation":cfg["citation"],
      "source_participants":list(cfg["source"]),"target_participants":list(cfg["target"]),
      "target_information_budget":"Z0",
      "target_labels_or_statistics_used_for_selection":False,
      "common_channels":list(COMMON21),"n_channels":len(COMMON21),
      "band_hz":[8,30],"resample_hz":160,"epoch_seconds":[0,cfg["tmax"]],
      "reject_by_annotation":False,
      "full_domain_grid":list(FULL_GRID),
      "source_path":path,
      "LedoitWolf_shrinkage":float(lwlam),"OAS_shrinkage":float(oalam),
      "participant_data_receipt":receipt,
      "mean_auc_by_method":{name:float(np.mean([r["auc"][name] for r in rows])) for name in scores},
      "per_target_subject":rows,
      "primary_H2_source_macro_vs_cov_nll":h2,
      "secondary_H3_source_macro_vs_LW":h3,
      "secondary_FCTR_vs_source_macro":fctr,
      "participant_is_inferential_unit":True,
      "shared_source_training_uncertainty_in_primary_bootstrap":False,
      "environment":{"python":sys.version,"platform":platform.platform()},
      "wall_seconds":round(time.monotonic()-started,2)}
    (out/"external_full_domain_result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    np.savez_compressed(out/"trial_covariances.npz",covs=C,y=y,subject=g)
    print(json.dumps({k:result[k] for k in (
      "status","dataset","mean_auc_by_method","primary_H2_source_macro_vs_cov_nll",
      "secondary_H3_source_macro_vs_LW","secondary_FCTR_vs_source_macro","wall_seconds")},
      indent=2,sort_keys=True))
    return result

if __name__=="__main__":main()
