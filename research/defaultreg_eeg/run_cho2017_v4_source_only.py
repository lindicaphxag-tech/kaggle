"""Cho2017 v4 SOURCE-ONLY freeze for Default Regularisation.

This runner intentionally has no code path that loads confirmatory target
subjects. It freezes the source objective landscape before the Cho2017 target
block is opened.

MOABB 1.7.2; upstream provider; 20 common motor channels; left/right imagery;
8-30 Hz; task-relative 0-3 s; 250 Hz; all trials retained.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path

import numpy as np
import moabb
from moabb.datasets import Cho2017
from moabb.paradigms import LeftRightImagery
from sklearn.covariance import LedoitWolf,OAS

SEED=20261010
SOURCE=(3,5,19,26,31,32,36,37,44,45,46,47)
TARGET=(4,6,8,10,11,12,14,15,17,18,20,28,33,34,39,40,43,48,50,52)
RESERVE=(1,2,7,9,13,16,21,22,23,24,25,27,29,30,35,38,41,42,49,51)

COMMON20=(
 "FC5","FC3","FC1","FC2","FC4","FC6",
 "C5","C3","C1","Cz","C2","C4","C6",
 "CP5","CP3","CP1","CPz","CP2","CP4","CP6")

FULL_GRID=(
 0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
 0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
 0.90,0.95,0.98,0.99,1.0)

NEAR_ZERO=(0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.05)
DENSE_GRID=tuple(sorted(set(NEAR_ZERO+tuple(round(i*.0025,10) for i in range(401)))))

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def flatten_paths(x):
    if isinstance(x,(str,Path)):
        return [Path(x)]
    out=[]
    for v in x:
        out.extend(flatten_paths(v))
    return out

def fixed_split_assertion():
    perm=np.random.default_rng(SEED).permutation(np.arange(1,53)).tolist()
    assert tuple(sorted(perm[:12]))==SOURCE
    assert tuple(sorted(perm[12:32]))==TARGET
    assert tuple(sorted(perm[32:]))==RESERVE
    assert set(SOURCE).isdisjoint(TARGET)
    assert set(SOURCE).isdisjoint(RESERVE)
    assert set(TARGET).isdisjoint(RESERVE)

def reference(covs):
    center=np.mean(covs,axis=0)
    val,vec=np.linalg.eigh(center)
    if np.any(val<=0):
        raise ValueError("Source reference not SPD")
    return (vec*(val**(-.5)))@vec.T

def tangent(covs,inv):
    z=np.einsum("ab,nbc,cd->nad",inv,covs,inv)
    eig,rot=np.linalg.eigh(z)
    if np.any(eig<=0):
        raise ValueError("Trial covariance not SPD in source geometry")
    mat=(rot*np.log(eig)[:,None,:])@rot.transpose(0,2,1)
    p=mat.shape[1]
    idx=np.triu_indices(p)
    out=mat[:,idx[0],idx[1]]
    out[:,idx[0]!=idx[1]]*=np.sqrt(2.)
    return out

def within_residual(X,y):
    a=X[y==0];b=X[y==1]
    if len(a)<2 or len(b)<2:
        raise ValueError("Each fold needs both classes")
    m0=a.mean(0);m1=b.mean(0)
    Z=np.concatenate((a-m0,b-m1),axis=0)
    return Z,m1-m0,(m0+m1)/2

def rank_auc(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    n1=int(np.sum(y==1));n0=int(np.sum(y==0))
    if n1==0 or n0==0:
        raise ValueError("AUC needs both classes")
    order=np.argsort(s,kind="mergesort");sv=s[order]
    rank=np.empty(len(s),float);i=0
    while i<len(s):
        j=i+1
        while j<len(s) and sv[j]==sv[i]:
            j+=1
        rank[order[i:j]]=(i+j+1)/2
        i=j
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def source_path(covs,y,groups,grid):
    people=tuple(sorted(set(map(int,groups))))
    if people!=SOURCE:
        raise ValueError(f"Unexpected source IDs: {people}")
    store={lam:{"auc":[],"nll":[]} for lam in grid}
    feasible={lam:True for lam in grid}
    for held in people:
        tr=groups!=held;va=~tr
        inv=reference(covs[tr])
        Xtr=tangent(covs[tr],inv);Xva=tangent(covs[va],inv)
        Ztr,dtr,mid=within_residual(Xtr,y[tr])
        Zva,_,_=within_residual(Xva,y[va])
        Str=(Ztr.T@Ztr)/(len(Ztr)-2);Str=(Str+Str.T)/2
        eig,U=np.linalg.eigh(Str)
        if not np.isfinite(eig).all():
            raise ValueError("Nonfinite source spectrum")
        nu=max(float(np.mean(eig)),np.finfo(float).eps)
        du=U.T@dtr
        V=(Xva-mid)@U
        Zv=Zva@U
        for lam in grid:
            if not feasible[lam]:
                continue
            r=(1-lam)*eig+lam*nu
            if np.any(~np.isfinite(r)) or float(np.min(r))<=64*np.finfo(float).eps*nu:
                feasible[lam]=False
                continue
            invr=1./r
            score=V@(du*invr)
            auc=rank_auc(y[va],score)
            nll=float((np.sum(np.log(r))+np.mean(np.sum((Zv*Zv)*invr,axis=1)))/len(r))
            if not np.isfinite(auc) or not np.isfinite(nll):
                feasible[lam]=False
                continue
            store[lam]["auc"].append(auc)
            store[lam]["nll"].append(nll)
    good=tuple(l for l in grid if feasible[l] and len(store[l]["auc"])==len(people))
    if not good:
        raise ValueError("No feasible scalar candidate")
    crit={}
    for lam in good:
        a=np.asarray(store[lam]["auc"],float)
        c=np.asarray(store[lam]["nll"],float)
        crit[str(lam)]={
          "source_macro_auc":float(np.mean(a)),
          "source_auc_se":float(np.std(a,ddof=1)/np.sqrt(len(a))),
          "covariance_mean_nll":float(np.mean(c)),
          "covariance_nll_se":float(np.std(c,ddof=1)/np.sqrt(len(c))),
          "per_source_auc":list(map(float,a)),
          "per_source_cov_nll":list(map(float,c))}
    ld=max(good,key=lambda l:(crit[str(l)]["source_macro_auc"],l))
    lc=min(good,key=lambda l:(crit[str(l)]["covariance_mean_nll"],l))
    return {
      "lambda_discriminability":float(ld),
      "lambda_covariance_nll":float(lc),
      "feasible_lambdas":list(map(float,good)),
      "infeasible_lambdas":[float(l) for l in grid if l not in good],
      "criteria_by_lambda":crit,
      "source_auc_regret_at_covariance_lambda":float(
          crit[str(ld)]["source_macro_auc"]-crit[str(lc)]["source_macro_auc"]),
      "covariance_nll_regret_at_discriminability_lambda":float(
          crit[str(ld)]["covariance_mean_nll"]-crit[str(lc)]["covariance_mean_nll"])}

def final_analytic_shrinkage(covs,y):
    inv=reference(covs)
    X=tangent(covs,inv)
    Z,_,_=within_residual(X,y)
    lw=LedoitWolf(store_precision=False).fit(Z)
    oa=OAS(store_precision=False).fit(Z)
    return float(lw.shrinkage_),float(oa.shrinkage_),int(X.shape[1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="cho2017_source_data")
    ap.add_argument("--out",default="defaultreg_cho2017_v4_source")
    a=ap.parse_args()
    fixed_split_assertion()
    t0=time.monotonic()
    root=Path(a.data_root).resolve();root.mkdir(parents=True,exist_ok=True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)

    moabb.set_download_provider("upstream")
    dataset=Cho2017()
    raw_files=[]
    for sid in SOURCE:
        paths=flatten_paths(dataset.data_path(sid,path=str(root),force_update=False,verbose=False))
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
            raw_files.append({"subject":sid,"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})

    paradigm=LeftRightImagery(
        filters=([8,30],),
        tmin=0.0,tmax=3.0,baseline=None,
        channels=list(COMMON20),resample=250.0,
        reject_by_annotation=False)
    X,labels,meta=paradigm.get_data(dataset,subjects=list(SOURCE),n_jobs=1)
    X=np.asarray(X,float)
    labels=np.asarray(labels)
    groups=np.asarray(meta["subject"],dtype=int)

    if set(map(int,np.unique(groups)))!=set(SOURCE):
        raise ValueError("Loaded data contains missing/unexpected source participant")
    if set(map(int,np.unique(groups))) & (set(TARGET)|set(RESERVE)):
        raise ValueError("Target/reserve data touched in SOURCE-ONLY workflow")
    if X.ndim!=3 or X.shape[1]!=len(COMMON20):
        raise ValueError(f"Unexpected source array shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite source EEG")
    if X.shape[0]!=200*len(SOURCE):
        raise ValueError(f"Expected 200 left/right trials per Cho subject, got {X.shape[0]} total")

    names=tuple(map(str,labels))
    allowed={"left_hand","right_hand"}
    if set(names)!=allowed:
        raise ValueError(f"Unexpected class labels {set(names)}")
    y=np.asarray([1 if z=="right_hand" else 0 for z in names],dtype=np.int64)

    per_subject={}
    for sid in SOURCE:
        m=groups==sid
        per_subject[str(sid)]={
          "n_trials":int(np.sum(m)),
          "left_hand":int(np.sum(y[m]==0)),
          "right_hand":int(np.sum(y[m]==1))}
        if per_subject[str(sid)]["left_hand"]!=100 or per_subject[str(sid)]["right_hand"]!=100:
            raise ValueError(f"Unexpected class denominator for source {sid}: {per_subject[str(sid)]}")

    X=X-X.mean(axis=-1,keepdims=True)
    covs=X@X.transpose(0,2,1)/X.shape[-1]
    scale=np.trace(covs,axis1=1,axis2=2)/len(COMMON20)
    covs+=np.eye(len(COMMON20))[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)

    primary=source_path(covs,y,groups,FULL_GRID)
    dense=source_path(covs,y,groups,DENSE_GRID)
    lw,oa,p=final_analytic_shrinkage(covs,y)
    if p!=210:
        raise ValueError(f"Expected tangent dimension 210, got {p}")

    meta_cols={}
    for col in ("subject","session","run"):
        if col in meta.columns:
            meta_cols[col]=[str(v) for v in meta[col].tolist()]

    np.savez_compressed(out/"source_covariances.npz",
        covs=covs,y=y,subject=groups,
        label=np.asarray(names,dtype="U16"),
        **{k:np.asarray(v,dtype="U32") for k,v in meta_cols.items() if k!="subject"})
    cov_sha=sha(out/"source_covariances.npz")

    result={
      "status":"CHO2017_V4_SOURCE_ONLY_FROZEN_NO_TARGET_DATA",
      "protocol_version":"post-S021-v4",
      "dataset":"Cho2017",
      "data_doi":"10.5524/100295",
      "paper_doi":"10.1093/gigascience/gix034",
      "moabb_version":moabb.__version__,
      "download_provider":"upstream",
      "split_seed":SEED,
      "source_participants":list(SOURCE),
      "confirmatory_target_ids_metadata_only":list(TARGET),
      "reserve_ids_metadata_only":list(RESERVE),
      "target_data_loaded":False,
      "target_information_track":"SOURCE_ONLY_PRE_TARGET_FREEZE",
      "channels":list(COMMON20),
      "n_channels":len(COMMON20),
      "tangent_dimension":p,
      "band_hz":[8,30],
      "task_relative_window_s":[0.0,3.0],
      "resample_hz":250.0,
      "reject_by_annotation":False,
      "source_trials":int(len(y)),
      "per_source_subject":per_subject,
      "raw_source_files":raw_files,
      "source_covariance_npz_sha256":cov_sha,
      "primary_full_grid":list(FULL_GRID),
      "dense_frontier_grid":list(DENSE_GRID),
      "primary_source_path":primary,
      "dense_source_path":dense,
      "final_source_analytic_shrinkage":{
        "LedoitWolf":lw,"OAS":oa},
      "wall_seconds":round(time.monotonic()-t0,2)}
    path=out/"source_freeze.json"
    path.write_text(json.dumps(result,indent=2,sort_keys=True))
    result["source_freeze_json_sha256"]=sha(path)
    path.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
