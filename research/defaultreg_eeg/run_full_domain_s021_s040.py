"""Prospective full-domain Default Regularisation confirmation on EEGMMIDB S021-S040.

Frozen design: S001-S008 source; S021-S040 untouched targets; R04/R08/R12;
21 channels; 8-30 Hz; all original T1/T2 cues. Primary method is not a new
proxy objective: it is full-domain source-participant macro-AUC scalar
shrinkage. Main scientific comparison is objective mismatch against analytic
LW/OAS and source covariance-predictive-NLL optimum.

No target statistic or target label enters selection. Target labels are used
only after score vectors are frozen.
"""
from __future__ import annotations
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run_official_edf_baselines import (
    load_or_fetch,extract_original_trials,reference,tangent,
    within_residual,fit_analytic,sha)

SOURCE=tuple(range(1,9))
TARGET=tuple(range(21,41))
RUNS=(4,8,12)
FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)

def rank_auc(y,s):
    y=np.asarray(y);s=np.asarray(s,dtype=float)
    n1=int(np.sum(y==1));n0=int(np.sum(y==0))
    if n1==0 or n0==0:raise ValueError("AUC requires both classes")
    order=np.argsort(s,kind="mergesort");sv=s[order]
    rank=np.empty(len(s),dtype=float);i=0
    while i<len(s):
        j=i+1
        while j<len(s) and sv[j]==sv[i]:j+=1
        rank[order[i:j]]=(i+j+1)/2
        i=j
    return float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0))

def source_full_path(covs,y,groups,tail_fraction=.25):
    """Exact spectral lambda path with source-participant grouped folds."""
    people=sorted(set(map(int,groups)))
    if people!=list(SOURCE):raise ValueError("Unexpected source group identity")
    store={lam:{"auc":[],"signed":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID}
    t0=time.perf_counter()
    for held in people:
        tr=groups!=held;va=~tr
        inv=reference(covs[tr])
        Xtr=tangent(covs[tr],inv);Xva=tangent(covs[va],inv)
        resid_tr,dtr,mid=within_residual(Xtr,y[tr])
        resid_va,dva,_=within_residual(Xva,y[va])
        Str=(resid_tr.T@resid_tr)/(len(Xtr)-2)
        Str=(Str+Str.T)/2
        Sva=(resid_va.T@resid_va)/(len(Xva)-2)
        Sva=(Sva+Sva.T)/2
        eig,U=np.linalg.eigh(Str)
        if not np.isfinite(eig).all():raise ValueError("Nonfinite source spectrum")
        nu=max(float(np.mean(eig)),np.finfo(float).eps)
        dtr_u=U.T@dtr;dva_u=U.T@dva
        Sva_u=U.T@Sva@U
        V=(Xva-mid)@U
        Zv=resid_va@U
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            r=(1-lam)*eig+lam*nu
            if np.any(~np.isfinite(r)) or float(np.min(r))<=64*np.finfo(float).eps*nu:
                feasible[lam]=False;continue
            invr=1.0/r
            wu=dtr_u*invr
            score=V@wu
            den=float(wu@Sva_u@wu)
            if not np.isfinite(den) or den<=np.finfo(float).eps*max(1.,float(wu@wu)):
                feasible[lam]=False;continue
            auc=rank_auc(y[va],score)
            signed=float(wu@dva_u)/np.sqrt(den)
            nll=float((np.sum(np.log(r))+np.mean(np.sum((Zv*Zv)*invr,axis=1)))/len(r))
            if not all(map(np.isfinite,(auc,signed,nll))):
                feasible[lam]=False;continue
            store[lam]["auc"].append(auc)
            store[lam]["signed"].append(signed)
            store[lam]["nll"].append(nll)
    good=[lam for lam in FULL_GRID if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("No feasible lambda on all source folds")
    k=max(1,int(math.ceil(tail_fraction*len(people))))
    crit={}
    for lam in good:
        a=np.asarray(store[lam]["auc"]);m=np.asarray(store[lam]["signed"]);c=np.asarray(store[lam]["nll"])
        crit[lam]={
          "source_macro_auc":float(np.mean(a)),
          "source_auc_se":float(np.std(a,ddof=1)/np.sqrt(len(a))),
          "signed_mean":float(np.mean(m)),
          "squared_margin":float(np.mean(m*m)),
          "tail_signed":float(np.mean(np.sort(m)[:k])),
          "negative_margin_fraction":float(np.mean(m<0)),
          "covariance_mean_nll":float(np.mean(c)),
          "covariance_nll_se":float(np.std(c,ddof=1)/np.sqrt(len(c))),
          "per_source_auc":list(map(float,a)),
          "per_source_signed_margin":list(map(float,m)),
          "per_source_cov_nll":list(map(float,c))}
    lambda_d=max(good,key=lambda x:(crit[x]["source_macro_auc"],x))
    lambda_c=min(good,key=lambda x:(crit[x]["covariance_mean_nll"],x))
    best_auc=crit[lambda_d]["source_macro_auc"];se_auc=crit[lambda_d]["source_auc_se"]
    best_nll=crit[lambda_c]["covariance_mean_nll"];se_nll=crit[lambda_c]["covariance_nll_se"]
    D1=tuple(l for l in good if crit[l]["source_macro_auc"]>=best_auc-se_auc-1e-15)
    C1=tuple(l for l in good if crit[l]["covariance_mean_nll"]<=best_nll+se_nll+1e-15)
    inter=tuple(sorted(set(D1).intersection(C1)))
    union=set(D1).union(C1)
    result={
      "lambda_discriminability":float(lambda_d),
      "lambda_covariance_nll":float(lambda_c),
      "discriminability_1se_lambdas":list(map(float,D1)),
      "covariance_1se_lambdas":list(map(float,C1)),
      "one_se_intersection":list(map(float,inter)),
      "one_se_jaccard":float(len(inter)/len(union)) if union else 1.0,
      "source_auc_regret_at_covariance_lambda":float(
          crit[lambda_d]["source_macro_auc"]-crit[lambda_c]["source_macro_auc"]),
      "covariance_nll_regret_at_discriminability_lambda":float(
          crit[lambda_d]["covariance_mean_nll"]-crit[lambda_c]["covariance_mean_nll"]),
      "lambda_ratio_disc_over_cov":float(lambda_d/lambda_c) if lambda_c>0 else None,
      "discriminability_domain_boundary_hit":bool(lambda_d in (0.0,1.0)),
      "covariance_domain_boundary_hit":bool(lambda_c in (0.0,1.0)),
      "feasible_lambdas":list(map(float,good)),
      "infeasible_lambdas":[float(l) for l in FULL_GRID if l not in good],
      "criteria_by_lambda":{str(l):crit[l] for l in good},
      "eigendecompositions":len(people),
      "candidate_fold_evaluations":len(people)*len(FULL_GRID),
      "path_wall_seconds":float(time.perf_counter()-t0)}
    return result

def fixed_lambda_scores(Xtrain,ytrain,Xtest,lam):
    resid,d,mid=within_residual(Xtrain,ytrain)
    S=(resid.T@resid)/(len(Xtrain)-2);S=(S+S.T)/2
    eig,U=np.linalg.eigh(S);nu=max(float(np.mean(eig)),np.finfo(float).eps)
    r=(1-lam)*eig+lam*nu
    if np.any(r<=64*np.finfo(float).eps*nu):raise ValueError("Selected lambda infeasible")
    w=U@((U.T@d)/r)
    return (Xtest-mid)@w

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--edf-root",default="edf_original")
    ap.add_argument("--out",default="defaultreg_full_domain_s021_s040")
    a=ap.parse_args()
    root=Path(a.edf_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    covs=[];ys=[];groups=[];ids=[];files=[]
    for subject in SOURCE+TARGET:
        for run in RUNS:
            file=load_or_fetch(root,subject,run)
            c,y,trialids=extract_original_trials(file,subject,run)
            files.append({"subject":subject,"run":run,"name":file.name,
                          "sha256":sha(file),"edf_bytes":file.stat().st_size,
                          "all_original_cues":len(trialids),
                          "selected_and_scored":len(y)})
            covs.append(c);ys.append(y);groups.extend([subject]*len(y));ids.extend(trialids)
    C=np.concatenate(covs);y=np.concatenate(ys);g=np.asarray(groups)
    src=np.isin(g,SOURCE);tgt=np.isin(g,TARGET)
    if np.intersect1d(np.unique(g[src]),np.unique(g[tgt])).size:
        raise ValueError("Source/target subject overlap")
    if not all(f["all_original_cues"]==f["selected_and_scored"] for f in files):
        raise ValueError("Original task cue denominator drift")

    path=source_full_path(C[src],y[src],g[src])
    inv=reference(C[src])
    Xs=tangent(C[src],inv);Xt=tangent(C[tgt],inv)
    scores={}
    lw,lwlam=fit_analytic(Xs,y[src],Xt,LedoitWolf(store_precision=False))
    oa,oalam=fit_analytic(Xs,y[src],Xt,OAS(store_precision=False))
    scores["LedoitWolf_LDA"]=lw;scores["OAS_LDA"]=oa
    scores["source_macro_full_domain_LDA"]=fixed_lambda_scores(
        Xs,y[src],Xt,path["lambda_discriminability"])
    scores["source_cov_nll_full_domain_LDA"]=fixed_lambda_scores(
        Xs,y[src],Xt,path["lambda_covariance_nll"])
    lr=LogisticRegression(C=1.0,solver="lbfgs",max_iter=3000)
    lr.fit(Xs,y[src]);scores["LogisticRegression"]=lr.decision_function(Xt)

    gt=g[tgt];yt=y[tgt]
    result={
      "status":"REAL_EEGMMIDB_S021_S040_FULL_DOMAIN_PROSPECTIVE",
      "official_dataset":"PhysioNet EEGMMIDB 1.0.0 DOI 10.13026/C28G6P",
      "license":"Open Data Commons Attribution v1.0",
      "source_participants":list(SOURCE),"heldout_participants":list(TARGET),
      "runs":list(RUNS),
      "features":"8-30Hz, 21 channels, source-trained arithmetic-SPD-whitened log tangent, 231D",
      "target_information_track":"Z0",
      "target_data_used_for_training_or_hyperparam_selection":False,
      "target_batch_statistics_used":False,
      "no_task_cues_dropped":True,
      "full_domain_grid":list(FULL_GRID),
      "source_path":path,
      "LedoitWolf_shrinkage":float(lwlam),"OAS_shrinkage":float(oalam),
      "train_original_task_trials":int(src.sum()),
      "target_original_task_trials":int(tgt.sum()),
      "per_target_subject":{},
      "source_files":files}
    for sid in TARGET:
        m=gt==sid
        result["per_target_subject"][str(sid)]={
          "target_trials":int(m.sum()),
          "task_label_counts":{"T1":int(np.sum(yt[m]==0)),"T2":int(np.sum(yt[m]==1))},
          "auc":{name:round(float(roc_auc_score(yt[m],score[m])),8)
                 for name,score in scores.items()}}
    np.savez_compressed(out/"all_original_trial_covariances.npz",
                        covs=C,y=y,subject=g,
                        run=np.asarray([x["run"] for x in ids]),
                        event_sample=np.asarray([x["event_sample"] for x in ids]),
                        cue=np.asarray([x["cue"] for x in ids]))
    result["cov_npz_sha256"]=sha(out/"all_original_trial_covariances.npz")
    result["wall_seconds"]=round(time.monotonic()-t0,2)
    (out/"full_domain_result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))
    return result

if __name__=="__main__":main()
