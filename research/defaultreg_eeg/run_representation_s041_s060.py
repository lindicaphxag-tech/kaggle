"""Prospective S041-S060 representation x regularisation experiment.

Frozen before target outcomes:
  source S001-S008 only; targets S041-S060; EEGMMIDB R04/R08/R12.
  Representations: WIDE231, FB8_1848, FB8_SOURCE_PCA231,
  WIDE_NUISANCE1848.
  Regularisers: analytic LedoitWolf, OAS, full-domain source-participant
  macro-AUC shrinkage, full-domain source covariance-predictive-NLL shrinkage.

The source-LOPO path RE-FITS every unsupervised representation component
(tangent reference, PCA, nuisance scale) without the held source participant.
No target feature, label, batch moment, PCA component or quality statistic
enters selection. Target labels are read only after score vectors are frozen.
"""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import mne
from sklearn.covariance import LedoitWolf,OAS
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from run_official_edf_baselines import (
    CHANS,load_or_fetch,sha,reference,tangent,within_residual)

SOURCE=tuple(range(1,9))
TARGET=tuple(range(41,61))
RUNS=(4,8,12)
WIDE=(8.,30.)
FB8=((8.,12.),(12.,16.),(16.,20.),(20.,24.),
     (24.,28.),(28.,32.),(32.,36.),(36.,40.))
BANDS=(WIDE,)+FB8
FULL_GRID=(
  0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
  0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
  0.90,0.95,0.98,0.99,1.0)
REPRESENTATIONS=("WIDE231","FB8_1848","FB8_SOURCE_PCA231","WIDE_NUISANCE1848")
METHODS=("LedoitWolf","OAS","source_macro_auc","source_cov_nll")
SEED=20261010

def extract_all_bands(path,subject,run):
    raw=mne.io.read_raw_edf(str(path),preload=True,verbose="ERROR")
    names={x:x.replace(".","").replace("-","").upper() for x in raw.ch_names}
    if len(set(names.values()))!=len(names):raise ValueError("Duplicate normalized channels")
    raw.rename_channels(names)
    if not set(CHANS).issubset(raw.ch_names):
        raise ValueError("Missing required 21 channels in "+str(path))
    raw.pick(list(CHANS))
    events,_=mne.events_from_annotations(raw,event_id={"T1":1,"T2":2},verbose=False)
    if not len(events):raise ValueError("No T1/T2 task events")
    cov_by_band=[]
    for lo,hi in BANDS:
        x=raw.copy()
        x.filter(lo,hi,method="iir",verbose="ERROR")
        ep=mne.Epochs(x,events,event_id={"T1":1,"T2":2},
                      tmin=.5,tmax=3.5,baseline=None,
                      reject_by_annotation=False,proj=False,
                      preload=True,verbose="ERROR")
        if not np.array_equal(ep.events,events):
            raise ValueError(f"Lost original task cue in band {lo}-{hi}")
        data=ep.get_data(copy=True)
        if not np.isfinite(data).all():raise ValueError("Nonfinite EEG epoch")
        data=data-data.mean(axis=-1,keepdims=True)
        c=data@data.transpose(0,2,1)/data.shape[-1]
        scale=np.trace(c,axis1=1,axis2=2)/len(CHANS)
        c+=np.eye(len(CHANS))[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
        cov_by_band.append(c)
    y=(events[:,2]==2).astype(np.int64)
    ids=[{"subject":subject,"run":run,"event_sample":int(e[0]),"cue":int(e[2])}
         for e in events]
    return cov_by_band,y,ids

def base_tangent(covs_by_band,train_mask,apply_mask,rep):
    """Fit geometry on train_mask only and transform train/apply rows."""
    if rep.startswith("WIDE"):
        inv=reference(covs_by_band[0][train_mask])
        return tangent(covs_by_band[0][train_mask],inv),tangent(covs_by_band[0][apply_mask],inv)
    parts_train=[];parts_apply=[]
    for b in range(1,9):
        inv=reference(covs_by_band[b][train_mask])
        parts_train.append(tangent(covs_by_band[b][train_mask],inv))
        parts_apply.append(tangent(covs_by_band[b][apply_mask],inv))
    return np.concatenate(parts_train,axis=1),np.concatenate(parts_apply,axis=1)

def transform_representation(covs_by_band,train_mask,apply_mask,rep,*,seed):
    Xtr,Xap=base_tangent(covs_by_band,train_mask,apply_mask,rep)
    if rep=="FB8_SOURCE_PCA231":
        if Xtr.shape[0] < 231:raise ValueError("Insufficient source trials for PCA231")
        pca=PCA(n_components=231,svd_solver="full")
        Xtr=pca.fit_transform(Xtr)
        Xap=pca.transform(Xap)
        return Xtr,Xap,{"pca_source_only":True,
                         "pca_explained_variance_ratio_sum":float(np.sum(pca.explained_variance_ratio_))}
    if rep=="WIDE_NUISANCE1848":
        extra=1848-Xtr.shape[1]
        if extra<=0:raise ValueError("Nuisance control requires positive extra dimensions")
        sd=np.std(Xtr,axis=0,ddof=1)
        positive=sd[np.isfinite(sd)&(sd>np.finfo(float).eps)]
        scale=float(np.median(positive)) if len(positive) else 1.0
        # Draws are deterministic and label-independent. Separate streams make
        # apply values unavailable to source fitting by construction.
        rng_tr=np.random.default_rng(seed)
        rng_ap=np.random.default_rng(seed ^ 0x5EED5EED)
        Xtr=np.concatenate((Xtr,rng_tr.normal(0,scale,size=(len(Xtr),extra))),axis=1)
        Xap=np.concatenate((Xap,rng_ap.normal(0,scale,size=(len(Xap),extra))),axis=1)
        return Xtr,Xap,{"nuisance_extra_dims":extra,
                         "nuisance_scale_source_only":scale,
                         "labels_used_to_generate_nuisance":False}
    return Xtr,Xap,{}

def _svd_state(X,y):
    resid,d,mid=within_residual(X,y)
    denom=len(X)-2
    _,s,Vt=np.linalg.svd(resid,full_matrices=False)
    eig=(s*s)/denom
    nu=max(float(np.sum(eig)/X.shape[1]),np.finfo(float).eps)
    tol=64*np.finfo(float).eps*max(float(eig[0]) if len(eig) else 0.,nu,1.)
    rank=int(np.sum(eig>tol))
    eig=eig[:rank]
    V=Vt[:rank].T
    return d,mid,eig,V,nu

def ridge_scores_from_state(state,X,lam):
    d,mid,eig,V,nu=state
    p=len(d);rank=len(eig)
    if lam==0.0 and rank<p:raise np.linalg.LinAlgError("lambda=0 singular")
    coef=V.T@d
    if lam==0.0:
        w=V@(coef/eig)
    else:
        base=lam*nu
        r=(1-lam)*eig+base
        invbase=1.0/base
        w=d*invbase + V@((1.0/r-invbase)*coef)
    return (X-mid)@w

def held_cov_nll(state,resid_valid,lam):
    d,mid,eig,V,nu=state
    p=len(d);rank=len(eig)
    if lam==0.0 and rank<p:raise np.linalg.LinAlgError("lambda=0 singular")
    proj=resid_valid@V
    norm2=np.sum(resid_valid*resid_valid,axis=1)
    if lam==0.0:
        logdet=float(np.sum(np.log(eig)))
        quad=np.sum((proj*proj)/eig,axis=1)
    else:
        base=lam*nu
        r=(1-lam)*eig+base
        logdet=float(np.sum(np.log(r))+(p-rank)*np.log(base))
        invbase=1.0/base
        quad=norm2*invbase+np.sum((proj*proj)*(1.0/r-invbase),axis=1)
    return float((logdet+np.mean(quad))/p)

def source_full_path(covs_by_band,y,groups,rep):
    """Grouped source-only path; every representation fit excludes held source."""
    people=sorted(set(map(int,groups)))
    if people!=list(SOURCE):raise ValueError("Unexpected source identities")
    store={lam:{"auc":[],"nll":[]} for lam in FULL_GRID}
    feasible={lam:True for lam in FULL_GRID}
    fold_meta=[]
    t0=time.perf_counter()
    all_rows=np.ones(len(y),dtype=bool)
    for fold,held in enumerate(people):
        tr=groups!=held;va=~tr
        Xtr,Xva,meta=transform_representation(
            covs_by_band,tr,va,rep,seed=SEED+1009*fold)
        state=_svd_state(Xtr,y[tr])
        resid_va,_,_=within_residual(Xva,y[va])
        fold_meta.append({"held_source_subject":held,
                          "train_rows":int(tr.sum()),"valid_rows":int(va.sum()),
                          "feature_dimension":int(Xtr.shape[1]),**meta})
        for lam in FULL_GRID:
            if not feasible[lam]:continue
            try:
                score=ridge_scores_from_state(state,Xva,lam)
                auc=float(roc_auc_score(y[va],score))
                nll=held_cov_nll(state,resid_va,lam)
            except (np.linalg.LinAlgError,FloatingPointError,ValueError):
                feasible[lam]=False;continue
            if not np.isfinite(auc) or not np.isfinite(nll):
                feasible[lam]=False;continue
            store[lam]["auc"].append(auc);store[lam]["nll"].append(nll)
    good=[lam for lam in FULL_GRID
          if feasible[lam] and len(store[lam]["auc"])==len(people)]
    if not good:raise ValueError("No full-domain lambda feasible on all folds")
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
    return {
      "lambda_discriminability":float(ld),"lambda_covariance_nll":float(lc),
      "feasible_lambdas":list(map(float,good)),
      "infeasible_lambdas":[float(x) for x in FULL_GRID if x not in good],
      "criteria_by_lambda":{str(k):v for k,v in crit.items()},
      "discriminability_domain_boundary_hit":ld in (0.,1.),
      "covariance_domain_boundary_hit":lc in (0.,1.),
      "fold_representation_meta":fold_meta,
      "candidate_fold_evaluations":len(people)*len(FULL_GRID),
      "path_wall_seconds":float(time.perf_counter()-t0)}

def analytic_lambda(X,y,kind):
    resid,_,_=within_residual(X,y)
    if kind=="LedoitWolf":
        return float(LedoitWolf(store_precision=False,block_size=500).fit(resid).shrinkage_)
    if kind=="OAS":
        return float(OAS(store_precision=False).fit(resid).shrinkage_)
    raise ValueError(kind)

def exact_signflip_p(d):
    d=np.asarray(d,dtype=float);n=len(d);obs=abs(float(np.mean(d)))
    total=1<<n;exceed=0;tol=64*np.finfo(float).eps*max(1.,obs)
    # Chunked exact enumeration to keep memory bounded at n=20.
    chunk=1<<15
    bit=np.arange(n,dtype=np.uint64)
    for start in range(0,total,chunk):
        ids=np.arange(start,min(start+chunk,total),dtype=np.uint64)[:,None]
        signs=1.0-2.0*((ids>>bit)&1).astype(float)
        stats=np.abs(np.mean(signs*d[None,:],axis=1))
        exceed+=int(np.sum(stats>=obs-tol))
    return float(exceed/total),total

def bootstrap_ci(d,seed,reps=20000):
    d=np.asarray(d,dtype=float);rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(reps,len(d)))
    q=np.quantile(np.mean(d[idx],axis=1),[.025,.975])
    return [float(q[0]),float(q[1])]

def holm(ps):
    ps=np.asarray(ps,dtype=float);order=np.argsort(ps);m=len(ps)
    out=np.empty(m);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(m-rank)*ps[i]));out[i]=running
    return list(map(float,out))

def contrast_summary(values,name):
    d=np.asarray(values,dtype=float)
    p,draws=exact_signflip_p(d)
    return {"contrast":name,"n_participants":len(d),
            "mean":float(np.mean(d)),"median":float(np.median(d)),
            "q25":float(np.quantile(d,.25)),"min":float(np.min(d)),
            "max":float(np.max(d)),"improved":int(np.sum(d>0)),
            "harmed":int(np.sum(d<0)),"harmed_gt_0_02":int(np.sum(d<-.02)),
            "bootstrap_ci95":bootstrap_ci(d,SEED+len(name)),
            "exact_signflip_p_two_sided":p,"exact_signflip_draws":draws}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--edf-root",default="edf_original")
    ap.add_argument("--out",default="defaultreg_representation_s041_s060")
    a=ap.parse_args();root=Path(a.edf_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic()
    band_lists=[[] for _ in BANDS];ys=[];groups=[];ids=[];files=[]
    for subject in SOURCE+TARGET:
        for run in RUNS:
            p=load_or_fetch(root,subject,run)
            bands,y,trialids=extract_all_bands(p,subject,run)
            for b,c in enumerate(bands):band_lists[b].append(c)
            ys.append(y);groups.extend([subject]*len(y));ids.extend(trialids)
            files.append({"subject":subject,"run":run,"name":p.name,
                          "sha256":sha(p),"edf_bytes":p.stat().st_size,
                          "all_original_cues":len(trialids),
                          "selected_and_scored_every_band":len(y)})
    C=[np.concatenate(x) for x in band_lists]
    y=np.concatenate(ys);g=np.asarray(groups)
    src=np.isin(g,SOURCE);tgt=np.isin(g,TARGET)
    if np.intersect1d(np.unique(g[src]),np.unique(g[tgt])).size:
        raise ValueError("Source/target participant overlap")
    paths={};target_scores={};rep_meta={}
    for ri,rep in enumerate(REPRESENTATIONS):
        paths[rep]=source_full_path([c[src] for c in C],y[src],g[src],rep)
        Xs,Xt,meta=transform_representation(
            C,src,tgt,rep,seed=SEED+50000+ri)
        rep_meta[rep]={"feature_dimension":int(Xs.shape[1]),**meta}
        l_lw=analytic_lambda(Xs,y[src],"LedoitWolf")
        l_oa=analytic_lambda(Xs,y[src],"OAS")
        lambdas={
          "LedoitWolf":l_lw,"OAS":l_oa,
          "source_macro_auc":paths[rep]["lambda_discriminability"],
          "source_cov_nll":paths[rep]["lambda_covariance_nll"]}
        rep_meta[rep]["final_lambda_by_method"]={k:float(v) for k,v in lambdas.items()}
        target_scores[rep]={}
        state=_svd_state(Xs,y[src])
        for method,lam in lambdas.items():
            target_scores[rep][method]=ridge_scores_from_state(state,Xt,float(lam))
    gt=g[tgt];yt=y[tgt]
    per_subject={}
    for sid in TARGET:
        m=gt==sid
        per_subject[str(sid)]={"target_trials":int(m.sum()),"auc":{}}
        for rep in REPRESENTATIONS:
            per_subject[str(sid)]["auc"][rep]={
              method:float(roc_auc_score(yt[m],target_scores[rep][method][m]))
              for method in METHODS}
    def arr(rep,method):
        return np.asarray([per_subject[str(s)]["auc"][rep][method] for s in TARGET])
    c1=(arr("FB8_1848","source_macro_auc")-arr("WIDE231","source_macro_auc"))-(
        arr("FB8_1848","LedoitWolf")-arr("WIDE231","LedoitWolf"))
    c2=arr("FB8_1848","source_macro_auc")-arr("WIDE_NUISANCE1848","source_macro_auc")
    c3=arr("FB8_SOURCE_PCA231","source_macro_auc")-arr("WIDE231","source_macro_auc")
    c4=(arr("FB8_1848","source_macro_auc")-arr("WIDE231","source_macro_auc"))-(
        arr("FB8_1848","source_cov_nll")-arr("WIDE231","source_cov_nll"))
    summaries=[
      contrast_summary(c1,"C1_FB8xRegularisation_LW_PRIMARY"),
      contrast_summary(c2,"C2_FB8_minus_WIDE_NUISANCE1848"),
      contrast_summary(c3,"C3_FB8_PCA231_minus_WIDE231"),
      contrast_summary(c4,"C4_FB8xRegularisation_covNLL")]
    adj=holm([x["exact_signflip_p_two_sided"] for x in summaries])
    for x,padj in zip(summaries,adj):x["holm_p_C1_C4"]=padj
    result={
      "status":"REAL_EEGMMIDB_S041_S060_REPRESENTATION_REGULARISATION_PROSPECTIVE",
      "official_dataset":"PhysioNet EEGMMIDB 1.0.0 DOI 10.13026/C28G6P",
      "source_participants":list(SOURCE),"heldout_participants":list(TARGET),
      "runs":list(RUNS),"target_information_track":"Z0",
      "target_data_used_for_selection":False,"target_batch_statistics_used":False,
      "all_original_task_cues_retained_in_every_band":True,
      "full_domain_grid":list(FULL_GRID),
      "representations":rep_meta,"source_paths":paths,
      "per_target_subject":per_subject,"predeclared_contrasts":summaries,
      "files":files,"wall_seconds":round(time.monotonic()-t0,2),
      "interpretation":"Prospective participant-level mechanism experiment; no cross-paper SOTA inference."}
    (out/"representation_regularisation_result.json").write_text(
        json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps(result,indent=2,sort_keys=True))
    return result

if __name__=="__main__":main()
