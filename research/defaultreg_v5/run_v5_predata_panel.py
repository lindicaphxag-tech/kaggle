"""Default Regularisation v5 true PRE-DATA external panel.

Datasets:
- BNCI2014_002
- BNCI2015_001

The workflow uses a two-phase target protocol:
A) source-only fit/selection + target score freeze + score-bundle hash;
B) only then target AUC / inference.

No target performance can choose a lambda or method.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,time
from pathlib import Path
import numpy as np
import moabb,sklearn,pyriemann,mne
from moabb.datasets import BNCI2014_002,BNCI2015_001
from moabb.paradigms import MotorImagery
from sklearn.covariance import LedoitWolf,OAS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pyriemann.classification import MDM

SEED=20261010
FULL_GRID=(
 0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,
 0.05,0.075,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,
 0.90,0.95,0.98,0.99,1.0)
NEAR_ZERO=(0.0,0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02,0.03,0.05)
DENSE_GRID=tuple(sorted(set(NEAR_ZERO+tuple(round(i*.0025,10) for i in range(401)))))
METHODS=(
 "LedoitWolf_LDA","OAS_LDA","source_cov_nll_full_domain_LDA",
 "source_macro_full_domain_LDA","discriminability_first_LDA",
 "mean_tail_fidelity_LDA","TS_LogisticRegression","Riemannian_MDM")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def flatten(x):
    if isinstance(x,(str,Path)):return [Path(x)]
    out=[]
    for z in x:out.extend(flatten(z))
    return out

def build_dataset(name):
    if name=="BNCI2014_002":
        ds=BNCI2014_002()
        return ds,tuple(range(1,15)),5.0
    if name=="BNCI2015_001":
        ds=BNCI2015_001()
        return ds,tuple(range(1,13)),5.0
    raise ValueError(name)

def native_channels(ds):
    ch=tuple(ds.METADATA.acquisition.sensors)
    expected={"BNCI2014-002":15,"BNCI2015-001":13}[ds.code]
    if len(ch)!=expected or len(set(ch))!=expected:
        raise ValueError(f"Unexpected native montage for {ds.code}: {ch}")
    return ch

def covariances(X,nch):
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=3 or X.shape[1]!=nch or not np.isfinite(X).all():
        raise ValueError(f"Expected finite [trial,{nch},time], got {X.shape}")
    X=X-X.mean(axis=-1,keepdims=True)
    C=X@X.transpose(0,2,1)/X.shape[-1]
    scale=np.trace(C,axis1=1,axis2=2)/nch
    C+=np.eye(nch)[None,:,:]*np.maximum(scale[:,None,None]*1e-6,1e-20)
    return (C+C.transpose(0,2,1))/2

def reference(C):
    M=np.mean(C,axis=0);e,U=np.linalg.eigh((M+M.T)/2)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("reference not SPD")
    return (U*(e**-.5))@U.T

def tangent(C,inv):
    Z=np.einsum("ab,nbc,cd->nad",inv,C,inv)
    Z=(Z+Z.transpose(0,2,1))/2
    e,U=np.linalg.eigh(Z)
    if np.any(~np.isfinite(e)) or np.any(e<=0):raise ValueError("trial not SPD")
    L=(U*np.log(e)[:,None,:])@U.transpose(0,2,1)
    i,j=np.triu_indices(L.shape[1]);out=L[:,i,j]
    out[:,i!=j]*=np.sqrt(2.)
    return out

def moments(X,y):
    a=X[y==0];b=X[y==1]
    if len(a)<2 or len(b)<2:raise ValueError("both classes need >=2")
    m0=a.mean(0);m1=b.mean(0)
    R=np.concatenate((a-m0,b-m1))
    S=(R.T@R)/(len(R)-2)
    return (S+S.T)/2,m1-m0,(m0+m1)/2,R

def spectral_state(X,y):
    S,d,mid,R=moments(X,y);e,U=np.linalg.eigh(S)
    nu=max(float(np.mean(e)),np.finfo(float).eps)
    return e,U,d,mid,nu

def ridge_scores(state,X,lam):
    e,U,d,mid,nu=state;r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):raise np.linalg.LinAlgError("infeasible lambda")
    return (X-mid)@(U@((U.T@d)/r))

def predictive_nll(state,R,lam):
    e,U,d,mid,nu=state;r=(1-lam)*e+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(np.abs(e))))
    if np.any(~np.isfinite(r)) or np.any(r<=tol):raise np.linalg.LinAlgError("infeasible lambda")
    Z=R@U
    return float((np.sum(np.log(r))+np.mean(np.sum((Z*Z)/r,axis=1)))/len(r))

def analytic_scores(Xtr,ytr,Xte,kind):
    _,d,mid,R=moments(Xtr,ytr)
    est=(LedoitWolf(store_precision=False) if kind=="LW" else OAS(store_precision=False)).fit(R)
    C=(est.covariance_+est.covariance_.T)/2
    C+=np.eye(C.shape[0])*max(float(np.trace(C)),1.)*1e-12
    return (Xte-mid)@np.linalg.solve(C,d),float(est.shrinkage_)

def rank_auc(y,s):
    return float(roc_auc_score(np.asarray(y),np.asarray(s)))

def tail_cvar(x,f=.25):
    a=np.sort(np.asarray(x,float));k=max(1,int(math.ceil(f*len(a))))
    return float(np.mean(a[:k]))

def source_dense_path(C,y,g):
    people=tuple(sorted(set(map(int,g))))
    store={l:{"auc":[],"nll":[]} for l in DENSE_GRID}
    feasible={l:True for l in DENSE_GRID}
    receipts=[]
    for held in people:
        tr=g!=held;va=~tr
        inv=reference(C[tr]);Xtr=tangent(C[tr],inv);Xv=tangent(C[va],inv)
        state=spectral_state(Xtr,y[tr]);_,_,_,Rv=moments(Xv,y[va])
        receipts.append({"held_source":int(held),"train_subjects":[int(x) for x in people if x!=held],
                         "train_trials":int(tr.sum()),"validation_trials":int(va.sum())})
        for lam in DENSE_GRID:
            if not feasible[lam]:continue
            try:
                auc=rank_auc(y[va],ridge_scores(state,Xv,lam))
                nll=predictive_nll(state,Rv,lam)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                feasible[lam]=False;continue
            if not np.isfinite(auc) or not np.isfinite(nll):
                feasible[lam]=False;continue
            store[lam]["auc"].append(auc);store[lam]["nll"].append(nll)
    good=[l for l in DENSE_GRID if feasible[l] and len(store[l]["auc"])==len(people)]
    if not good:raise ValueError("no dense candidate feasible across source LOPO")
    crit={}
    for l in good:
        a=np.asarray(store[l]["auc"]);n=np.asarray(store[l]["nll"])
        crit[str(l)]={
          "source_macro_auc":float(a.mean()),
          "source_auc_se":float(a.std(ddof=1)/np.sqrt(len(a))),
          "source_tail_cvar25":tail_cvar(a,.25),
          "covariance_mean_nll":float(n.mean()),
          "covariance_nll_se":float(n.std(ddof=1)/np.sqrt(len(n))),
          "covariance_nll_sd":float(n.std(ddof=1)),
          "per_source_auc":list(map(float,a)),
          "per_source_cov_nll":list(map(float,n))}
    primary=[l for l in FULL_GRID if l in good]
    if len(primary)<2:raise ValueError("primary grid insufficient")
    lm=max(primary,key=lambda l:(crit[str(l)]["source_macro_auc"],l))
    lc=min(primary,key=lambda l:(crit[str(l)]["covariance_mean_nll"],l))
    bestm=crit[str(lm)]["source_macro_auc"]
    mean_set=[l for l in primary if crit[str(l)]["source_macro_auc"]>=bestm-.01-1e-15]
    llex=min(mean_set,key=lambda l:(crit[str(l)]["covariance_mean_nll"],abs(l-lm),l))
    besttail=max(crit[str(l)]["source_tail_cvar25"] for l in mean_set)
    tail_set=[l for l in mean_set if crit[str(l)]["source_tail_cvar25"]>=besttail-.005-1e-15]
    lmtf=min(tail_set,key=lambda l:(crit[str(l)]["covariance_mean_nll"],abs(l-lm),l))
    dense_lc=min(good,key=lambda l:(crit[str(l)]["covariance_mean_nll"],l))
    return {
      "lambda_source_macro":float(lm),"lambda_covariance_nll":float(lc),
      "lambda_discriminability_first":float(llex),
      "lambda_mean_tail_fidelity":float(lmtf),
      "mean_admissible_lambdas":list(map(float,mean_set)),
      "tail_admissible_lambdas":list(map(float,tail_set)),
      "dense_lambda_covariance_nll":float(dense_lc),
      "feasible_dense_lambdas":list(map(float,good)),
      "criteria_by_lambda":crit,
      "source_covariance_regret_by_lambda":{
        str(l):float(crit[str(l)]["covariance_mean_nll"]-crit[str(dense_lc)]["covariance_mean_nll"])
        for l in good},
      "source_fold_receipts":receipts}

def load_dataset(name,data_root):
    ds,subjects,tmax=build_dataset(name);ch=native_channels(ds)
    raw_files=[]
    for sid in subjects:
        for p in flatten(ds.data_path(sid,path=str(data_root),force_update=False,verbose=False)):
            if p.exists():raw_files.append({"subject":sid,"name":p.name,"bytes":p.stat().st_size,"sha256":sha(p)})
    paradigm=MotorImagery(n_classes=2,events=["right_hand","feet"],
        fmin=8,fmax=30,tmin=0.,tmax=tmax,baseline=None,
        channels=list(ch),resample=160.,reject_by_annotation=False)
    Cs=[];ys=[];gs=[];sessions=[];runs=[];receipt=[]
    for sid in subjects:
        X,lab,meta=paradigm.get_data(dataset=ds,subjects=[sid],n_jobs=1)
        lab=np.asarray(lab)
        if set(lab)!={"right_hand","feet"}:raise ValueError(f"labels {set(lab)}")
        y=(lab=="feet").astype(np.int64)
        C=covariances(X,len(ch))
        ses=np.asarray(meta["session"].astype(str) if "session" in meta else ["0"]*len(y))
        run=np.asarray(meta["run"].astype(str) if "run" in meta else ["0"]*len(y))
        if len(ses)!=len(y) or len(run)!=len(y):raise ValueError("metadata denominator drift")
        Cs.append(C);ys.append(y);gs.extend([sid]*len(y));sessions.extend(ses);runs.extend(run)
        receipt.append({"subject":sid,"trials":int(len(y)),"right_hand":int(np.sum(y==0)),
                        "feet":int(np.sum(y==1)),"sessions":sorted(set(ses.tolist())),
                        "runs":sorted(set(run.tolist())),"samples_per_trial":int(X.shape[-1])})
        del X
    C=np.concatenate(Cs);y=np.concatenate(ys);g=np.asarray(gs,int)
    return ds,subjects,ch,C,y,g,np.asarray(sessions),np.asarray(runs),receipt,raw_files

def freeze_scores(name,C,y,g,sessions,methods_out):
    people=tuple(sorted(set(map(int,g))))
    all_method_scores=[];all_dense_scores=[];dense_lams=None;fold_meta=[]
    for target in people:
        tr=g!=target;te=~tr
        path=source_dense_path(C[tr],y[tr],g[tr])
        inv=reference(C[tr]);Xtr=tangent(C[tr],inv);Xte=tangent(C[te],inv)
        state=spectral_state(Xtr,y[tr])
        dense={str(l):ridge_scores(state,Xte,l) for l in path["feasible_dense_lambdas"]}
        if dense_lams is None:dense_lams=tuple(path["feasible_dense_lambdas"])
        if tuple(path["feasible_dense_lambdas"])!=dense_lams:
            raise ValueError("dense feasible path differs across outer folds")
        ms={}
        ms["LedoitWolf_LDA"],lw=analytic_scores(Xtr,y[tr],Xte,"LW")
        ms["OAS_LDA"],oas=analytic_scores(Xtr,y[tr],Xte,"OAS")
        ms["source_cov_nll_full_domain_LDA"]=dense[str(path["lambda_covariance_nll"])]
        ms["source_macro_full_domain_LDA"]=dense[str(path["lambda_source_macro"])]
        ms["discriminability_first_LDA"]=dense[str(path["lambda_discriminability_first"])]
        ms["mean_tail_fidelity_LDA"]=dense[str(path["lambda_mean_tail_fidelity"])]
        lr=LogisticRegression(C=1.,solver="lbfgs",max_iter=3000);lr.fit(Xtr,y[tr])
        ms["TS_LogisticRegression"]=lr.decision_function(Xte)
        mdm=MDM(metric="riemann",n_jobs=1);mdm.fit(C[tr],y[tr]);dist=mdm.transform(C[te])
        ms["Riemannian_MDM"]=dist[:,0]-dist[:,1]
        if set(ms)!=set(METHODS):raise ValueError("method set drift")
        all_method_scores.append(np.stack([ms[m] for m in METHODS],axis=1))
        all_dense_scores.append(np.stack([dense[str(l)] for l in dense_lams],axis=1))
        fold_meta.append({"target_subject":int(target),"target_rows":int(te.sum()),
          "source_path":path,"analytic_shrinkage":{"LedoitWolf":lw,"OAS":oas},
          "target_sessions":sessions[te].tolist()})
    return np.concatenate(all_method_scores),np.concatenate(all_dense_scores),dense_lams,fold_meta

def exact_signflip(d):
    d=np.asarray(d,float);obs=abs(float(d.mean()));n=len(d);ex=0;total=1<<n
    for mask in range(total):
        s=np.ones(n)
        for i in range(n):
            if mask&(1<<i):s[i]=-1
        if abs(float(np.mean(s*d)))>=obs-1e-15:ex+=1
    return float(ex/total)

def tail_stats(d):
    d=np.asarray(d,float)
    def cv(f):
        x=np.sort(d);k=max(1,int(math.ceil(f*len(x))));return float(x[:k].mean()),k
    c25,k25=cv(.25);c10,k10=cv(.10)
    return {"mean":float(d.mean()),"median":float(np.median(d)),
      "q25":float(np.quantile(d,.25)),"q10":float(np.quantile(d,.10)),
      "q05":float(np.quantile(d,.05)),"worst":float(d.min()),"best":float(d.max()),
      "cvar25":c25,"cvar25_n":k25,"cvar10":c10,"cvar10_n":k10,
      "improved":int(np.sum(d>0)),"harm_gt_0_01":int(np.sum(d<-.01)),
      "harm_gt_0_02":int(np.sum(d<-.02)),
      "harm_rate_gt_0_02":float(np.mean(d<-.02))}

def paired_summary(vals_c,vals_b,seed_offset=0):
    d=np.asarray(vals_c)-np.asarray(vals_b);rng=np.random.default_rng(SEED+seed_offset)
    idx=rng.integers(0,len(d),size=(20000,len(d)));bm=d[idx].mean(1)
    out=tail_stats(d);out.update({
      "bootstrap95":[float(x) for x in np.quantile(bm,[.025,.975])],
      "exact_signflip_p":exact_signflip(d),"per_subject_delta":list(map(float,d))})
    return out

def holm(ps):
    p=np.asarray(ps,float);order=np.argsort(p);out=np.empty(len(p));run=0.
    for rank,i in enumerate(order):
        run=max(run,min(1.,(len(p)-rank)*float(p[i])));out[i]=run
    return list(map(float,out))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset",required=True,choices=["BNCI2014_002","BNCI2015_001"])
    ap.add_argument("--data-root",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    start=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ds,subjects,ch,C,y,g,sessions,runs,receipt,raw_files=load_dataset(a.dataset,Path(a.data_root))
    if set(np.unique(g))!=set(subjects):raise ValueError("participant denominator drift")
    expected_dim=len(ch)*(len(ch)+1)//2

    # PHASE A: freeze target scores before target-label evaluation.
    method_scores,dense_scores,dense_lams,fold_meta=freeze_scores(a.dataset,C,y,g,sessions,METHODS)
    target_order=np.concatenate([np.flatnonzero(g==sid) for sid in subjects])
    if method_scores.shape[0]!=len(target_order) or dense_scores.shape[0]!=len(target_order):
        raise ValueError("score denominator mismatch")
    score_path=out/"frozen_target_scores_before_auc.npz"
    np.savez_compressed(score_path,
      row_index=target_order,subject=g[target_order],session=sessions[target_order],run=runs[target_order],
      method_names=np.asarray(METHODS,dtype="U48"),method_scores=method_scores,
      dense_lambdas=np.asarray(dense_lams,float),dense_scores=dense_scores)
    score_sha=sha(score_path)

    # PHASE B: only now map frozen score rows to target labels and evaluate.
    yy=y[target_order];ss=g[target_order];ses=sessions[target_order]
    per_subject=[];path_auc_by_subject={}
    for pos,sid in enumerate(subjects):
        m=ss==sid
        auc={METHODS[j]:float(roc_auc_score(yy[m],method_scores[m,j])) for j in range(len(METHODS))}
        pA={str(dense_lams[j]):float(roc_auc_score(yy[m],dense_scores[m,j])) for j in range(len(dense_lams))}
        session_auc={}
        for s in sorted(set(ses[m].tolist())):
            q=m&(ses==s)
            if len(set(yy[q]))==2:
                session_auc[str(s)]={METHODS[j]:float(roc_auc_score(yy[q],method_scores[q,j]))
                                     for j in range(len(METHODS))}
        per_subject.append({"target_subject":int(sid),"target_trials":int(np.sum(m)),
          "class_counts":{"right_hand":int(np.sum(yy[m]==0)),"feet":int(np.sum(yy[m]==1))},
          "auc":auc,"session_auc":session_auc,
          "source_path":fold_meta[pos]["source_path"],
          "analytic_shrinkage":fold_meta[pos]["analytic_shrinkage"]})
        path_auc_by_subject[str(sid)]=pA

    macro={m:float(np.mean([r["auc"][m] for r in per_subject])) for m in METHODS}
    h2=paired_summary([r["auc"]["source_macro_full_domain_LDA"] for r in per_subject],
                      [r["auc"]["source_cov_nll_full_domain_LDA"] for r in per_subject],1)
    h3=paired_summary([r["auc"]["source_macro_full_domain_LDA"] for r in per_subject],
                      [r["auc"]["LedoitWolf_LDA"] for r in per_subject],2)
    padj=holm([h2["exact_signflip_p"],h3["exact_signflip_p"]])
    h2["holm_p_H2_H3"]=padj[0];h3["holm_p_H2_H3"]=padj[1]
    sec=paired_summary([r["auc"]["mean_tail_fidelity_LDA"] for r in per_subject],
                       [r["auc"]["source_macro_full_domain_LDA"] for r in per_subject],3)

    # Cross-fitted cohort validity frontier point estimates.
    macro_path={str(l):float(np.mean([path_auc_by_subject[str(s)][str(l)] for s in subjects]))
                for l in dense_lams}
    mean_regret={}
    for l in dense_lams:
        mean_regret[str(l)]=float(np.mean([
          r["source_path"]["source_covariance_regret_by_lambda"][str(l)]
          for r in per_subject]))
    budgets=sorted(set(mean_regret.values()))
    frontier=[]
    first_zero=None
    # Downstream here is improvement over cohort macro AUC at mean-regret minimum.
    lc=min(dense_lams,key=lambda l:(mean_regret[str(l)],l));base=macro_path[str(lc)]
    for b in budgets:
        adm=[l for l in dense_lams if mean_regret[str(l)]<=b+1e-15]
        vals=[macro_path[str(l)]-base for l in adm];lo=min(vals);hi=max(vals)
        cross=lo<=0<=hi
        if cross and first_zero is None and b>0:first_zero=float(b)
        frontier.append({"budget":float(b),"n_lambdas":len(adm),
          "delta_auc_min":float(lo),"delta_auc_max":float(hi),"span":float(hi-lo)})
    result={
      "status":"V5_TRUE_PREDATA_EXTERNAL_COMPLETE",
      "dataset":a.dataset,"dataset_code":ds.code,"subjects":list(subjects),
      "n_subjects":len(subjects),"native_channels":list(ch),"n_channels":len(ch),
      "tangent_dimension":expected_dim,"task":"right_hand_vs_feet",
      "band_hz":[8,30],"epoch_seconds":[0,5],"resample_hz":160.,
      "reject_by_annotation":False,"target_information_track":"Z0",
      "score_freeze_precedes_target_auc":True,"score_bundle_sha256":score_sha,
      "dense_grid":list(map(float,dense_lams)),"methods":list(METHODS),
      "participant_receipt":receipt,"raw_files":raw_files,
      "macro_auc":macro,"primary_H2_macro_vs_cov":h2,"primary_H3_macro_vs_LW":h3,
      "secondary_mean_tail_vs_macro":sec,
      "per_subject":per_subject,"target_auc_path_by_subject":path_auc_by_subject,
      "crossfitted_validity_frontier":{
        "definition":"mean outer-source covariance regret vs participant-macro target AUC path",
        "lambda_mean_regret_min":float(lc),"macro_auc_at_regret_min":float(base),
        "mean_covariance_regret_by_lambda":mean_regret,
        "participant_macro_auc_by_lambda":macro_path,
        "envelope":frontier,"first_nonzero_budget_with_zero_in_delta_envelope":first_zero},
      "environment":{"python":platform.python_version(),"moabb":moabb.__version__,
        "numpy":np.__version__,"sklearn":sklearn.__version__,"mne":mne.__version__,
        "pyriemann":pyriemann.__version__},
      "wall_seconds":round(time.monotonic()-start,2)}
    rp=out/(a.dataset+"_v5_result.json");rp.write_text(json.dumps(result,indent=2,sort_keys=True))
    result_sha=sha(rp)
    (out/"receipt.json").write_text(json.dumps({
      "dataset":a.dataset,"result_sha256":result_sha,"score_bundle_sha256":score_sha,
      "n_subjects":len(subjects),"total_trials":int(len(y)),
      "per_subject_trials":{str(r["target_subject"]):r["target_trials"] for r in per_subject},
      "status":"V5_RESULT_RECEIPT"},indent=2,sort_keys=True))
    print(json.dumps({"dataset":a.dataset,"status":result["status"],"macro_auc":macro,
      "H2":h2,"H3":h3,"secondary_mean_tail_vs_macro":sec,
      "score_bundle_sha256":score_sha,"result_sha256":result_sha,
      "wall_seconds":result["wall_seconds"]},indent=2))

if __name__=="__main__":main()
