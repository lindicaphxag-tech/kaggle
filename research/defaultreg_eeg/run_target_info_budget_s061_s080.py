"""Prospective target-information budget experiment on EEGMMIDB S061-S080.

Z0: source-only.
U5/U10/U20: first chronological k target trials are UNLABELED calibration,
future trials only are evaluated.
U-all: complete unlabeled target batch; explicitly transductive.

Calibration operator is fixed source-tangent feature translation only.
No target label changes geometry, shrinkage, classifier or translation.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from run_official_edf_baselines import (
    load_or_fetch,extract_original_trials,reference,tangent,within_residual,sha)
from run_full_domain_s021_s040 import source_full_path,FULL_GRID

SOURCE=tuple(range(1,9))
TARGET=tuple(range(61,81))
RUNS=(4,8,12)
BUDGETS=(5,10,20)
SEED=20261010

def fit_ridge(X,y,lam):
    R,d,mid=within_residual(X,y)
    S=(R.T@R)/(len(X)-2);S=(S+S.T)/2
    nu=max(float(np.trace(S))/len(d),np.finfo(float).eps)
    A=(1-lam)*S+lam*nu*np.eye(len(d))
    w=np.linalg.solve(A,d)
    return w,mid,float(lam)

def fit_lw(X,y):
    R,d,mid=within_residual(X,y)
    est=LedoitWolf(store_precision=False).fit(R)
    C=(est.covariance_+est.covariance_.T)/2
    C+=np.eye(C.shape[0])*max(float(np.trace(C)),1.)*1e-12
    return np.linalg.solve(C,d),mid,float(est.shrinkage_)

def score(model,X):
    w,mid,_=model
    return (X-mid)@w

def paired_summary(d,name):
    d=np.asarray(d,dtype=float)
    rng=np.random.default_rng(SEED+len(name))
    idx=rng.integers(0,len(d),size=(20000,len(d)))
    lo,hi=np.quantile(np.mean(d[idx],axis=1),[.025,.975])
    total=1<<len(d);obs=abs(float(np.mean(d)));exceed=0;bit=np.arange(len(d),dtype=np.uint64)
    chunk=1<<15
    for start in range(0,total,chunk):
        ids=np.arange(start,min(total,start+chunk),dtype=np.uint64)[:,None]
        signs=1.-2.*((ids>>bit)&1).astype(float)
        vals=np.abs(np.mean(signs*d[None,:],axis=1))
        exceed+=int(np.sum(vals>=obs-1e-15))
    return {"contrast":name,"n":len(d),"mean":float(np.mean(d)),
            "median":float(np.median(d)),"q25":float(np.quantile(d,.25)),
            "min":float(np.min(d)),"max":float(np.max(d)),
            "improved":int(np.sum(d>0)),"harmed_gt_0_02":int(np.sum(d<-.02)),
            "bootstrap_ci95":[float(lo),float(hi)],
            "exact_signflip_p_two_sided":float(exceed/total)}

def holm(ps):
    ps=np.asarray(ps);order=np.argsort(ps);out=np.empty(len(ps));r=0.
    for rank,i in enumerate(order):
        r=max(r,min(1.,(len(ps)-rank)*float(ps[i])));out[i]=r
    return list(map(float,out))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--edf-root",default="edf_original")
    ap.add_argument("--out",default="defaultreg_target_budget_s061_s080")
    a=ap.parse_args();root=Path(a.edf_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic();Cs=[];ys=[];gs=[];ids=[];files=[]
    for sid in SOURCE+TARGET:
        for run in RUNS:
            p=load_or_fetch(root,sid,run)
            C,y,trialids=extract_original_trials(p,sid,run)
            Cs.append(C);ys.append(y);gs.extend([sid]*len(y));ids.extend(trialids)
            files.append({"subject":sid,"run":run,"sha256":sha(p),
                          "edf_bytes":p.stat().st_size,
                          "all_original_cues":len(trialids),
                          "selected_and_scored":len(y)})
    C=np.concatenate(Cs);y=np.concatenate(ys);g=np.asarray(gs)
    run_arr=np.asarray([x["run"] for x in ids]);sample=np.asarray([x["event_sample"] for x in ids])
    src=np.isin(g,SOURCE);tgt=np.isin(g,TARGET)
    if np.intersect1d(np.unique(g[src]),np.unique(g[tgt])).size:raise ValueError("source-target overlap")
    if not all(f["all_original_cues"]==f["selected_and_scored"] for f in files):
        raise ValueError("task cue denominator drift")
    path=source_full_path(C[src],y[src],g[src])
    inv=reference(C[src]);Xs=tangent(C[src],inv);Xt=tangent(C[tgt],inv)
    source_mean=Xs.mean(0)
    models={
      "source_discriminability":fit_ridge(Xs,y[src],path["lambda_discriminability"]),
      "LedoitWolf":fit_lw(Xs,y[src])}
    gt=g[tgt];yt=y[tgt];rt=run_arr[tgt];st=sample[tgt]
    result_by_subject={}
    for sid in TARGET:
        m=gt==sid
        order=np.lexsort((st[m],rt[m]))
        X=Xt[m][order];yy=yt[m][order];rr=rt[m][order];ss=st[m][order]
        # Freeze all score vectors BEFORE target labels are used for AUC.
        raw_scores={name:score(model,X) for name,model in models.items()}
        uall_shift=X.mean(0)-source_mean
        uall_scores={name:score(model,X-uall_shift) for name,model in models.items()}
        budgets={}
        for k in BUDGETS:
            if len(X)<=k:raise ValueError(f"subject {sid} lacks >{k} target trials")
            shift=X[:k].mean(0)-source_mean
            future=X[k:]
            z={name:score(model,future) for name,model in models.items()}
            u={name:score(model,future-shift) for name,model in models.items()}
            yfuture=yy[k:]
            cell={"calibration_trials":k,"evaluation_trials":int(len(yfuture)),
                  "evaluation_label_counts":{"T1":int(np.sum(yfuture==0)),"T2":int(np.sum(yfuture==1))},
                  "chronology_prefix":[{"run":int(rr[i]),"event_sample":int(ss[i])} for i in range(k)],
                  "auc_Z0_suffix":{},"auc_Uk_suffix":{}}
            if set(np.unique(yfuture))=={0,1}:
                cell["auc_Z0_suffix"]={name:float(roc_auc_score(yfuture,v)) for name,v in z.items()}
                cell["auc_Uk_suffix"]={name:float(roc_auc_score(yfuture,v)) for name,v in u.items()}
            budgets[str(k)]=cell
        result_by_subject[str(sid)]={
          "total_target_trials":int(len(yy)),
          "label_counts":{"T1":int(np.sum(yy==0)),"T2":int(np.sum(yy==1))},
          "Z0_all_auc":{name:float(roc_auc_score(yy,v)) for name,v in raw_scores.items()},
          "Uall_transductive_all_auc":{name:float(roc_auc_score(yy,v)) for name,v in uall_scores.items()},
          "budgets":budgets}
    # Predeclared participant contrasts.
    contrasts=[]
    d_model=[v["Z0_all_auc"]["source_discriminability"]-v["Z0_all_auc"]["LedoitWolf"]
             for v in result_by_subject.values()]
    contrasts.append(paired_summary(d_model,"D1_Z0_sourceDiscriminability_minus_LW"))
    for k in BUDGETS:
        for method in ("source_discriminability","LedoitWolf"):
            vals=[]
            for v in result_by_subject.values():
                cell=v["budgets"][str(k)]
                if cell["auc_Uk_suffix"]:
                    vals.append(cell["auc_Uk_suffix"][method]-cell["auc_Z0_suffix"][method])
            if len(vals)!=len(TARGET):raise ValueError("Unavailable matched suffix AUC")
            contrasts.append(paired_summary(vals,f"D2_U{k}_minus_Z0_suffix_{method}"))
    # U-all is descriptive but still participant paired.
    for method in ("source_discriminability","LedoitWolf"):
        vals=[v["Uall_transductive_all_auc"][method]-v["Z0_all_auc"][method]
              for v in result_by_subject.values()]
        contrasts.append(paired_summary(vals,f"D3_Uall_minus_Z0_all_{method}"))
    adj=holm([c["exact_signflip_p_two_sided"] for c in contrasts])
    for c,padj in zip(contrasts,adj):c["holm_p_deployment_family"]=padj
    result={
      "status":"REAL_EEGMMIDB_S061_S080_TARGET_INFORMATION_BUDGET_PROSPECTIVE",
      "source_participants":list(SOURCE),"heldout_participants":list(TARGET),
      "runs":list(RUNS),"features":"8-30Hz 21ch source-tangent 231D",
      "full_domain_grid":list(FULL_GRID),"source_path":path,
      "source_selected_lambda":path["lambda_discriminability"],
      "LedoitWolf_lambda":models["LedoitWolf"][2],
      "information_budgets":{"Z0":"no target statistic",
                             "U5/U10/U20":"unlabeled chronological prefix feature translation; evaluate future suffix only",
                             "Uall":"complete unlabeled target batch feature translation; transductive"},
      "target_labels_used_for_calibration_or_model_selection":False,
      "target_batch_statistics_used_in_Z0":False,
      "per_target_subject":result_by_subject,
      "predeclared_contrasts":contrasts,
      "source_files":files,"wall_seconds":round(time.monotonic()-t0,2),
      "interpretation":"Offline deployment-budget sensitivity; U-all is transductive, not calibration-free or online evidence."}
    (out/"target_information_budget_result.json").write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({"status":result["status"],"source_selected_lambda":result["source_selected_lambda"],
                      "LedoitWolf_lambda":result["LedoitWolf_lambda"],
                      "predeclared_contrasts":contrasts,"wall_seconds":result["wall_seconds"]},indent=2))
    return result
if __name__=="__main__":main()
