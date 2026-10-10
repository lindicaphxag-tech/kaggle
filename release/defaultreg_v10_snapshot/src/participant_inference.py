"""Participant-level inference for Default Regularisation.

This module intentionally treats participant as the outcome unit.
It provides:
- paired participant mean/median/hard-tail deltas;
- exact two-sided sign-flip test for n<=20;
- deterministic Monte-Carlo sign-flip otherwise;
- participant bootstrap CI for mean delta;
- Holm family-wise correction;
- equal-dataset hierarchical bootstrap summary across independent cohorts.

Important limitation:
Participant resampling does NOT reproduce uncertainty caused by overlapping
source training sets in LOSO. That source-training uncertainty must be audited
separately by source-subset/jackknife refits. The output records this explicitly.
"""
from __future__ import annotations
import argparse,itertools,json,math
from pathlib import Path
import numpy as np

def _rows(data):
    if isinstance(data.get("per_target_subject"),dict):
        return [(str(k),v["auc"]) for k,v in sorted(
            data["per_target_subject"].items(),key=lambda x:int(x[0]))]
    if isinstance(data.get("folds"),list):
        out=[(str(v["target_subject"]),v["auc"]) for v in data["folds"]]
        if len({x[0] for x in out})!=len(out):raise ValueError("Duplicate participant")
        return out
    raise ValueError("Unsupported participant result format")

def paired_deltas(data,baseline,candidate):
    rows=_rows(data)
    if len(rows)<2:raise ValueError("Need >=2 participants")
    vals=[]
    for sid,auc in rows:
        if baseline not in auc or candidate not in auc:
            raise ValueError(f"Missing method for participant {sid}")
        b=float(auc[baseline]);c=float(auc[candidate])
        if not all(map(math.isfinite,(b,c))):raise ValueError("Nonfinite AUC")
        vals.append(c-b)
    return np.asarray(vals,dtype=float)

def signflip_pvalue(deltas,seed=20261010,mc_draws=200000):
    d=np.asarray(deltas,dtype=float)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():
        raise ValueError("Finite 1D paired deltas required")
    obs=abs(float(np.mean(d)))
    n=len(d)
    tol=64*np.finfo(float).eps*max(1.0,obs)
    if n<=20:
        exceed=0;total=1<<n
        for mask in range(total):
            s=np.ones(n)
            for i in range(n):
                if mask&(1<<i):s[i]=-1
            if abs(float(np.mean(s*d)))>=obs-tol:exceed+=1
        return float(exceed/total),"exact",int(total)
    rng=np.random.default_rng(seed)
    exceed=0
    batch=10000
    done=0
    while done<mc_draws:
        q=min(batch,mc_draws-done)
        signs=rng.choice((-1.,1.),size=(q,n))
        stats=np.abs(np.mean(signs*d[None,:],axis=1))
        exceed+=int(np.sum(stats>=obs-tol));done+=q
    return float((exceed+1)/(mc_draws+1)),"monte_carlo",int(mc_draws)

def bootstrap_mean_ci(deltas,seed=20261010,reps=20000,alpha=.05):
    d=np.asarray(deltas,dtype=float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(reps,len(d)))
    means=np.mean(d[idx],axis=1)
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2])
    return float(lo),float(hi),int(reps)

def bootstrap_mean_direction(deltas,seed=20261010,reps=20000):
    """Descriptive bootstrap stability of the mean-effect direction.

    This is not a posterior probability and must not be reported as one.
    """
    d=np.asarray(deltas,dtype=float)
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():
        raise ValueError("Finite 1D paired deltas required")
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(d),size=(reps,len(d)))
    means=np.mean(d[idx],axis=1)
    return {
      "fraction_mean_gt_zero":float(np.mean(means>0)),
      "fraction_mean_lt_zero":float(np.mean(means<0)),
      "fraction_mean_eq_zero":float(np.mean(means==0)),
      "bootstrap_reps":int(reps),
      "interpretation":"descriptive bootstrap direction stability; not a posterior probability"}

def leave_one_participant_influence(deltas):
    """Audit whether one participant determines the mean-effect sign."""
    d=np.asarray(deltas,dtype=float)
    if d.ndim!=1 or len(d)<3 or not np.isfinite(d).all():
        raise ValueError("Need >=3 finite paired deltas")
    full=float(np.mean(d))
    loo=np.asarray([np.mean(np.delete(d,i)) for i in range(len(d))],dtype=float)
    sign=lambda x: 1 if x>0 else (-1 if x<0 else 0)
    full_sign=sign(full)
    loo_signs=np.asarray([sign(x) for x in loo],dtype=int)
    return {
      "full_mean_delta_auc":full,
      "loo_mean_min":float(np.min(loo)),
      "loo_mean_max":float(np.max(loo)),
      "loo_mean_median":float(np.median(loo)),
      "max_abs_loo_mean_shift":float(np.max(np.abs(loo-full))),
      "loo_mean_sign_stable":bool(np.all(loo_signs==full_sign)),
      "any_single_participant_changes_mean_sign":bool(np.any(loo_signs!=full_sign)),
      "n_leave_one_out":int(len(loo))}

def lower_cvar(deltas,fraction=.10):
    """Mean of the worst ceil(fraction*n) paired participant deltas."""
    d=np.sort(np.asarray(deltas,dtype=float))
    if d.ndim!=1 or len(d)<2 or not np.isfinite(d).all():
        raise ValueError("Finite 1D paired deltas required")
    if not 0<fraction<=1:
        raise ValueError("fraction must lie in (0,1]")
    k=max(1,int(math.ceil(fraction*len(d))))
    return float(np.mean(d[:k])),int(k)

def summarize(deltas):
    d=np.asarray(deltas,dtype=float)
    p,mode,draws=signflip_pvalue(d)
    lo,hi,reps=bootstrap_mean_ci(d)
    direction=bootstrap_mean_direction(d)
    influence=leave_one_participant_influence(d) if len(d)>=3 else None
    return {
      "n_participants":int(len(d)),
      "mean_delta_auc":float(np.mean(d)),
      "median_delta_auc":float(np.median(d)),
      "q25_delta_auc":float(np.quantile(d,.25)),
      "q10_delta_auc":float(np.quantile(d,.10)),
      "q05_delta_auc":float(np.quantile(d,.05)),
      "cvar10_delta_auc":lower_cvar(d,.10)[0],
      "cvar10_n":lower_cvar(d,.10)[1],
      "cvar25_delta_auc":lower_cvar(d,.25)[0],
      "cvar25_n":lower_cvar(d,.25)[1],
      "min_delta_auc":float(np.min(d)),
      "max_delta_auc":float(np.max(d)),
      "improved":int(np.sum(d>0)),
      "tied":int(np.sum(d==0)),
      "harmed_gt_0_01":int(np.sum(d < -.01)),
      "harmed_gt_0_02":int(np.sum(d < -.02)),
      "harm_rate_gt_0_01":float(np.mean(d < -.01)),
      "harm_rate_gt_0_02":float(np.mean(d < -.02)),
      "improvement_rate":float(np.mean(d > 0)),
      "participant_bootstrap_mean_ci95":[lo,hi],
      "participant_bootstrap_reps":reps,
      "bootstrap_direction_stability":direction,
      "leave_one_participant_influence":influence,
      "signflip_p_two_sided":p,
      "signflip_mode":mode,
      "signflip_draws":draws,
      "shared_training_refit_uncertainty_included":False}

def holm(pvalues):
    """Return Holm-adjusted p-values preserving original order."""
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or len(p)==0 or np.any((p<0)|(p>1)|~np.isfinite(p)):
        raise ValueError("Valid p-values required")
    order=np.argsort(p)
    adj=np.empty(len(p),dtype=float)
    running=0.0
    m=len(p)
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*p[idx])
        running=max(running,val)
        adj[idx]=running
    return adj.tolist()

def hierarchical_equal_dataset_bootstrap(dataset_deltas,seed=20261010,reps=20000):
    """Equal-dataset-weighted descriptive bootstrap.

    Resample datasets, then participants within each selected dataset. This
    avoids silently weighting a 100-subject cohort 10x a 10-subject cohort.
    It is descriptive when the number of independent datasets is small.
    """
    arrays=[np.asarray(x,dtype=float) for x in dataset_deltas]
    if len(arrays)<2 or any(len(x)<2 for x in arrays):
        raise ValueError("Need >=2 datasets with >=2 participants each")
    rng=np.random.default_rng(seed)
    vals=np.empty(reps)
    D=len(arrays)
    for r in range(reps):
        ds=rng.integers(0,D,size=D)
        means=[]
        for j in ds:
            a=arrays[int(j)]
            idx=rng.integers(0,len(a),size=len(a))
            means.append(float(np.mean(a[idx])))
        vals[r]=float(np.mean(means))
    return {
      "equal_dataset_mean_delta":float(np.mean([np.mean(x) for x in arrays])),
      "hierarchical_bootstrap_ci95":[float(x) for x in np.quantile(vals,[.025,.975])],
      "datasets":D,"bootstrap_reps":reps,
      "interpretation":"descriptive replication summary; small-D datasets are not treated as thousands of independent trials"}

def analyse(files,baseline,candidate,out):
    results=[]
    deltas=[]
    for file in files:
        data=json.loads(Path(file).read_text())
        d=paired_deltas(data,baseline,candidate)
        s=summarize(d)
        s["file"]=str(file)
        s["dataset"]=data.get("dataset") or data.get("official_dataset") or Path(file).stem
        results.append(s);deltas.append(d)
    adjusted=holm([x["signflip_p_two_sided"] for x in results])
    for s,padj in zip(results,adjusted):s["holm_p_across_dataset_primary_tests"]=padj
    output={
      "baseline":baseline,"candidate":candidate,
      "per_dataset":results,
      "participant_is_inferential_unit":True,
      "trial_level_pseudoreplication_used":False,
      "shared_loso_training_uncertainty_included":False,
      "shared_training_warning":"Participant bootstrap/sign-flip do not emulate refitting under changed source pools; add source-jackknife stability before strong uncertainty claims.",
      "status":"PARTICIPANT_LEVEL_INFERENCE_WITH_EXPLICIT_SHARED_TRAINING_LIMIT"}
    if len(deltas)>=2:
        output["cross_dataset"]=hierarchical_equal_dataset_bootstrap(deltas)
    Path(out).write_text(json.dumps(output,indent=2,sort_keys=True))
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("files",nargs="+");p.add_argument("--baseline",required=True)
    p.add_argument("--candidate",required=True);p.add_argument("--out",required=True)
    a=p.parse_args()
    print(json.dumps(analyse(a.files,a.baseline,a.candidate,a.out),indent=2,sort_keys=True))
if __name__=="__main__":main()
