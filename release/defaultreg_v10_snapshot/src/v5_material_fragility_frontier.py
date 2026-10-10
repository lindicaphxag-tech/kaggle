"""Corrected material-fragility frontier for frozen v5 EEG results.

Implements the pre-outcome correction frozen in
V5_PREOUTCOME_FRONTIER_CRITICAL_BUDGET_CORRECTION_20261010.md.

The raw v5 runner emitted a generic zero-crossing field for AUC centered at
its own covariance optimum. That field is structurally uninformative because
zero is present by construction. This module deliberately ignores it.

For each outer participant g and dense-grid lambda, inputs are:
    r_g(lambda): source-only covariance predictive-NLL regret
    A_g(lambda): frozen target AUC

For one paired participant bootstrap draw, the SAME participant indices are
used at every lambda:
    r_bar(lambda) = mean_g r_g(lambda)
    A_bar(lambda) = mean_g A_g(lambda)

The aggregate fidelity coordinate is shifted by its finite-grid minimum:
    delta(lambda) = r_bar(lambda) - min_lambda r_bar(lambda).

This makes delta=0 the best common finite-grid covariance-fidelity compromise
for that bootstrap draw while preserving all lambda ordering and regret
differences.

The downstream validity envelope is
    E(delta) = [min A_bar(lambda), max A_bar(lambda)]
over lambdas whose shifted regret <= delta.

Primary material-fragility threshold:
    delta_0.02 = inf{delta: span(E(delta)) >= 0.02 AUC}.

Secondary descriptive thresholds are 0.01 and 0.05 AUC.

Raw per-dimension NLL regret is primary. Regret normalized by the source-fold
NLL SD / SE at each outer fold's covariance optimum is reported only as
scale-aware sensitivity.

This is a post-score measurement-validity analysis. It performs NO target-
dependent model selection and must never replace a frozen method.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


DEFAULT_THRESHOLDS=(0.01,0.02,0.05)
DEFAULT_STRIDES=(1,2,4,8)
DEFAULT_SEED=20261010
DEFAULT_BOOTSTRAPS=20000


@dataclass(frozen=True)
class FrontierPoint:
    axis:str
    n_participants:int
    n_lambdas:int
    aggregate_regret_min_raw:float
    max_auc_span:float
    material_budget_by_auc_span:dict[str,float|None]
    auc_at_min_regret:float
    lambda_at_min_regret:float
    target_path_used_for_selection:bool=False
    status:str="POST_SCORE_VALIDITY_AUDIT_NO_TARGET_SELECTION"


def _float_dict(d:Mapping[object,object])->dict[float,float]:
    out={}
    for k,v in d.items():
        x=float(k); y=float(v)
        if math.isfinite(x) and math.isfinite(y):
            out[x]=y
    return out


def _lookup_lambda(mapping:Mapping[object,object], lam:float):
    if lam in mapping:
        return mapping[lam]
    if str(lam) in mapping:
        return mapping[str(lam)]
    matches=[v for k,v in mapping.items()
             if math.isclose(float(k),lam,rel_tol=0.0,abs_tol=1e-12)]
    if len(matches)!=1:
        raise ValueError(f"Missing/ambiguous lambda={lam}")
    return matches[0]


def extract_v5_paths(result:Mapping)->dict:
    if result.get("status")!="V5_TRUE_PREDATA_EXTERNAL_COMPLETE":
        raise ValueError("Expected completed frozen v5 result")
    if result.get("target_information_track")!="Z0":
        raise ValueError("Corrected frontier currently admits Z0 v5 result only")
    if result.get("score_freeze_precedes_target_auc") is not True:
        raise ValueError("Score-freeze chronology is not verified")

    subjects=tuple(map(int,result["subjects"]))
    per={int(row["target_subject"]):row for row in result["per_subject"]}
    if set(per)!=set(subjects):
        raise ValueError("per_subject denominator differs from frozen subjects")

    auc_by_subject=result["target_auc_path_by_subject"]
    if set(map(int,auc_by_subject))!=set(subjects):
        raise ValueError("target AUC path denominator differs from frozen subjects")

    dense=tuple(map(float,result["dense_grid"]))
    if len(dense)<8 or dense[0]!=0.0 or dense[-1]!=1.0:
        raise ValueError("Expected frozen dense [0,1] path")

    regret_rows=[]
    auc_rows=[]
    sd_scale=[]
    se_scale=[]
    for sid in subjects:
        sp=per[sid]["source_path"]
        regret=_float_dict(sp["source_covariance_regret_by_lambda"])
        auc=_float_dict(auc_by_subject[str(sid)])
        if any(l not in regret or l not in auc for l in dense):
            raise ValueError(f"Missing dense-grid path for subject {sid}")
        regret_rows.append([regret[l] for l in dense])
        auc_rows.append([auc[l] for l in dense])

        lc=float(sp["dense_lambda_covariance_nll"])
        criterion=_lookup_lambda(sp["criteria_by_lambda"],lc)
        sd=float(criterion["covariance_nll_sd"])
        se=float(criterion["covariance_nll_se"])
        eps=np.finfo(float).eps
        if not math.isfinite(sd) or not math.isfinite(se) or sd<=eps or se<=eps:
            raise ValueError(f"Invalid covariance NLL dispersion for subject {sid}")
        sd_scale.append(sd);se_scale.append(se)

    raw=np.asarray(regret_rows,float)
    auc=np.asarray(auc_rows,float)
    sd=np.asarray(sd_scale,float)[:,None]
    se=np.asarray(se_scale,float)[:,None]
    if not np.isfinite(raw).all() or not np.isfinite(auc).all():
        raise ValueError("Nonfinite v5 path")
    if np.any(raw < -1e-10):
        raise ValueError("Source covariance regret must be nonnegative")

    return {
        "dataset":str(result["dataset"]),
        "subjects":subjects,
        "lambdas":np.asarray(dense,float),
        "auc":auc,
        "regret_axes":{
            "raw_nll":np.maximum(raw,0.0),
            "sd_units":np.maximum(raw/sd,0.0),
            "se_units":np.maximum(raw/se,0.0),
        },
    }


def _envelope_from_aggregate(
    lambdas:np.ndarray,
    aggregate_regret:np.ndarray,
    aggregate_auc:np.ndarray,
    thresholds:Iterable[float],
)->tuple[FrontierPoint,list[dict]]:
    lambdas=np.asarray(lambdas,float)
    r=np.asarray(aggregate_regret,float)
    q=np.asarray(aggregate_auc,float)
    if lambdas.ndim!=1 or r.shape!=lambdas.shape or q.shape!=lambdas.shape:
        raise ValueError("Aligned 1D lambda/regret/AUC paths required")
    if not (np.isfinite(lambdas).all() and np.isfinite(r).all()
            and np.isfinite(q).all()):
        raise ValueError("Finite path required")

    raw_min=float(np.min(r))
    shifted=np.maximum(0.0,r-raw_min)
    # Build the envelope at unique fidelity budgets. Ties are evaluated
    # together so candidate ordering within one budget cannot affect delta_tau.
    budgets=np.unique(shifted)
    envelope=[]
    material={str(float(t)):None for t in thresholds}
    for budget in budgets:
        mask=shifted<=budget+1e-15
        vals=q[mask]
        lo=float(np.min(vals)); hi=float(np.max(vals)); span=hi-lo
        envelope.append({
            "budget":float(budget),
            "n_lambdas":int(np.sum(mask)),
            "auc_min":lo,
            "auc_max":hi,
            "auc_span":float(span),
        })
        for t in thresholds:
            key=str(float(t))
            if material[key] is None and span>=float(t)-1e-15:
                material[key]=float(budget)

    idx=int(np.argmin(r))
    point=FrontierPoint(
        axis="",
        n_participants=0,
        n_lambdas=int(len(lambdas)),
        aggregate_regret_min_raw=raw_min,
        max_auc_span=float(np.max(q)-np.min(q)),
        material_budget_by_auc_span=material,
        auc_at_min_regret=float(q[idx]),
        lambda_at_min_regret=float(lambdas[idx]),
    )
    return point,envelope


def _axis_point(lambdas,regret_rows,auc_rows,thresholds,axis):
    mean_r=np.mean(regret_rows,axis=0)
    mean_q=np.mean(auc_rows,axis=0)
    p,env=_envelope_from_aggregate(lambdas,mean_r,mean_q,thresholds)
    p=FrontierPoint(
        axis=axis,
        n_participants=int(regret_rows.shape[0]),
        n_lambdas=p.n_lambdas,
        aggregate_regret_min_raw=p.aggregate_regret_min_raw,
        max_auc_span=p.max_auc_span,
        material_budget_by_auc_span=p.material_budget_by_auc_span,
        auc_at_min_regret=p.auc_at_min_regret,
        lambda_at_min_regret=p.lambda_at_min_regret,
    )
    return p,env


def _grid_sensitivity(lambdas,regret_rows,auc_rows,thresholds,strides):
    out=[]
    L=len(lambdas)
    for stride in strides:
        stride=int(stride)
        if stride<1: raise ValueError("Grid strides must be positive")
        idx=list(range(0,L,stride))
        if idx[-1]!=L-1: idx.append(L-1)
        idx=np.asarray(sorted(set(idx)),int)
        p,_=_axis_point(
            lambdas[idx],regret_rows[:,idx],auc_rows[:,idx],
            thresholds,"raw_nll",
        )
        out.append({
            "stride":stride,
            "n_lambdas":int(len(idx)),
            "max_auc_span":p.max_auc_span,
            "material_budget_by_auc_span":p.material_budget_by_auc_span,
        })
    return out


def bootstrap_frontier(
    lambdas:np.ndarray,
    regret_rows:np.ndarray,
    auc_rows:np.ndarray,
    *,
    thresholds=DEFAULT_THRESHOLDS,
    reps:int=DEFAULT_BOOTSTRAPS,
    seed:int=DEFAULT_SEED,
)->dict:
    if reps<100:
        raise ValueError("Use >=100 bootstrap reps")
    n=regret_rows.shape[0]
    if n<2 or auc_rows.shape!=regret_rows.shape:
        raise ValueError("Aligned participant x lambda matrices required")
    rng=np.random.default_rng(seed)

    max_spans=np.empty(reps,float)
    delta={str(float(t)):np.full(reps,np.nan,float) for t in thresholds}
    raw_min=np.empty(reps,float)
    for b in range(reps):
        idx=rng.integers(0,n,size=n)
        r=np.mean(regret_rows[idx],axis=0)
        q=np.mean(auc_rows[idx],axis=0)
        p,_=_envelope_from_aggregate(lambdas,r,q,thresholds)
        max_spans[b]=p.max_auc_span
        raw_min[b]=p.aggregate_regret_min_raw
        for t in thresholds:
            v=p.material_budget_by_auc_span[str(float(t))]
            if v is not None:
                delta[str(float(t))][b]=v

    def ci(x):
        lo,hi=np.quantile(np.asarray(x,float),[.025,.975])
        return [float(lo),float(hi)]

    out={
        "bootstrap_reps":int(reps),
        "seed":int(seed),
        "max_auc_span_ci95":ci(max_spans),
        "aggregate_regret_min_raw_ci95":ci(raw_min),
        "material_budget":{},
    }
    for t in thresholds:
        key=str(float(t)); x=delta[key]; finite=np.isfinite(x)
        row={
            "threshold_auc":float(t),
            "finite_fraction":float(np.mean(finite)),
            "conditional_finite_ci95":(
                ci(x[finite]) if np.any(finite) else None
            ),
        }
        out["material_budget"][key]=row
    return out


def analyse_v5_result(
    result:Mapping,
    *,
    thresholds=DEFAULT_THRESHOLDS,
    reps:int=DEFAULT_BOOTSTRAPS,
    seed:int=DEFAULT_SEED,
    strides=DEFAULT_STRIDES,
)->dict:
    paths=extract_v5_paths(result)
    axes={}
    for axis,regret_rows in paths["regret_axes"].items():
        point,env=_axis_point(
            paths["lambdas"],regret_rows,paths["auc"],thresholds,axis
        )
        row={
            "point":asdict(point),
            "envelope":env,
        }
        if axis=="raw_nll":
            row["bootstrap"]=bootstrap_frontier(
                paths["lambdas"],regret_rows,paths["auc"],
                thresholds=thresholds,reps=reps,seed=seed,
            )
            row["grid_sensitivity"]=_grid_sensitivity(
                paths["lambdas"],regret_rows,paths["auc"],
                thresholds,strides,
            )
        axes[axis]=row

    return {
        "schema":"defaultreg_v5_material_fragility_frontier_v1",
        "dataset":paths["dataset"],
        "n_participants":len(paths["subjects"]),
        "subjects":list(paths["subjects"]),
        "primary_axis":"raw_nll",
        "primary_material_threshold_auc":0.02,
        "secondary_thresholds_auc":[0.01,0.05],
        "generic_auc_zero_crossing_field_used":False,
        "paired_participant_bootstrap_across_lambda":True,
        "target_path_used_for_selection":False,
        "axes":axes,
        "status":"POST_SCORE_CORRECTED_FRONTIER_FROM_FROZEN_V5_PATHS",
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--bootstrap-reps",type=int,default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--seed",type=int,default=DEFAULT_SEED)
    a=ap.parse_args()
    result=json.loads(a.input.read_text())
    out=analyse_v5_result(
        result,reps=a.bootstrap_reps,seed=a.seed
    )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    raw=out["axes"]["raw_nll"]
    print(json.dumps({
        "dataset":out["dataset"],
        "n_participants":out["n_participants"],
        "max_auc_span":raw["point"]["max_auc_span"],
        "max_auc_span_ci95":raw["bootstrap"]["max_auc_span_ci95"],
        "material_budget":raw["point"]["material_budget_by_auc_span"],
        "material_budget_uncertainty":raw["bootstrap"]["material_budget"],
    },indent=2))


if __name__=="__main__":
    main()
