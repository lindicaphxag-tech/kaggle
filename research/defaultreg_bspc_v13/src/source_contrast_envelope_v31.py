"""Source-only spectral contrast envelope for EEG shrinkage (RESEARCH DRAFT).

The conditional robustness bound requires shared Gaussian within-class
covariance and a target class contrast inside a specified ellipsoid.
Empirical source-group scatter does NOT establish target coverage.
No target data may enter any of these functions.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np

@dataclass(frozen=True)
class SourceEnvelope:
    center: np.ndarray
    shape: np.ndarray
    evaluation_covariance: np.ndarray
    spectral_eigenvalues: np.ndarray
    spectral_eigenvectors: np.ndarray
    source_contrast: np.ndarray
    scale: float
    group_count: int

def _finite(x, name):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} has nonfinite entries")
    return a

def source_envelope(X: np.ndarray, y: np.ndarray, groups: Sequence,
                    *, shape_ridge: float = 0.01) -> SourceEnvelope:
    """One contrast per source recording group; no target statistics.

    Within-class noise is computed after *group and class* centering;
    cross-group contrast covariance describes source heterogeneity, not
    uncertainty of the empirical group-mean contrast.
    """
    x = _finite(X, "X")
    yy, gg = np.asarray(y), np.asarray(groups)
    if x.ndim != 2 or yy.shape != (x.shape[0],) or gg.shape != (x.shape[0],):
        raise ValueError("X [trial,feature], y and groups [trial] required")
    if x.shape[1] < 2 or not np.isin(yy, (0, 1)).all():
        raise ValueError("features >=2 and binary labels required")
    if not (np.isfinite(shape_ridge) and 0 < shape_ridge <= 1):
        raise ValueError("shape_ridge must lie in (0,1]")
    ids = np.unique(gg)
    if ids.size < 4:
        raise ValueError("need at least four distinct source recording groups")
    contrasts, residuals = [], []
    for gid in ids:
        mask = gg == gid
        a, b = x[mask & (yy == 0)], x[mask & (yy == 1)]
        if min(len(a),len(b)) < 2:
            raise ValueError("each source group needs two observations per class")
        m0, m1 = a.mean(axis=0), b.mean(axis=0)
        contrasts.append(m1-m0)
        residuals.extend([a-m0,b-m1])
    d = np.asarray(contrasts)
    center = d.mean(axis=0)
    deviations = d-center
    shape = deviations.T@deviations/(len(ids)-1)
    R = np.concatenate(residuals,axis=0)
    covariance = (R.T@R)/(len(R)-2*len(ids))
    covariance = (covariance+covariance.T)/2
    p = x.shape[1]
    level = max(float(np.trace(covariance)/p),np.finfo(float).tiny)
    covariance += np.eye(p)*(level*1e-8)
    shape += np.eye(p)*(shape_ridge*level)
    eig,basis = np.linalg.eigh(covariance)
    if eig[0] <= 0:
        raise ValueError("invalid source within-class geometry")
    return SourceEnvelope(center,shape,covariance,eig,basis,center.copy(),
                          float(eig.mean()),len(ids))

def direction(model: SourceEnvelope, lam: float) -> np.ndarray:
    if not np.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("lambda outside [0,1]")
    diag = (1-lam)*model.spectral_eigenvalues+lam*model.scale
    if diag.min() <= 64*np.finfo(float).eps*max(1,model.scale):
        raise ValueError("singular lambda candidate")
    return model.spectral_eigenvectors @ (
        (model.spectral_eigenvectors.T@model.source_contrast)/diag)

def robust_margin(model: SourceEnvelope, w: np.ndarray,
                  *, radius: float = 1.) -> dict:
    """Exact Gaussian margin infimum in the declared contrast ellipsoid.

    Ellipsoid: (d-center)^T shape^-1 (d-center) <= radius^2.
    Lower margin = (w.center-radius*sqrt(w.shape.w))/
                   sqrt(2*w.Sigma.w).
    AUC floor only applies if the target is Gaussian, has the SAME Sigma
    and its mean contrast actually belongs to the ellipsoid.
    """
    v = _finite(w,"w")
    if v.shape != model.center.shape or not np.isfinite(radius) or radius < 0:
        raise ValueError("invalid direction/radius")
    variance = float(v@model.evaluation_covariance@v)
    if variance<=0:
        raise ValueError("zero direction")
    nominal = float(v@model.center)/math.sqrt(2*variance)
    penalty = radius*math.sqrt(max(0.,float(v@model.shape@v)))/math.sqrt(2*variance)
    lower = nominal-penalty
    return {"nominal_probit":nominal,"shift_penalty_probit":penalty,
            "worst_case_probit":lower,
            "conditional_auc_floor":.5*math.erfc(-lower/math.sqrt(2)),
            "not_target_coverage_guarantee":True}

def pairwise_dominance(model: SourceEnvelope, wa: np.ndarray, wb: np.ndarray,
                       *, radius: float = 1.) -> dict:
    """Exact conditional rank ordering, with attainable worst-case witness."""
    if not np.isfinite(radius) or radius<0:
        raise ValueError("invalid radius")
    a,b = _finite(wa,"wa"),_finite(wb,"wb")
    if a.shape!=model.center.shape or b.shape!=model.center.shape:
        raise ValueError("invalid direction dimensions")
    S,B=model.evaluation_covariance,model.shape
    def norm(v):
        q=float(v@S@v)
        if q<=0: raise ValueError("zero direction")
        return v/math.sqrt(q)
    delta=norm(a)-norm(b)
    nominal=float(delta@model.center)
    sens=math.sqrt(max(0.,float(delta@B@delta)))
    worst=nominal-radius*sens
    witness=model.center-radius*(B@delta)/sens if sens>0 else model.center.copy()
    return {"nominal_margin_difference":nominal,"worst_margin_difference":worst,
            "critical_radius":nominal/sens if sens>1e-12 and nominal>0 else None,
            "certified_a_dominates_b_conditionally":bool(worst>1e-12),
            "worst_contrast_witness":witness.tolist(),
            "no_unconditional_transfer_guarantee":True}

def select_source_envelope(model: SourceEnvelope, lambdas: Sequence[float],
                           *, radius: float = 1.) -> dict:
    """Choose maximal lower Gaussian margin among PREDECLARED lambdas."""
    if not len(lambdas):
        raise ValueError("empty candidate set")
    rows=[]
    for lam in lambdas:
        lam=float(lam)
        try:
            row=robust_margin(model,direction(model,lam),radius=radius)
            rows.append({"lambda":lam,**row})
        except ValueError:
            rows.append({"lambda":lam,"infeasible":True})
    good=[r for r in rows if "worst_case_probit" in r]
    if not good:
        raise ValueError("no feasible lambda")
    winner=max(good,key=lambda r:(r["worst_case_probit"],r["lambda"]))
    return {"selected_lambda":winner["lambda"],"source_only":True,
            "assumed_radius":float(radius),
            "n_source_record_groups":model.group_count,
            "candidate_results":rows,
            "model_assumption":"shared Gaussian covariance; target contrast in source ellipsoid",
            "not_empirical_coverage":True}
