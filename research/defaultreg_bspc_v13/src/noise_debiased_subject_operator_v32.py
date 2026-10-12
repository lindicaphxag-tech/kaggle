"""V32: noise-debiased, non-isotropic source-contrast regularisation.

Research prototype, not a new-EEG generalisation guarantee.
All per-person contrast and covariance estimates are SOURCE-only.
The group-heterogeneity matrix is PSD projection of (between-group
contrast covariance - within-group finite-trial contrast sampling noise).
When alpha=0, this operator is *exactly* the original pooled-LDA
scalar-shrinkage path; alpha>0 may rotate the discriminant direction.

The source-only selection and all hyperparameters must be frozen before
any held target person is opened. Never use held-person signal statistics,
class labels, or AUC for lambda or alpha adaptation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass(frozen=True)
class Fitted:
    pooled_cov: np.ndarray
    d: np.ndarray
    middle: np.ndarray
    isotropic_level: float
    between_debiased_psd: np.ndarray
    n_groups: int
    noise_removed_trace: float
    between_raw_trace: float

def _input(X,y,group):
    X=np.asarray(X,dtype=np.float64)
    y=np.asarray(y)
    group=np.asarray(group)
    if X.ndim!=2 or X.shape[1]<2 or y.shape!=(len(X),) or group.shape!=(len(X),):
        raise ValueError("require [trial,feature] and two aligned 1D arrays")
    if not (np.isfinite(X).all() and np.isin(y,[0,1]).all()):
        raise ValueError("invalid features or labels")
    if len(np.unique(group))<4:
        raise ValueError("at least four independent SOURCE group IDs required")
    return X,y,group

def fit(X,y,group) -> Fitted:
    X,y,group=_input(X,y,group)
    n,p=X.shape
    # Reproduce original fixed global class-centered pooled LDA precisely.
    c0,c1=X[y==0],X[y==1]
    if min(len(c0),len(c1))<4:
        raise ValueError("invalid source class denominator")
    m0,m1=c0.mean(0),c1.mean(0)
    residual=np.concatenate((c0-m0,c1-m1),axis=0)
    S=(residual.T@residual)/(n-2)
    S=(S+S.T)*.5
    level=max(float(np.trace(S)/p),np.finfo(float).eps)
    dg=[];v_noise=[]
    for sid in np.unique(group):
        a=X[(group==sid)&(y==0)]
        b=X[(group==sid)&(y==1)]
        if min(len(a),len(b))<4:
            raise ValueError("each source group needs >=4 trials per class")
        da=b.mean(0)-a.mean(0)
        # Finite-sample uncertainty of estimated contrast, separate for
        # each source group: cov(mean1-mean0).
        v=np.cov(a,rowvar=False,ddof=1)/len(a)+np.cov(b,rowvar=False,ddof=1)/len(b)
        dg.append(da)
        v_noise.append(v)
    contrasts=np.stack(dg)
    Bobs=np.cov(contrasts,rowvar=False,ddof=1)
    V=np.mean(v_noise,axis=0)
    Bh=(Bobs-V+Bobs.T-V.T)*.5
    eigen,basis=np.linalg.eigh(Bh)
    # PSD projection truncates negative estimates; sample-size limited.
    B=(basis*np.maximum(eigen,0))@basis.T
    return Fitted(S,m1-m0,(m0+m1)/2,level,(B+B.T)*.5,
                  len(dg),float(np.trace(V)),float(np.trace(Bobs)))

def weight(model:Fitted,lam:float,alpha:float) -> np.ndarray:
    lam=float(lam);alpha=float(alpha)
    if not np.isfinite(lam) or not 0<=lam<=1 or not np.isfinite(alpha) or alpha<0:
        raise ValueError("invalid shrinkage parameters")
    p=len(model.d)
    A=(1-lam)*model.pooled_cov+lam*model.isotropic_level*np.eye(p)
    A=A+alpha*model.between_debiased_psd
    A=(A+A.T)/2
    eig=np.linalg.eigvalsh(A)
    tol=64*np.finfo(float).eps*max(1,model.isotropic_level,float(abs(eig).max()))
    if eig[0]<=tol:
        raise ValueError("infeasible singular classifier matrix")
    return np.linalg.solve(A,model.d)

def score(model:Fitted,X,lam:float,alpha:float) -> np.ndarray:
    X=np.asarray(X,dtype=np.float64)
    if X.ndim!=2 or X.shape[1]!=len(model.d) or not np.isfinite(X).all():
        raise ValueError("invalid prediction rows")
    return (X-model.middle)@weight(model,lam,alpha)

def group_cv(C,y,g,lambdas,alphas,*,beta=.25):
    """SOURCE participant-grouped CV; refit tangent and B only on fold-train.

    The original baseline is the maximal grouped source macro-AUC at alpha0.
    The candidate uses source group macro-AUC minus beta*inter-group SD;
    alpha positive introduces an anisotropic B operator. No held TARGET I/O.
    """
    if not np.isfinite(beta) or beta<0:
        raise ValueError("invalid risk coefficient")
    C=np.asarray(C)
    y=np.asarray(y);g=np.asarray(g)
    groups=sorted(np.unique(g).tolist())
    if len(groups)<8 or C.ndim!=3 or len(y)!=len(g) or len(C)!=len(y):
        raise ValueError("need >=8 source groups, aligned trial covariances")
    lambdas=tuple(float(v) for v in lambdas)
    alphas=tuple(float(v) for v in alphas)
    if len(lambdas)==0 or len(alphas)==0 or 0. not in alphas:
        raise ValueError("must have candidate lambda, alpha0 baseline")
    if not all(np.isfinite(z) and 0<=z<=1 for z in lambdas):
        raise ValueError("invalid lambda grid")
    if not all(np.isfinite(z) and z>=0 for z in alphas):
        raise ValueError("invalid alpha grid")
    # Exact existing repository grouping: round-robin 5 folds.
    folds=[groups[i::5] for i in range(5)]
    candidates=[(l,a) for l in lambdas for a in alphas]
    values={k:[] for k in candidates}
    valid={k:True for k in candidates}
    # Lazy import ensures this unit is importable without MOABB.
    from pathlib import Path
    import sys
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/"research"/"defaultreg_proper_predictive"))
    import original_frozen_runner as original
    for held in folds:
        va=np.isin(g,held);tr=~va
        inv=original.reference(C[tr])
        Xt=original.tangent(C[tr],inv)
        Xv=original.tangent(C[va],inv)
        model=fit(Xt,y[tr],g[tr])
        scores={}
        for cand in candidates:
            try:
                scores[cand]=score(model,Xv,*cand)
            except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                valid[cand]=False
        for subject in held:
            sub=(g[va]==subject)
            labels=y[va][sub]
            if set(labels.tolist())!={0,1}:
                raise ValueError("source held group missing binary class")
            for cand,v in scores.items():
                values[cand].append(float(roc_auc_score(labels,v[sub])))
    rows=[]
    for l,a in candidates:
        cand=(l,a)
        if not valid[cand] or len(values[cand])!=len(groups):
            continue
        v=np.array(values[cand])
        rows.append({"lambda":l,"alpha":a,"macro_auc":float(v.mean()),
                     "source_group_auc_sd":float(v.std(ddof=1)),
                     "risk_score":float(v.mean()-beta*v.std(ddof=1)),
                     "source_group_aucs":v.tolist()})
    if not rows or not any(q["alpha"]==0 for q in rows):
        raise ValueError("no feasible baseline in the same CV")
    baseline=max((r for r in rows if r["alpha"]==0),
                 key=lambda r:(r["macro_auc"],r["lambda"]))
    # Complexity-preserving tie breaker prioritizes alpha=0.
    risk=max(rows,key=lambda r:(r["risk_score"],-r["alpha"],r["lambda"]))
    robust=max(rows,key=lambda r:(r["macro_auc"],-r["alpha"],r["lambda"]))
    return {"source_only":True,"source_n_groups":len(groups),
            "baseline":{"lambda":baseline["lambda"],"alpha":0.,
                        "macro_auc":baseline["macro_auc"]},
            "mean_best":{"lambda":robust["lambda"],"alpha":robust["alpha"],
                         "macro_auc":robust["macro_auc"]},
            "risk_best":{"lambda":risk["lambda"],"alpha":risk["alpha"],
                         "macro_auc":risk["macro_auc"],
                         "risk_score":risk["risk_score"]},
            "beta_predeclared":float(beta),
            "n_fit_candidate_pairs":len(rows),
            "candidate_group_source_cv":rows,
            "no_target_coverage_guarantee":True}
