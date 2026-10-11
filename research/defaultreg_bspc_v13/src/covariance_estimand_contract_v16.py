"""Executable, source-only estimand contract for DefaultReg's frozen NLL baseline.

IMPORTANT: The original Shin2017A runner constructs validation class-centered
residuals from held SOURCE participant labels and participant-specific class
means, then scores them against a SOURCE-fitted covariance model. This is a
label-centred covariance log score, not proper class-conditional prediction
using SOURCE-fitted class means. No held outer TARGET data enter this routine.

The methods below define both scores separately so a new prospective
comparison cannot silently reuse the frozen class-centered results.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _check(Xtr,ytr,Xval,yval):
    a,b=np.asarray(Xtr,float),np.asarray(Xval,float)
    u,v=np.asarray(ytr),np.asarray(yval)
    if (a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or
        len(u)!=len(a) or len(v)!=len(b) or
        not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Need finite 2D source-fit and held-source-validation features")
    if any(set(np.unique(labels))!={0,1} for labels in (u,v)):
        raise ValueError("Both participant groups need two labeled source classes")
    if any(sum(labels==k)<2 for labels in (u,v) for k in [0,1]):
        raise ValueError("Need >=2 source trials per class for means and residuals")
    return a,u,b,v


def _source_state(X,y):
    mu=np.stack([X[y==k].mean(0) for k in [0,1]])
    residual=X-mu[y]
    S=(residual.T@residual)/(len(X)-2)
    S=(S+S.T)/2
    nu=float(np.trace(S)/len(mu[0]))
    if nu<=0: raise ValueError("Degenerate source within-class covariance")
    return mu,S,nu


def _score(residual,A):
    e,U=np.linalg.eigh((A+A.T)/2)
    if e[0]<=np.finfo(float).eps*max(1.,e[-1]):
        raise ValueError("Covariance candidate not SPD")
    Z=np.asarray(residual,float)@U
    return float((np.log(e).sum()+np.mean(np.sum(Z*Z/e,axis=1)))/len(e))


@dataclass(frozen=True)
class ObjectivePath:
    lambda_grid:list[float]
    frozen_class_centered_covariance_score:list[float]
    genuine_label_conditional_predictive_score:list[float]
    frozen_selected_lambda:float
    predictive_selected_lambda:float
    estimand_contract:str="FROZEN_SOURCE_CLASS_CENTERED_SCORE_IS_NOT_PROPER_PREDICTIVE_NLL"
    no_outer_target_inputs_accepted:bool=True


def compare_source_validation_objectives(Xtr,ytr,Xval,yval,lambda_grid):
    X,y,V,v=_check(Xtr,ytr,Xval,yval)
    grid=np.asarray(lambda_grid,float)
    if (grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or
        (grid<0).any() or (grid>1).any() or np.any(np.diff(grid)<=0)):
        raise ValueError("Regularisation candidates must be strictly ascending in [0,1]")
    mu,S,nu=_source_state(X,y)
    # Crucial frozen runner semantics: validation subject's OWN CLASS means.
    val_mu=np.stack([V[v==k].mean(0) for k in [0,1]])
    R_center=V-val_mu[v]
    # Genuinely predictive alternative: SOURCE training class means,
    # held fixed throughout evaluation of a source-validation subject.
    R_predict=V-mu[v]
    c,d=[],[]
    for lam in grid:
        A=(1-lam)*S+lam*nu*np.eye(S.shape[0])
        c.append(_score(R_center,A))
        d.append(_score(R_predict,A))
    return ObjectivePath(grid.tolist(),c,d,
        float(grid[int(np.argmin(c))]),float(grid[int(np.argmin(d))]))


def _selftest():
    rng=np.random.default_rng(20261011)
    cov=np.array([[3.,.65],[.65,.8]])
    X=np.vstack([rng.multivariate_normal([-1.,0.],cov,100),
                 rng.multivariate_normal([1.,0.],cov,100)])
    y=np.array([0]*100+[1]*100)
    V=np.vstack([rng.multivariate_normal([-1.,0.],cov,35),
                 rng.multivariate_normal([1.,0.],cov,35)])
    v=np.array([0]*35+[1]*35)
    grid=[0.,.01,.05,.1,.25,.5,.75,1.]
    original=compare_source_validation_objectives(X,y,V,v,grid)
    shifted=V.copy()
    shifted[v==0]+=np.array([12.,0.])
    shifted[v==1]+=np.array([-10.,1.])
    changed=compare_source_validation_objectives(X,y,shifted,v,grid)
    assert np.allclose(original.frozen_class_centered_covariance_score,
                       changed.frozen_class_centered_covariance_score,rtol=0,atol=1e-12)
    assert (np.max(np.abs(np.asarray(original.genuine_label_conditional_predictive_score)-
                         np.asarray(changed.genuine_label_conditional_predictive_score))) > 1.0)
    # Reproduce the exact residual-NLL rule in original Kaggle runner.
    # It computes moments(XX,yy) => residual and predictive_nll(state,R,lam).
    for i,lam in enumerate(grid):
        _,S,nu=_source_state(X,y)
        held_center=shifted-np.stack([shifted[v==k].mean(0) for k in [0,1]])[v]
        A=(1-lam)*S+lam*nu*np.eye(S.shape[0])
        assert abs(_score(held_center,A)-changed.frozen_class_centered_covariance_score[i])<1e-12
    return {"status":"PASS_SYNTHETIC_ORIGINAL_ESTIMAND_AUDIT",
            "no_outer_target_data":True,
            "class_shift_invariant_centered_score":True,
            "class_shift_changes_true_predictive_score":True,
            "original_lambda":original.frozen_selected_lambda,
            "shifted_centered_lambda":changed.frozen_selected_lambda,
            "shifted_predictive_lambda":changed.predictive_selected_lambda}


if __name__=="__main__":
    import json
    print(json.dumps(_selftest(),indent=2))
