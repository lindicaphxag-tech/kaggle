"""Paper 1: exact finite-sample falsification counterexample.

The proper predictive Gaussian mean shift can vary arbitrarily while
a source-trained LINEAR classifier's fixed within-person ROC ranking
is unchanged. This is a construction in a fixed feature coordinate
system, NOT a new prospective EEG discovery or universal statement
about refitting an EEG tangent reference. No target model tuning.
"""
from __future__ import annotations
import json
import numpy as np


def two_class_auc(labels, scores):
    labels, scores=np.asarray(labels), np.asarray(scores)
    p=scores[labels==1]; n=scores[labels==0]
    if min(len(p),len(n))==0:raise ValueError("need two classes")
    d=p[:,None]-n[None,:]
    return float(np.mean((d>0)+.5*(d==0)))


def shift_factors(X, y, train_m, A):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
    A=np.asarray(A,dtype=float);train_m=np.asarray(train_m,dtype=float)
    p=X.shape[1]
    if train_m.shape!=(2,p) or A.shape!=(p,p):raise ValueError("shape")
    Q=np.linalg.inv(A);d=train_m[1]-train_m[0];w=Q@d
    z=float(d@w)
    if z<=0:raise ValueError("degenerate trained discriminant")
    pi=np.array([(y==0).mean(),(y==1).mean()])
    m=np.array([X[y==0].mean(0),X[y==1].mean(0)])
    delta=m-train_m
    centered=X-m[y];predictive=X-train_m[y]
    cent=float(np.mean(np.einsum("ij,jk,ik->i",centered,Q,centered)))/p
    pred=float(np.mean(np.einsum("ij,jk,ik->i",predictive,Q,predictive)))/p
    u=pi@delta
    Gamma=delta[1]-delta[0]
    loc=float(u@Q@u)/p
    aligned=float(pi.prod()*(Gamma@w)**2/z)/p
    orth=float(pi.prod()*(Gamma@Q@Gamma-(Gamma@w)**2/z))/p
    terms=np.array([loc,orth,aligned])
    if terms.min()<-1e-10 or abs(pred-cent-terms.sum())>1e-10:
        raise AssertionError("finite-sample weighted three-factor identity failure")
    auc=two_class_auc(y,X@w)
    return {"source_fixed_linear_auc":auc,
            "proper_minus_centered_quadratic_score":pred-cent,
            "common_location":loc,
            "score_orthogonal":orth,
            "score_aligned":aligned,
            "fixed_auc_invariant_fraction":
              (loc+orth)/max(pred-cent,1e-15)}


def demo():
    train_m=np.array([[0.,0.],[2.,0.]])
    A=np.diag([1.5,1.])
    residual=np.array([[-.3,-.1],[-.1,.2],[.1,-.2],[.3,.1]])
    y=np.array([0]*4+[1]*4)
    X=np.vstack([residual+train_m[0],residual+train_m[1]])
    shifts={
        "baseline":X,
        "common_shift_orthogonal_to_score":X+np.array([0.,8.]),
        "class_specific_score_orthogonal_shift":
          X+np.where(y[:,None]==1,np.array([0.,5.]),np.array([0.,-5.])),
        "both_location_and_class_orthogonal":
          X+np.array([0.,8.])+
          np.where(y[:,None]==1,np.array([0.,5.]),np.array([0.,-5.])),
    }
    result={k:shift_factors(xx,y,train_m,A) for k,xx in shifts.items()}
    aucs=[q["source_fixed_linear_auc"] for q in result.values()]
    if not np.allclose(aucs,aucs[0],atol=0,rtol=0):
        raise AssertionError("controlled rank-invariance demo failed")
    if result["both_location_and_class_orthogonal"]["score_aligned"]>1e-12:
        raise AssertionError("constructed nuisance change has score alignment")
    if not result["common_shift_orthogonal_to_score"]["common_location"]>result["baseline"]["common_location"]+10:
        raise AssertionError("failure to expose unbounded density-score drift")
    return {"scope":"synthetic counterexample, not independent EEG experiment",
            "no_target_trained_model":True,"all_examples_fixed_auc_identical":True,
            "results":result}


if __name__=="__main__":
    print(json.dumps(demo(),indent=2,sort_keys=True,allow_nan=False))
