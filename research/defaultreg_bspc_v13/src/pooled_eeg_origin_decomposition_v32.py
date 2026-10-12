"""Exact pooled-EEG covariance origin decomposition, SOURCE-only research.

Class-wise pooled covariance contains BOTH within-person stochastic
variation and between-person shifts of class means; those have different
roles for held-person *within-person* ROC-AUC.
Algebra only, no target information or transfer theorem.
"""
from __future__ import annotations
import numpy as np

def split_within_person_vs_cross_person_covariance(X, y, group):
    X=np.asarray(X,dtype=float);y=np.asarray(y);group=np.asarray(group)
    if (X.ndim!=2 or y.shape!=(len(X),) or group.shape!=(len(X),)
        or len(np.unique(group))<2 or not np.isfinite(X).all()
        or set(np.unique(y).tolist())!={0,1}):
        raise ValueError("invalid source groups or classes")
    n,p=X.shape
    global_means={c:X[y==c].mean(axis=0) for c in (0,1)}
    exact_global=np.zeros((p,p))
    within=np.zeros_like(exact_global)
    between=np.zeros_like(exact_global)
    idg=np.unique(group)
    group_receipts=[]
    for c in (0,1):
        xc=X[y==c]
        exact_global+=(xc-global_means[c]).T@(xc-global_means[c])
    for gid in idg:
        for c in (0,1):
            xgc=X[(group==gid)&(y==c)]
            if len(xgc)<2:
                raise ValueError("each group/class requires >=2")
            mu=xgc.mean(axis=0)
            within+=(xgc-mu).T@(xgc-mu)
            gap=mu-global_means[c]
            between+=len(xgc)*np.outer(gap,gap)
            group_receipts.append({"group":str(gid),"class":int(c),"n":len(xgc)})
    global_cov=exact_global/(n-2)
    within_component=within/(n-2)
    between_component=between/(n-2)
    if not np.allclose(global_cov,within_component+between_component,
                       atol=1e-9,rtol=1e-9):
        raise AssertionError("ANOVA covariance identity violated")
    total_trace=float(np.trace(global_cov))
    # Importantly, between contains class-shared translation AND
    # class-specific contrast shift, not just discrimination shift.
    return {"global_covariance":global_cov,
            "source_within_group_component":within_component,
            "source_between_group_means_component":between_component,
            "trace_between_fraction":float(np.trace(between_component)/total_trace)
                if total_trace>0 else 0.0,
            "n_source_group_ids":len(idg),
            "n_trials":n,
            "no_target_transfer_identification":True,
            "receipts":group_receipts}
