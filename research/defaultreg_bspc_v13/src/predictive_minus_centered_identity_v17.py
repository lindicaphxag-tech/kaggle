"""Exact source-only decomposition of proper predictive vs class-centred NLL.

For fixed *source-trained* SPD covariance A_lambda, held-source validation
labels y, source-trained class means mu_tr and held-source class means mu_val:

 C_predictive - C_centered =
    sum_c n_c (mu_val,c - mu_tr,c)^T A_lambda^-1
                    (mu_val,c - mu_tr,c) / (n * p) >= 0.

This is an EXACT identity, not a novel EEG decoder, target-safe prediction,
or a claim that minimising either criterion improves test AUC. It explains
which class-conditional between-person mean-shift term the old frozen
covariance criterion deletes by re-centring each validation source person.

No held outer target data is accepted or used.
"""
from __future__ import annotations
import numpy as np


def source_mean_shift_gap(X_val, y_val, mu_source, A):
    X = np.asarray(X_val,float)
    y = np.asarray(y_val,int)
    mu = np.asarray(mu_source,float)
    A = np.asarray(A,float)
    if (X.ndim!=2 or y.shape!=(len(X),) or mu.shape!=(2,X.shape[1]) or
        A.shape!=(X.shape[1],X.shape[1]) or
        not np.isfinite(X).all() or not np.isfinite(mu).all() or
        not np.isfinite(A).all() or
        set(np.unique(y))!={0,1}):
        raise ValueError("Finite 2-class held-SOURCE validation and source-trained means needed")
    if not np.allclose(A,A.T,atol=1e-10,rtol=0):
        raise ValueError("non symmetric covariance")
    if np.linalg.eigvalsh(A).min() <= 1e-12*np.linalg.eigvalsh(A).max():
        raise ValueError("SPD source covariance required")
    mu_v=np.array([X[y==c].mean(axis=0) for c in (0,1)])
    res_c=X-mu_v[y]
    res_p=X-mu[y]
    n,p=X.shape
    def score(r):
        return float((np.linalg.slogdet(A)[1]+
            np.mean(np.einsum('nd,nd->n',r,np.linalg.solve(A,r.T).T)))/p)
    c_cent=score(res_c)
    c_pred=score(res_p)
    delta=mu_v-mu
    weights=np.array([np.sum(y==0),np.sum(y==1)],float)
    q=np.einsum('cd,cd->c',delta,np.linalg.solve(A,delta.T).T)
    penalty=float((weights@q)/(n*p))
    tol=1e-9*max(1.,abs(c_pred),abs(c_cent),abs(penalty))
    if abs(c_pred-c_cent-penalty)>tol or penalty < -tol:
        raise AssertionError("Exact residual-mean decomposition failed")
    return {
        "source_val_centered_covscore":c_cent,
        "source_train_mean_predictive_score":c_pred,
        "nonnegative_between_source_person_class_mean_penalty":penalty,
        "per_class_weighted_mean_shift_penalty":(weights*q/(n*p)).tolist(),
        "n_held_source_trials":n,
        "class_sizes":weights.astype(int).tolist(),
        "identical_source_covariance_used":True,
        "status":"EXACT_ALGEBRA_NOT_NEW_EEG_OUTCOME"
    }


def synthetic_checks():
    rng=np.random.default_rng(20261012)
    checked=0
    for _ in range(300):
        p=int(rng.integers(2,9))
        M=rng.normal(size=(p,p))
        S=M@M.T + np.eye(p)*.4
        X=rng.normal(size=(30,p))
        y=np.repeat([0,1],15)
        mu=rng.normal(size=(2,p))
        x=source_mean_shift_gap(X,y,mu,S)
        assert x["nonnegative_between_source_person_class_mean_penalty"]>=-1e-9
        # Shift exactly each validation class to the source-trained mean:
        # then class-centred and proper predictive scores coincide.
        target_mu=np.array([X[y==k].mean(0) for k in (0,1)])
        Y=X+(mu-target_mu)[y]
        z=source_mean_shift_gap(Y,y,mu,S)
        assert abs(z["nonnegative_between_source_person_class_mean_penalty"])<1e-9
        checked+=1
    return {"status":"PASS_EXACT_SOURCE_CLASS_MEAN_SHIFT_DECOMPOSITION",
            "random_spd_cases":checked,"raw_target_data_access":False}


if __name__=="__main__":
    import json
    print(json.dumps(synthetic_checks(),indent=2))
