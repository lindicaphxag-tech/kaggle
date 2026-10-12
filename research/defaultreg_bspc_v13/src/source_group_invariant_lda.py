"""Source-group translation-invariant LDA with exact original-LDA ablation.

Research prototype. This is classical within-group-centred discriminant
analysis, not a claimed novel mathematical primitive. Source-group
translation invariance is exact in the FIXED feature coordinate system
(e.g. a source-fitted Riemannian tangent representation). It is NOT
invariance to physically translated EEG voltages before covariance
features or to re-learning a source tangent reference.

Inputs contain ONLY SOURCE trial features, class labels, and group IDs.
No target feature statistics, target labels or target geometry allowed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass(frozen=True)
class SourceGroupState:
    S_pooled: np.ndarray
    W_group_class: np.ndarray
    d_pooled: np.ndarray
    d_balanced: np.ndarray
    global_mid: np.ndarray
    pooled_nu: float
    within_nu: float
    n_groups: int
    n_source_trials: int
    trace_between_fraction: float

def fit_group_state(X,y,groups):
    X=np.asarray(X,dtype=float)
    y=np.asarray(y)
    groups=np.asarray(groups)
    if (X.ndim!=2 or X.shape[1]<2 or y.shape!=(len(X),)
        or groups.shape!=(len(X),) or not np.isfinite(X).all()
        or not np.isin(y,[0,1]).all()):
        raise ValueError("invalid source features/classes/groups")
    g=np.unique(groups);G=len(g)
    if G<4: raise ValueError("at least four source groups required")
    class0=X[y==0];class1=X[y==1]
    if min(len(class0),len(class1))<4:
        raise ValueError("both source classes need >=4 samples")
    m0=class0.mean(0);m1=class1.mean(0)
    pooled_res=np.concatenate([class0-m0,class1-m1])
    Sp=pooled_res.T@pooled_res/(len(X)-2)
    within=np.zeros_like(Sp)
    contrast=[]
    for group in g:
        a=X[(groups==group)&(y==0)]
        b=X[(groups==group)&(y==1)]
        if min(len(a),len(b))<4:
            raise ValueError("every source group needs >=4 trials/class")
        ma=a.mean(0);mb=b.mean(0)
        within+=(a-ma).T@(a-ma)+(b-mb).T@(b-mb)
        contrast.append(mb-ma)
    W=within/(len(X)-2*G)
    Sp=(Sp+Sp.T)*.5
    W=(W+W.T)*.5
    p=X.shape[1]
    nu=max(float(np.trace(Sp)/p),np.finfo(float).eps)
    wn=max(float(np.trace(W)/p),np.finfo(float).eps)
    # Pooled covariance decomposes into within and between group terms;
    # use their raw sums with their own unbiased denominators.
    between_ss=(len(X)-2)*Sp-(len(X)-2*G)*W
    ratio=float(np.trace(between_ss)/max(np.trace((len(X)-2)*Sp),1e-15))
    if ratio < -1e-8 or ratio>1+1e-8:
        raise AssertionError("source covariance decomposition inconsistent")
    return SourceGroupState(Sp,W,m1-m0,np.mean(contrast,axis=0),
                            (m0+m1)/2,nu,wn,G,len(X),ratio)

def vector(st,lam,operator="gid"):
    lam=float(lam)
    if not np.isfinite(lam) or not 0<=lam<=1:
        raise ValueError("lambda outside [0,1]")
    if operator not in ("original","gid","within_only","balanced_only"):
        raise ValueError("unknown operator")
    group_cov=operator in ("gid","within_only")
    balanced=operator in ("gid","balanced_only")
    S=st.W_group_class if group_cov else st.S_pooled
    nu=st.within_nu if group_cov else st.pooled_nu
    d=st.d_balanced if balanced else st.d_pooled
    A=(1-lam)*S+lam*nu*np.eye(len(d))
    e=np.linalg.eigvalsh(A)
    tol=64*np.finfo(float).eps*max(1.,nu,float(abs(e).max()))
    if e.min()<=tol: raise ValueError("singular classifier matrix")
    return np.linalg.solve(A,d)

def predict(st,X,lam,operator="gid"):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=len(st.d_pooled) or not np.isfinite(X).all():
        raise ValueError("invalid prediction features")
    return (X-st.global_mid)@vector(st,lam,operator)

def source_group_cv(C,y,groups,lambdas):
    """Original round-robin 5-fold source-participant CV with tangent refit.

    Baseline selection here MUST exactly reproduce original_frozen_runner.
    Each classifier path is evaluated on the same held-source persons.
    """
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/"research"/"defaultreg_proper_predictive"))
    import original_frozen_runner as original
    C=np.asarray(C);y=np.asarray(y);groups=np.asarray(groups)
    ids=sorted(np.unique(groups).tolist())
    if (C.ndim!=3 or len(C)!=len(y) or len(C)!=len(groups)
        or len(ids)<8):
        raise ValueError("invalid original grouped source EEG inputs")
    lambdas=tuple(float(z) for z in lambdas)
    if (not lambdas or any(not np.isfinite(z) or z<0 or z>1 for z in lambdas)):
        raise ValueError("bad shrinkage grid")
    modes=("original","gid","within_only","balanced_only")
    totals={(m,l):[] for m in modes for l in lambdas}
    feasible={(m,l):True for m in modes for l in lambdas}
    details={(m,l):[] for m in modes for l in lambdas}
    for held in [ids[i::5] for i in range(5)]:
        val=np.isin(groups,held);train=~val
        inv=original.reference(C[train])
        Xt=original.tangent(C[train],inv)
        Xv=original.tangent(C[val],inv)
        st=fit_group_state(Xt,y[train],groups[train])
        oldst=original.spectral_state(Xt,y[train])
        for mode in modes:
            for lam in lambdas:
                if not feasible[(mode,lam)]:continue
                try:
                    sv=predict(st,Xv,lam,mode)
                    if mode=="original":
                        expected=original.scalar_scores(oldst,Xv,lam)
                        if not np.allclose(sv,expected,atol=1e-8,rtol=1e-8):
                            raise AssertionError("original pooled LDA not preserved")
                except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                    feasible[(mode,lam)]=False
                    continue
                for sid in held:
                    mask=groups[val]==sid
                    if set(np.unique(y[val][mask]))!={0,1}:
                        raise ValueError("validation participant missing class")
                    auc=float(roc_auc_score(y[val][mask],sv[mask]))
                    totals[(mode,lam)].append(auc)
                    details[(mode,lam)].append({"source_person_id":int(sid),
                                                "auc":auc})
    rows=[]
    for mode in modes:
        for lam in lambdas:
            scores=totals[(mode,lam)]
            if feasible[(mode,lam)] and len(scores)==len(ids):
                rows.append({"operator":mode,"lambda":lam,
                             "source_person_macro_auc":float(np.mean(scores)),
                             "source_person_auc_sd":float(np.std(scores,ddof=1)),
                             "held_source_group_rows":details[(mode,lam)]})
    chosen={}
    for mode in modes:
        available=[r for r in rows if r["operator"]==mode]
        if not available:raise ValueError("empty feasible operator source path "+mode)
        top=max(available,key=lambda r:(r["source_person_macro_auc"],r["lambda"]))
        chosen[mode]={"lambda":float(top["lambda"]),
                      "source_macro_auc":top["source_person_macro_auc"],
                      "source_person_auc_sd":top["source_person_auc_sd"]}
    orig_row=next(r for r in rows if r["operator"]=="original" and r["lambda"]==chosen["original"]["lambda"])
    new_row=next(r for r in rows if r["operator"]=="gid" and r["lambda"]==chosen["gid"]["lambda"])
    old_aucs={z["source_person_id"]:z["auc"] for z in orig_row["held_source_group_rows"]}
    new_aucs={z["source_person_id"]:z["auc"] for z in new_row["held_source_group_rows"]}
    delta=np.array([new_aucs[int(g)]-old_aucs[int(g)] for g in ids])
    chosen["gid"]["source_paired_mean_delta"]=float(np.mean(delta))
    chosen["gid"]["source_delta_positive_group_count"]=int((delta>1e-12).sum())
    chosen["gid"]["source_delta_negative_group_count"]=int((delta< -1e-12).sum())
    chosen["gid"]["source_delta_gt002_harmed"]=int((delta<-.02).sum())
    # Freeze a diagnostic source-only risk forecast; this does not become a
    # distribution-free promise about new persons.
    chosen["gid"]["source_only_forecast_positive"]=bool(
        delta.mean()>=.005 and (delta>0).sum()>=int(np.ceil(.6*len(ids)))
    )
    return {
      "source_only":True,"source_n_groups":len(ids),
      "criteria":"same grouped source macro AUC per operator, no new target input",
      "original_27grid_parity_required":True,
      "decisions":chosen,"all_cv_candidate_rows":rows,
      "paired_source_deltas":[{"source_person_id":int(k),"paired_delta":float(v)}
                               for k,v in zip(ids,delta)],
      "no_distribution_free_target_gain_guarantee":True,
      "not_claimed_novel_invention_of_within_group_LDA":True,
    }
