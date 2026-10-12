"""Exact partition of source-trained predictive Gaussian log-score shift.

Predictive NLL - held-source CLASS-CENTRED covariance score =
  weighted common-location shift + weighted CLASS-CONTRAST shift.

The first part can change while target-person ROC-AUC is unchanged for
every fixed linear classifier. The second part is still only a density
discrepancy: it is NOT sufficient for target AUC impact.

All held groups in this module refer to SOURCE-validation groups, NOT
an unknown target person. Intended for honest explanatory mechanisms,
not for target-derived model selection.
"""
from __future__ import annotations
import numpy as np

def _arrays(Xtrain,ytrain,Xval,yval):
    T=np.asarray(Xtrain,dtype=float);V=np.asarray(Xval,dtype=float)
    a=np.asarray(ytrain);b=np.asarray(yval)
    if (T.ndim!=2 or V.ndim!=2 or T.shape[1]!=V.shape[1]
        or a.shape!=(len(T),) or b.shape!=(len(V),)
        or not np.isfinite(T).all() or not np.isfinite(V).all()):
        raise ValueError("invalid aligned source training/validation features")
    if (not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all()
        or min(np.bincount(a.astype(int),minlength=2))<2
        or min(np.bincount(b.astype(int),minlength=2))<2):
        raise ValueError("require both binary classes in every source subset")
    return T,V,a.astype(int),b.astype(int)

def exact_shift_partition(Xtrain,ytrain,Xval,yval,lambda_:float):
    """Return exact fixed-covariance decomposition with no held TARGET IO.

    Reproduces original_frozen_runner spectral pooled-LDA regularisation:
      S=(sum class-centred residual outer products)/(N_train-2)
      A_lambda=(1-lambda)*S+lambda*(trace(S)/p)*I
    NLL omits fixed p log(2pi) and has 1/p scaling as in BSPC paper.
    """
    T,V,a,b=_arrays(Xtrain,ytrain,Xval,yval)
    lam=float(lambda_)
    if not np.isfinite(lam) or not 0<=lam<=1:
        raise ValueError("lambda must be finite in [0,1]")
    p=T.shape[1]
    mt=np.stack([T[a==c].mean(0) for c in [0,1]])
    mv=np.stack([V[b==c].mean(0) for c in [0,1]])
    R=np.concatenate([T[a==c]-mt[c] for c in [0,1]],axis=0)
    S=(R.T@R)/(len(T)-2)
    eig,U=np.linalg.eigh((S+S.T)/2)
    nu=max(float(np.mean(eig)),np.finfo(float).eps)
    spec=(1-lam)*eig+lam*nu
    tol=64*np.finfo(float).eps*max(1.,nu,float(np.max(abs(eig))))
    if not np.isfinite(spec).all() or np.any(spec<=tol):
        raise ValueError("singular shrinkage candidate")

    def quad(v):
        return float(np.sum(((U.T@v)**2)/spec))
    def quadratic_rows(resid):
        projection=resid@U
        return float(np.mean(np.sum(projection**2/spec,axis=1)))
    common=float(np.log(spec).sum())
    centered=(common+quadratic_rows(V-mv[b]))/p
    predictive=(common+quadratic_rows(V-mt[b]))/p
    delta=mv-mt
    pi=np.array([(b==c).mean() for c in [0,1]])
    u=pi[0]*delta[0]+pi[1]*delta[1]
    v=delta[1]-delta[0]
    translation=quad(u)/p
    contrast=float(pi[0]*pi[1])*quad(v)/p
    # The contrast penalty itself splits exactly into the component
    # along the ACTUAL source-trained LDA score and a Q-orthogonal
    # component that cannot shift fixed-score class-mean separation.
    # Q=A_lambda^-1 and w=Q*d_train.
    d=mt[1]-mt[0]
    w=U@((U.T@d)/spec)
    discriminator_norm=float(d@w)
    decision_aligned=(
        float(pi[0]*pi[1])*(float(v@w)**2)/discriminator_norm/p
        if discriminator_norm>1e-12 else 0.0
    )
    orthogonal=contrast-decision_aligned
    if orthogonal < -1e-8*max(1.,contrast):
        raise AssertionError("decision projection Cauchy-Schwarz violated")
    orthogonal=max(0.,orthogonal)
    unaggregated=(pi[0]*quad(delta[0])+pi[1]*quad(delta[1]))/p
    residual=predictive-centered
    if abs(contrast-decision_aligned-orthogonal)>1e-8*(1+abs(contrast)):
        raise AssertionError('contrast-alignment partition failed')
    if (abs(residual-unaggregated)>1e-8*(1+abs(predictive))
        or abs(residual-translation-contrast)>1e-8*(1+abs(predictive))):
        raise AssertionError("weighted Gaussian shift factorisation failed")
    return {
        "lambda":lam,"class0_fraction":float(pi[0]),
        "class1_fraction":float(pi[1]),"validation_n":len(V),
        "validation_class_centred_score":float(centered),
        "proper_predictive_score":float(predictive),
        "mean_shift_total":float(residual),
        "location_shift_auc_invariant":float(translation),
        "class_contrast_shift_density_only":float(contrast),
        "class_contrast_shift_score_aligned":float(decision_aligned),
        "class_contrast_shift_score_orthogonal":float(orthogonal),
        "contrast_fraction_of_mean_shift":(
            float(contrast/residual) if residual>1e-12 else None),
        "not_held_target_validation":True,
        "contrast_magnitude_is_not_guaranteed_auc_change":True,
    }

def source_group_partition(C,y,groups,lambdas):
    """Exact same 5 source participant folds and original LDA tangent path.

    Returns descriptive SOURCE-only 27-point scores, no held target data.
    The contrast-only score is an exploratory source diagnostic,
    NOT the true predictive likelihood, and not a proven new selector.
    """
    from pathlib import Path
    import sys
    from sklearn.metrics import roc_auc_score
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/"research"/"defaultreg_proper_predictive"))
    import original_frozen_runner as o
    C=np.asarray(C);y=np.asarray(y);groups=np.asarray(groups)
    if (C.ndim!=3 or len(C)!=len(y) or len(C)!=len(groups)
        or len(np.unique(groups))<8):
        raise ValueError("source group covariance/fold mismatch")
    people=sorted(np.unique(groups).tolist())
    candidates=tuple(float(x) for x in lambdas)
    if not candidates or any(not 0<=x<=1 for x in candidates):
        raise ValueError("invalid lambda grid")
    cache={lam:[] for lam in candidates}
    valid={lam:True for lam in candidates}
    for held in [people[i::5] for i in range(5)]:
        vmask=np.isin(groups,held);tmask=~vmask
        inv=o.reference(C[tmask])
        Xt=o.tangent(C[tmask],inv)
        Xv=o.tangent(C[vmask],inv)
        st=o.spectral_state(Xt,y[tmask])
        for subject in held:
            m=(groups[vmask]==subject)
            yy=y[vmask][m];xx=Xv[m]
            for lam in candidates:
                if not valid[lam]:continue
                try:
                    row=exact_shift_partition(Xt,y[tmask],xx,yy,lam)
                    row["group_id"]=int(subject)
                    if not np.isclose(
                         row["validation_class_centred_score"],
                         o.predictive_nll(st,o.moments(xx,yy)[3],lam),
                         atol=1e-10,rtol=1e-10):
                        raise AssertionError("historical score not original")
                    row["held_source_auc"]=float(roc_auc_score(
                         yy,o.scalar_scores(st,xx,lam)))
                    cache[lam].append(row)
                except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                    valid[lam]=False
    out=[]
    for lam in candidates:
        rows=cache[lam]
        if not valid[lam] or len(rows)!=len(people):
            continue
        def mean(k):
            return float(np.mean([r[k] for r in rows]))
        out.append({
            "lambda":lam,
            "source_macro_auc":mean("held_source_auc"),
            "class_centred_mean":mean("validation_class_centred_score"),
            "proper_predictive_mean":mean("proper_predictive_score"),
            "contrast_only_diagnostic":mean("validation_class_centred_score")+
                mean("class_contrast_shift_density_only"),
            "source_mean_shift_gap":mean("mean_shift_total"),
            "auc_invariant_translation_contribution":mean("location_shift_auc_invariant"),
            "contrast_shift_contribution":mean("class_contrast_shift_density_only"),
            "contrast_shift_score_aligned_contribution":mean("class_contrast_shift_score_aligned"),
            "contrast_shift_score_orthogonal_contribution":mean("class_contrast_shift_score_orthogonal"),
            "per_source_group":rows,
        })
    if not out:
        raise ValueError("no feasible source candidate")
    def minlambda(k):
        r=min(out,key=lambda v:(v[k],v["lambda"]))
        return float(r["lambda"])
    return {
      "source_group_count":len(people),
      "lambdas":[row["lambda"] for row in out],
      "selection_summary":{
        "source_macro_auc":max(out,key=lambda v:(v["source_macro_auc"],v["lambda"]))["lambda"],
        "historical_class_centered":minlambda("class_centred_mean"),
        "proper_predictive":minlambda("proper_predictive_mean"),
        "contrast_only_diagnostic":minlambda("contrast_only_diagnostic"),
      },
      "rows":out,
      "exploratory_post_prior_target_exposure":True,
      "no_held_target_data_used":True,
      "not_new_algorithm_performance_evidence":True,
    }
