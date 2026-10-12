"""Proof contracts for 2-factor predictive log-score anatomy."""
import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from exact_prediction_shift_partition import exact_shift_partition, source_group_partition

def trial(seed=17,p=5):
    rng=np.random.default_rng(seed)
    def draw(n0,n1,shift):
        A=rng.normal(size=(n0,p))*np.linspace(.3,1.5,p)+np.array(shift)
        B=rng.normal(size=(n1,p))*np.linspace(.3,1.5,p)+np.array(shift)+np.r_[1.2,np.zeros(p-1)]
        return np.concatenate([A,B]),np.r_[np.zeros(n0,int),np.ones(n1,int)]
    return draw(92,88,[0]*p),draw(37,83,[.6]*p)

@pytest.mark.parametrize("lam",[0.,.0005,.01,.1,.4,.8,1.])
def test_exact_two_factor_identity_and_valid_source_classes(lam):
    (T,ty),(V,vy)=trial()
    r=exact_shift_partition(T,ty,V,vy,lam)
    assert np.isclose(
       r["mean_shift_total"],
       r["location_shift_auc_invariant"]+r["class_contrast_shift_density_only"],
       atol=1e-10,rtol=1e-10)
    assert r["location_shift_auc_invariant"]>=0
    assert r["class_contrast_shift_density_only"]>=0
    assert np.isclose(r["class0_fraction"],37/120)
    assert np.isclose(r["class1_fraction"],83/120)

@pytest.mark.parametrize("lam",[.001,.02,.3,1.])
def test_subject_shared_translation_changes_proper_nll_but_not_auc(lam):
    from scipy.linalg import solve
    (T,ty),(V,vy)=trial()
    x=np.array([5,-8,9,-2,7.])
    a=exact_shift_partition(T,ty,V,vy,lam)
    b=exact_shift_partition(T,ty,V+x,vy,lam)
    assert np.isclose(a["validation_class_centred_score"],
                      b["validation_class_centred_score"],atol=1e-10)
    assert b["proper_predictive_score"]>a["proper_predictive_score"]+.1
    # Source-fitted original EEG path, within-person rank AUC invariant.
    import original_frozen_runner as old
    st=old.spectral_state(T,ty)
    assert roc_auc_score(vy,old.scalar_scores(st,V,lam))==roc_auc_score(
           vy,old.scalar_scores(st,V+x,lam))
    assert np.isclose(a["class_contrast_shift_density_only"],
                      b["class_contrast_shift_density_only"],atol=1e-10)

def test_class_contrast_shift_can_be_auc_invisible_when_orthogonal():
    import original_frozen_runner as old
    n=44
    v=np.linspace(-1,1,n)
    # Equal x2 class means in the fitted TRAIN pool, so w is exactly x1.
    T=np.vstack([np.stack([-np.ones(n)+v,v],axis=1),
                 np.stack([np.ones(n)+v,v],axis=1)])
    y=np.r_[np.zeros(n,int),np.ones(n,int)]
    V=T.copy();yv=y.copy()
    V[yv==1,1]+=9.0
    score0=old.scalar_scores(old.spectral_state(T,y),T,1.0)
    score1=old.scalar_scores(old.spectral_state(T,y),V,1.0)
    assert np.allclose(score0,score1,atol=1e-10,rtol=1e-10)
    a=exact_shift_partition(T,y,T,y,1.)
    b=exact_shift_partition(T,y,V,yv,1.)
    assert b["class_contrast_shift_density_only"]>10.
    assert np.isclose(a["validation_class_centred_score"],
                      b["validation_class_centred_score"])
    assert roc_auc_score(y,score0)==roc_auc_score(y,score1)==1.

def test_invalid_source_contract_fail_closed():
    (T,y),(V,v)=trial()
    with pytest.raises(ValueError):exact_shift_partition(T,y,V,v,.5+float("nan"))
    with pytest.raises(ValueError):exact_shift_partition(T,y,V,v,1.1)
    with pytest.raises(ValueError):exact_shift_partition(T,np.ones(len(y)),V,v,.5)
    with pytest.raises(ValueError):exact_shift_partition(T,y,V,np.zeros(len(v)),.5)

def test_real_original_pooled_source_cv_parity_on_miniature_spd():
    import original_frozen_runner as old
    rng=np.random.default_rng(729)
    C=[];y=[];g=[]
    for subject in range(9):
        for klass in [0,1]:
            for t in range(13):
                A=rng.normal(size=(3,3))*.04+np.diag(
                  [2.3+klass*.2,1.3+subject*.08,.9])
                C.append(A@A.T)
                y.append(klass);g.append(subject+1)
    C=np.asarray(C);y=np.asarray(y);g=np.asarray(g)
    path=old.source_path(C,y,g)
    sub=source_group_partition(C,y,g,path["feasible_full_lambdas"])
    assert sub["source_group_count"]==9
    assert sub["selection_summary"]["source_macro_auc"]==path["lambda_source_macro"]
    assert sub["selection_summary"]["historical_class_centered"]==path["lambda_covariance_full"]
    for row in sub["rows"]:
        oldrow=path["criteria_by_lambda"][str(row["lambda"])]
        assert np.isclose(row["source_macro_auc"],oldrow["source_macro_auc"],atol=1e-12)
        assert np.isclose(row["class_centred_mean"],oldrow["covariance_mean_nll"],
                          atol=1e-10,rtol=1e-10)
        assert np.isclose(row["proper_predictive_mean"]-row["class_centred_mean"],
                          row["auc_invariant_translation_contribution"]+
                          row["contrast_shift_contribution"],atol=1e-10,rtol=1e-10)
