"""Source group-invariant LDA: exact nuisance invariance and source CV parity."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from source_group_invariant_lda import fit_group_state,vector,predict,source_group_cv

def sample(seed=8,G=9,trials=27):
    r=np.random.default_rng(seed)
    x=[];y=[];ids=[]
    for g in range(G):
        b=r.normal(0,3,4)
        for label in (0,1):
            z=r.normal(0,[1,2,1,.4],size=(trials,4))
            z=z+b+label*np.array([.7,.9,.3,0.])
            x.append(z);y.extend([label]*trials);ids.extend([g]*trials)
    return np.vstack(x),np.array(y),np.array(ids)

@pytest.mark.parametrize("lam",[0.0001,.001,.1,.5,1.])
def test_operator_exact_translation_invariance_for_each_source_subject(lam):
    X,y,g=sample()
    a=fit_group_state(X,y,g)
    r=np.random.default_rng(77)
    shifts=r.normal(0,50,(9,4))
    Xt=X+shifts[g]
    b=fit_group_state(Xt,y,g)
    assert np.allclose(a.W_group_class,b.W_group_class,rtol=1e-12,atol=1e-10)
    assert np.allclose(a.d_balanced,b.d_balanced,rtol=1e-12,atol=1e-11)
    assert np.allclose(vector(a,lam,"gid"),vector(b,lam,"gid"),rtol=1e-10,atol=1e-11)
    assert not np.allclose(a.S_pooled,b.S_pooled)
    assert not np.allclose(vector(a,lam,"original"),vector(b,lam,"original"),
                           rtol=1e-4,atol=1e-5) if lam<1 else True

@pytest.mark.parametrize("lam",[.01,.2,1.])
def test_original_exact_pooled_class_centered_equivalence(lam):
    import original_frozen_runner as old
    X,y,g=sample()
    m=fit_group_state(X,y,g)
    d=old.spectral_state(X,y)
    assert np.allclose(m.S_pooled,old.moments(X,y)[0],atol=1e-12)
    expected=old.scalar_scores(d,X,lam)
    got=predict(m,X,lam,"original")
    assert np.allclose(got,expected,atol=1e-10,rtol=1e-10)

def test_true_anisotropy_changes_direction_without_alpha_endpoint():
    X,y,g=sample()
    m=fit_group_state(X,y,g)
    wa=vector(m,.1,"original")
    wb=vector(m,.1,"gid")
    cosine=(wa@wb)/(np.linalg.norm(wa)*np.linalg.norm(wb))
    assert cosine<1-1e-5,"new method actually changes classifier direction"
    assert abs(m.trace_between_fraction)>1e-3

def test_group_center_covariance_genuine_class_balanced_unbalanced_design():
    rng=np.random.default_rng(42)
    X=[];Y=[];G=[]
    for g in range(10):
        loc=rng.normal(0,5,3)
        for cls in (0,1):
            count=9+g*2 if cls==0 else 45-g*2
            z=rng.normal(size=(count,3))+loc+cls*np.array([1.0,.4,0.])
            X.append(z);Y.extend([cls]*count);G.extend([g]*count)
    x=np.vstack(X);y=np.array(Y);grp=np.array(G)
    m=fit_group_state(x,y,grp)
    shift=rng.normal(0,30,(10,3))
    mm=fit_group_state(x+shift[grp],y,grp)
    assert np.allclose(m.d_balanced,mm.d_balanced,atol=1e-10)
    assert np.allclose(m.W_group_class,mm.W_group_class,atol=1e-10)
    assert not np.allclose(m.d_pooled,mm.d_pooled)

def test_group_source_cv_exact_original_lambda_selection_on_spd_covs():
    import original_frozen_runner as old
    r=np.random.default_rng(32)
    covs=[];y=[];g=[]
    for person in range(9):
        for cls in (0,1):
            for _ in range(13):
                A=r.normal(size=(3,3))*.08+np.diag([1.4+.4*cls,1.1+.03*person,.7])
                covs.append(A@A.T)
                y.append(cls);g.append(person+1)
    C=np.asarray(covs);y=np.asarray(y);g=np.asarray(g)
    oldpath=old.source_path(C,y,g)
    new=source_group_cv(C,y,g,oldpath["feasible_full_lambdas"])
    assert new["source_only"] and new["source_n_groups"]==9
    assert new["decisions"]["original"]["lambda"]==oldpath["lambda_source_macro"]
    assert abs(new["decisions"]["original"]["source_macro_auc"]-
       oldpath["criteria_by_lambda"][str(oldpath["lambda_source_macro"])]["source_macro_auc"])<1e-12
    assert len(new["paired_source_deltas"])==9
    assert len(new["all_cv_candidate_rows"])>=len(oldpath["feasible_full_lambdas"])*4

def test_bad_group_design_and_invalid_matrix_fail_closed():
    X,y,g=sample()
    with pytest.raises(ValueError):fit_group_state(X,np.ones_like(y),g)
    with pytest.raises(ValueError):fit_group_state(X,y,np.zeros_like(g))
    with pytest.raises(ValueError):fit_group_state(X,y,g[:4])
    m=fit_group_state(X,y,g)
    with pytest.raises(ValueError):vector(m,-.1,"gid")
    with pytest.raises(ValueError):vector(m,.1,"novel_unapproved")
