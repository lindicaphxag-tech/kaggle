"""V32 mechanism unit tests: no real EEG access or unregistered target tuning."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from noise_debiased_subject_operator_v32 import fit,weight,score,group_cv

def synthetic(seed=37,n_group=9,n_each=30):
    rng=np.random.default_rng(seed)
    X=[];y=[];g=[]
    for k in range(n_group):
        mean_diff=np.array([1.2,0.6,0.2])+rng.normal(0,[.3,.7,.1],3)
        for label in (0,1):
            x=rng.normal(size=(n_each,3))*[1.,2.,.4]+label*mean_diff
            X.append(x);y.extend([label]*n_each);g.extend([k]*n_each)
    return np.concatenate(X),np.asarray(y),np.asarray(g)

def test_alpha0_identical_to_unmodified_original_lda():
    X,y,g=synthetic()
    m=fit(X,y,g)
    a,b=X[y==0],X[y==1]
    R=np.concatenate((a-a.mean(0),b-b.mean(0)),axis=0)
    S=R.T@R/(len(X)-2)
    d=b.mean(0)-a.mean(0)
    assert np.allclose(S,m.pooled_cov,rtol=1e-12,atol=1e-12)
    for lam in [0.,.005,.1,.5,1.]:
        e=np.linalg.solve((1-lam)*S+lam*np.trace(S)/3*np.eye(3),d)
        assert np.allclose(weight(m,lam,0),e,rtol=1e-11,atol=1e-11)

def test_noise_debiased_psd_and_real_anisotropic_rotation():
    m=fit(*synthetic())
    assert np.linalg.eigvalsh(m.between_debiased_psd).min()>-1e-11
    assert np.trace(m.between_debiased_psd)>1e-4
    w0=weight(m,.3,0)
    w1=weight(m,.3,3.)
    assert np.linalg.norm(w0-w1)>1e-5
    # Scale-invariant criterion for genuine non-parallel operator.
    cosine=(w0@w1)/(np.linalg.norm(w0)*np.linalg.norm(w1))
    assert cosine < 1-1e-8

def test_no_true_between_group_signal_means_no_added_operator():
    rng=np.random.default_rng(0)
    x0=rng.normal(size=(25,3))
    x1=x0+np.array([.4,1.2,.2])
    X=np.concatenate([np.vstack([x0,x1]) for _ in range(8)])
    y=np.tile(np.r_[np.zeros(25,int),np.ones(25,int)],8)
    g=np.repeat(np.arange(8),50)
    m=fit(X,y,g)
    assert np.allclose(m.between_debiased_psd,0,atol=1e-11)
    assert np.allclose(weight(m,.6,1.5),weight(m,.6,0),rtol=1e-11,atol=1e-11)

def test_bad_contracts_fail_closed():
    X,y,g=synthetic()
    m=fit(X,y,g)
    with pytest.raises(ValueError):fit(X,y,np.zeros_like(g))
    with pytest.raises(ValueError):fit(X,np.ones(len(y),int),g)
    with pytest.raises(ValueError):fit(X,y,g[:5])
    with pytest.raises(ValueError):score(m,np.ones((3,4)),.5,0)
    with pytest.raises(ValueError):weight(m,1.1,0)
    with pytest.raises(ValueError):weight(m,.5,-.1)

def test_eeg_covariance_path_matches_existing_original_cv():
    import original_frozen_runner as old
    rng=np.random.default_rng(20261013)
    Cs=[];labels=[];grp=[]
    for k in range(9):
        for label in [0,1]:
            for rep in range(14):
                A=rng.normal(0,.12,size=(3,3))
                A+=np.diag([1.5+label*.22,1+label*.1,.8])
                C=A@A.T
                Cs.append(C)
                labels.append(label);grp.append(k+1)
    C=np.asarray(Cs);y=np.asarray(labels);g=np.asarray(grp)
    grid=old.source_path(C,y,g)
    feasible=grid["feasible_full_lambdas"]
    result=group_cv(C,y,g,feasible,[0.,.5,2.],beta=.25)
    assert result["source_only"] and result["source_n_groups"]==9
    assert result["n_fit_candidate_pairs"]>=len(feasible)
    assert result["baseline"]["lambda"]==grid["lambda_source_macro"]
    assert np.isclose(result["baseline"]["macro_auc"],
                      grid["criteria_by_lambda"][str(grid["lambda_source_macro"])]["source_macro_auc"],
                      atol=1e-10,rtol=0)
    assert result["mean_best"]["macro_auc"]>=result["baseline"]["macro_auc"]-1e-12


def test_gaussian_random_contrast_expected_auc_optimum_is_anisotropic():
    """Exact conditional Gaussian model, NOT an EEG target performance claim."""
    from scipy.special import ndtr
    S=np.diag([1.,2.])
    B=np.diag([12.,.2])
    d=np.array([2.,1.])
    w_fisher=np.linalg.solve(S,d)
    w_new=np.linalg.solve(2*S+B,d)
    def expected_auc(w):
        return float(ndtr((w@d)/np.sqrt(2*w@S@w+w@B@w)))
    # Gaussian random-effect transfer model admits an *analytical*
    # optimum different from standard pooled-LDA even with known Sigma.
    assert expected_auc(w_new)>expected_auc(w_fisher)+.02
    assert np.linalg.norm(w_new/np.linalg.norm(w_new)-
                          w_fisher/np.linalg.norm(w_fisher))>.1
    rng=np.random.default_rng(313)
    draws=rng.multivariate_normal(d,B,size=40000)
    monte=float(ndtr((draws@w_new)/np.sqrt(2*w_new@S@w_new)).mean())
    assert abs(monte-expected_auc(w_new))<.008

def test_debiased_matrix_is_not_unbiased_after_psd_projection():
    """Detect the theoretical caveat: eigenclipping adds finite-G bias."""
    rng=np.random.default_rng(31)
    traces=[]
    for seed in range(18):
        X=[];y=[];g=[]
        for group in range(6):
            # Exactly common true class difference across SOURCE groups.
            for cls in [0,1]:
                x=rng.normal(0,1,(16,3))+cls*np.array([.9,.2,.3])
                X.append(x);y.extend([cls]*16);g.extend([group]*16)
        m=fit(np.concatenate(X),np.asarray(y),np.asarray(g))
        traces.append(np.trace(m.between_debiased_psd))
    assert all(v>=-1e-10 for v in traces)
    assert float(np.mean(traces))>0, "PSD clipping is biased upward in finite samples"
