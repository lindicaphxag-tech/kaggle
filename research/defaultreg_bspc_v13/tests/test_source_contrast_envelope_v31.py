"""Tests for V31 source-only spectral envelope; no EEG download or target data."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from source_contrast_envelope_v31 import (
    source_envelope,direction,robust_margin,pairwise_dominance,
    select_source_envelope)

def synthetic(seed=11):
    rng=np.random.default_rng(seed)
    X=[];y=[];g=[]
    for i in range(8):
        delta=np.array([1.5,.1])+rng.normal(0,.35,2)
        for label in [0,1]:
            z=rng.normal(0,1,(35,2))*[.5,2.0]+(label-.5)*delta
            X.append(z);y.extend([label]*35);g.extend([i]*35)
    return np.concatenate(X),np.asarray(y),np.asarray(g)

def test_envelope_source_only_and_scaling():
    m=source_envelope(*synthetic())
    a=select_source_envelope(m,[0.,.1,.5,1.],radius=1.)
    assert a["selected_lambda"] in [0,.1,.5,1.]
    assert a["n_source_record_groups"]==8 and a["source_only"]
    w=direction(m,.5)
    p=robust_margin(m,w)
    assert np.isclose(p["worst_case_probit"],
                      robust_margin(m,5*w)["worst_case_probit"])
    assert robust_margin(m,w,radius=2)["worst_case_probit"]<p["worst_case_probit"]

def test_same_original_lda_shrinkage_path():
    X,y,g=synthetic()
    m=source_envelope(X,y,g)
    a=X[y==0];b=X[y==1]
    R=np.concatenate([a-a.mean(0),b-b.mean(0)],axis=0)
    S=R.T@R/(len(X)-2)
    nu=np.trace(S)/S.shape[0]
    for lam in [.01,.1,.6,1.]:
        original=np.linalg.solve((1-lam)*S+lam*nu*np.eye(S.shape[0]),
                                 b.mean(0)-a.mean(0))
        assert np.allclose(direction(m,lam),original,rtol=1e-10,atol=1e-10)

def test_worst_witness_is_attainable():
    m=source_envelope(*synthetic())
    a,b=direction(m,.1),direction(m,.9)
    z=pairwise_dominance(m,a,b,radius=1.1)
    x=np.array(z["worst_contrast_witness"])-m.center
    assert np.isclose(x@np.linalg.solve(m.shape,x),1.1**2,atol=1e-7)
    u=lambda w:w/np.sqrt(w@m.evaluation_covariance@w)
    assert np.isclose((u(a)-u(b))@np.array(z["worst_contrast_witness"]),
                      z["worst_margin_difference"])

def test_invalid_input_fail_closed():
    X,y,g=synthetic()
    with pytest.raises(ValueError):source_envelope(X,y,g[:50])
    with pytest.raises(ValueError):source_envelope(X,y,np.zeros(len(g)))
    with pytest.raises(ValueError):source_envelope(X,y,g,shape_ridge=0)
    m=source_envelope(X,y,g)
    with pytest.raises(ValueError):direction(m,1.01)
    with pytest.raises(ValueError):select_source_envelope(m,[])
    with pytest.raises(ValueError):robust_margin(m,direction(m,.5),radius=-1)

def test_unconditional_target_counterexample():
    m=source_envelope(*synthetic())
    a,b=direction(m,.1),direction(m,.9)
    delta=(a/np.sqrt(a@m.evaluation_covariance@a)-
           b/np.sqrt(b@m.evaluation_covariance@b))
    assert np.linalg.norm(delta)>1e-6
    assert (delta@delta)>0 and (delta@(-delta))<0
    assert pairwise_dominance(m,a,b)["no_unconditional_transfer_guarantee"]
