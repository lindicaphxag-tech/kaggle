"""Exact covariance identity and translation-vs-AUC falsification tests."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pooled_eeg_origin_decomposition_v32 import split_within_person_vs_cross_person_covariance
from sklearn.metrics import roc_auc_score

def test_between_recording_translation_dominates_pooled_covariance_but_not_auc():
    rng=np.random.default_rng(31)
    xx=[]; yy=[]; gg=[]
    for gid in range(9):
        translation=np.array([0.,gid*8.,0.])
        for label in (0,1):
            x=rng.normal(0,.7,(40,3))+translation+label*np.array([1.,0.,0.])
            xx.append(x);yy.extend([label]*40);gg.extend([gid]*40)
    X=np.concatenate(xx);y=np.array(yy);g=np.array(gg)
    out=split_within_person_vs_cross_person_covariance(X,y,g)
    assert out["trace_between_fraction"]>.90
    S=out["global_covariance"]
    assert np.allclose(S,out["source_within_group_component"]+
                       out["source_between_group_means_component"])
    w=np.array([1.,.2,.1])
    before=np.array([roc_auc_score(y[g==id],X[g==id]@w) for id in range(9)])
    moved=X.copy()
    for id in range(9):
        moved[g==id]+=rng.normal(0,60,size=3)
    after=np.array([roc_auc_score(y[g==id],moved[g==id]@w) for id in range(9)])
    assert np.array_equal(before,after), "within-group AUC is shift-invariant"
    new=split_within_person_vs_cross_person_covariance(moved,y,g)
    assert not np.allclose(new["global_covariance"],S)

def test_invalid_group_classes_refused():
    X=np.ones((15,3))
    with pytest.raises(ValueError):
        split_within_person_vs_cross_person_covariance(X,np.zeros(15),np.arange(15))
    with pytest.raises(ValueError):
        split_within_person_vs_cross_person_covariance(X,np.arange(15)%2,np.zeros(15))
