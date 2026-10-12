"""Falsify claims of a novel classifier when SCE saturates at lambda=1."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from source_contrast_envelope_v31 import source_envelope, direction, select_source_envelope

def test_full_shrinkage_is_plain_linear_centroid_direction():
    rng=np.random.default_rng(123)
    x=[];y=[];ids=[]
    for k in range(8):
        for label in (0,1):
            v=rng.normal(size=(25,4)) + label*np.array([1.5,0.3,-0.2,0.1])
            x.append(v);y.extend([label]*25);ids.extend([k]*25)
    X=np.concatenate(x); yy=np.array(y); groups=np.array(ids)
    m=source_envelope(X,yy,groups)
    plain=X[yy==1].mean(0)-X[yy==0].mean(0)
    exact=direction(m,1.)
    assert np.allclose(exact*m.scale,plain,atol=1e-10,rtol=1e-10)
    # For every target X, scores differ only by strictly positive scaling,
    # not a new classifier or changed ranking.
    held=rng.normal(size=(101,4))
    p=held@plain
    q=held@exact
    assert np.array_equal(np.argsort(p),np.argsort(q))
    assert np.allclose(q,p/m.scale,rtol=1e-10,atol=1e-10)
    assert abs(m.scale)>0

def test_exact_same_lambda_1_is_not_selection_novelty():
    # A separate, fixed lambda=1 comparator is MATHEMATICALLY equal to
    # rho=1 SCE on Cho first+reserved blocks because the source-selected
    # lambda was exactly 1.0, independent of held EEG and the ellipsoid.
    # This test only validates the algebra; frozen external outcome receipts
    # must be read separately to assert that selected lambda = 1.0.
    from pathlib import Path
    protocol=Path(__file__).resolve().parents[1]/"V31B_CHO2017_RESERVED_AND_POOLED40_NEGATIVE.md"
    assert protocol.is_file()
    assert "SCE-rho1=1.0" in protocol.read_text()
