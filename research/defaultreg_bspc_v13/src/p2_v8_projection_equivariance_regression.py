#!/usr/bin/env python3
"""Matched-subset common-mode cancellation tests.
Pure numerical regression, NOT real EEG independent validation.
"""
import json,numpy as np
def H(X):return X-X.mean(axis=0,keepdims=True)
def corr(a,b):
    a=a-a.mean();b=b-b.mean();n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(abs(a@b/n)) if n>1e-10 else 0.
def run():
    rng=np.random.default_rng(20261015)
    passed=0
    for k in (4,5,6,8):
        for seed in range(25):
            U=rng.normal(size=(k,768)).cumsum(axis=1)
            X=U+100*rng.normal(size=(1,768))
            Y=U+100*rng.normal(size=(1,768))
            assert np.allclose(H(X),H(Y),rtol=0,atol=1e-11)
            passed+=1
    U=rng.normal(size=(8,768));order=rng.permutation(8)
    assert np.allclose(H(U[order]),H(U)[order],rtol=0,atol=1e-12)
    X=rng.normal(size=(8,768));Z=X.copy()
    Z[1]+=20*rng.normal(size=768)
    chosen=[0,2,3,5,6]
    pre=min(corr(a,b) for a,b in zip(H(X)[chosen],H(Z)[chosen]))
    subset=min(corr(a,b) for a,b in zip(H(X[chosen]),H(Z[chosen])))
    assert pre<.99 and abs(subset-1)<1e-12
    left=rng.normal(size=(8,768));right=rng.normal(size=(8,768))
    artifact=150*rng.normal(size=(1,768))
    false=min(corr(a,b) for a,b in zip(left+artifact,right+artifact))
    corrected=max(corr(a,b) for a,b in zip(H(left+artifact),H(right+artifact)))
    assert false>.999 and corrected<.15
    return {'status':'PASS','synthetic_invariance_cases':passed,
            'pre_CAR_min_corr_mismatched_unmatched':pre,
            'post_match_min_corr':subset,
            'artifact_raw_min_corr':false,
            'artifact_residual_max_corr':corrected,
            'scope':'classical matched-subset projection algebra; not physical provenance identity'}
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
