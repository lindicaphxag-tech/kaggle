#!/usr/bin/env python3
"""P1 exact finite-sample estimand identity (rational synthetic witness).

No EEG file or held target labels are loaded. This checks the exact
Gaussian quadratic-score algebra behind an interpretation claim.
"""
from fractions import Fraction as Q
import json

def vadd(a,b): return tuple(x+y for x,y in zip(a,b))
def vsub(a,b): return tuple(x-y for x,y in zip(a,b))
def vscale(a,s): return tuple(x*s for x in a)
def invnorm(v, invdiag): return sum(x*x*invdiag[i] for i,x in enumerate(v))

def auc(pos,neg,w):
    def dot(a,b): return sum(x*y for x,y in zip(a,b))
    n=Q(0)
    for a in pos:
        for b in neg:
            z=dot(w,a)-dot(w,b)
            n+=(Q(1) if z>0 else Q(0) if z<0 else Q(1,2))
    return n/(len(pos)*len(neg))

def run():
    # Balanced class means with within-class residuals summing to zero.
    source={0:(Q(0),Q(0)),1:(Q(1),Q(1))}
    shifts={0:(Q(1),Q(1)),1:(Q(3),Q(-1))}
    invdiag=[Q(1,2),Q(1,3)] # positive definite diagonal precision
    residuals=[(Q(-1),Q(2)),(Q(1),Q(-2)),
               (Q(-2),Q(-1)),(Q(2),Q(1))]
    k=len(residuals)
    classmeans={c:vadd(source[c],shifts[c]) for c in [0,1]}
    predictive=Q(0); centered=Q(0)
    for c in [0,1]:
        for r in residuals:
            x=vadd(classmeans[c],r)
            predictive+=Q(1,2)*invnorm(vsub(x,source[c]),invdiag)
            centered+=Q(1,2)*invnorm(vsub(x,classmeans[c]),invdiag)
    delta=predictive-centered
    exact=Q(k,2)*sum(invnorm(shifts[c],invdiag) for c in [0,1])
    assert delta==exact and delta>0
    common=vscale(vadd(shifts[0],shifts[1]),Q(1,2))
    contrast=vscale(vsub(shifts[1],shifts[0]),Q(1,2))
    # For the special diagonal SPD precision and score w=(1,0),
    # x-axis is score-aligned; y-axis is score-orthogonal.
    aligned=(contrast[0],Q(0)); orth=(Q(0),contrast[1])
    components=[k*invnorm(common,invdiag),
                k*invnorm(orth,invdiag),
                k*invnorm(aligned,invdiag)]
    assert sum(components)==delta
    neg=[(Q(0),Q(1)),(Q(2),Q(0)),(Q(-1),Q(0))]
    pos=[(Q(1),Q(1)),(Q(3),Q(0)),(Q(-2),Q(0))]
    w=(Q(1),Q(0))
    baseline=auc(pos,neg,w)
    for common_shift in [(Q(7),Q(-20)),(Q(-3),Q(17))]:
        assert auc([vadd(x,common_shift) for x in pos],
                   [vadd(x,common_shift) for x in neg],w)==baseline
    # Arbitrary class-specific y translations are invisible to x-score.
    assert auc([vadd(x,(Q(0),Q(23))) for x in pos],
               [vadd(x,(Q(0),Q(-19))) for x in neg],w)==baseline
    # Aligned class shift CAN change the fixed classifier's AUC:
    assert auc([vadd(x,(Q(10),Q(0))) for x in pos],neg,w)>baseline
    return {"score_difference":str(delta),
            "common_orthogonal_aligned":[str(x) for x in components],
            "fixed_classifier_AUC":str(baseline),
            "identity_exact":True,"uses_real_EEG":False,
            "tests_new_target_generalisation":False}

if __name__=="__main__":
    print(json.dumps(run(),indent=2))
