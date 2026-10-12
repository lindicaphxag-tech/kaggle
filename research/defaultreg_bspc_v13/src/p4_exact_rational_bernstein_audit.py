#!/usr/bin/env python3
"""Exact-rational verification of polynomial sign enclosures.

This is an algorithmic unit test on user-specified RATIONAL spectra.
It does not certify floating-point EEG feature extraction or the
previously reported experimental 94.267% coverage.
"""
from fractions import Fraction as Q
from math import comb
import json

def rational(x):
    if isinstance(x, float):
        raise TypeError("Use integer/string/Fraction, never implicit float")
    return Q(x)

def mul(a, b):
    c=[Q(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            c[i+j] += x*y
    return c

def numerator(spectrum, weights, nu):
    s=[rational(x) for x in spectrum]
    a=[rational(x) for x in weights]
    nu=rational(nu)
    if len(s)!=len(a) or not s or min(s)<=0 or nu<=0:
        raise ValueError("Only strictly SPD scalar shrinkage paths")
    out=[Q(0)]*len(s)
    for i,ai in enumerate(a):
        term=[Q(1)]
        for j,sj in enumerate(s):
            if j!=i:
                term=mul(term,[sj,nu-sj])
        for k,t in enumerate(term):
            out[k]+=ai*t
    return out

def coefficients_on_interval(power, left=Q(0), right=Q(1)):
    l,r=rational(left),rational(right)
    if not Q(0)<=l<r<=Q(1):
        raise ValueError("Require [l,r] within [0,1]")
    n=len(power)-1
    shifted=[sum(power[j]*comb(j,k)*l**(j-k)*(r-l)**k for j in range(k,n+1)) for k in range(n+1)]
    return [sum(shifted[j]*Q(comb(k,j),comb(n,j)) for j in range(k+1)) for k in range(n+1)]

def certify(power,left=Q(0),right=Q(1)):
    if all(x==0 for x in power): return "identically_tied"
    b=coefficients_on_interval(power,left,right)
    if min(b)>0: return "strict_positive"
    if max(b)<0: return "strict_negative"
    return "unresolved"

def eval_poly(coeffs,x):
    x=rational(x)
    ans=Q(0)
    for v in coeffs[::-1]: ans=ans*x+v
    return ans

def check():
    # Root at t=1/2: positive and negative halves cannot each
    # receive an inclusive strict sign certificate.
    n=numerator(["1","3"],["1","-2"],"2")
    assert n==[Q(1),Q(-2)]
    assert certify(n)=="unresolved"
    assert eval_poly(n,Q(1,2))==0
    assert certify(n,0,Q(1,4))=="strict_positive"
    assert certify(n,Q(3,4),1)=="strict_negative"
    # Whole-path strictly positive in a cancellation-sensitive case.
    stable=numerator(["1","4","9"],["1","-1","1"],"14/3")
    assert certify(stable)=="strict_positive"
    # Identically zero must retain half-credit for ties, never + or -.
    assert certify(numerator(["1","3"],["0","0"],"2"))=="identically_tied"
    # Stress very-near-zero rational intercept: no arbitrary eps.
    tiny=numerator(["1","3"],["1","-1/3"+"0"*0],"2")
    assert all(isinstance(v,Q) for v in tiny)
    examined=0
    for p in [2,3,6,9,12]:
        s=[Q(i*i+1) for i in range(1,p+1)]
        nu=sum(s)/p
        for pattern in range(8):
            a=[Q((-1)**(i+pattern)*(i+1), (i+3)*(pattern+2)) for i in range(p)]
            coef=numerator(s,a,nu)
            for l,r in [(Q(0),Q(1)),(Q(0),Q(1,3)),(Q(1,3),Q(2,3)),(Q(2,3),Q(1))]:
                cls=certify(coef,l,r)
                if cls.startswith("strict_"):
                    sign=1 if cls=="strict_positive" else -1
                    # Sample grid is cross-check ONLY, exact Bernstein
                    # enclosure, not these points, constitutes the proof.
                    for j in range(21):
                        z=eval_poly(coef,l+(r-l)*Q(j,20))
                        assert sign*z>0
                examined+=1
    return {"exact_rational_interval_cases":examined,
            "dimensions":[2,3,6,9,12],
            "scope":"Exact signs of rational-input numerator polynomials only",
            "raw_EEG_or_float_end_to_end_certified":False,
            "original_real_EEG_coverage_reestimated":False}

if __name__=="__main__":
    print(json.dumps(check(),indent=2))
