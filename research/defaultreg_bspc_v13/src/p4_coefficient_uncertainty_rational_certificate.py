#!/usr/bin/env python3
"""Coefficient-uncertainty-aware exact rational Bernstein certificate.
Scope: fixed positive spectrum, rational a_i, |delta_i|<=epsilon_i.
Not a certificate for floating EEG preprocessing/eigensolvers or IEEE-754.
"""
from fractions import Fraction as F
from math import comb
import argparse,hashlib,itertools,json,random
from pathlib import Path

def R(x):
    if isinstance(x,float): raise TypeError("Float inputs are not exact")
    return F(x)

def add(a,b):
    return [(a[k] if k<len(a) else 0)+(b[k] if k<len(b) else 0)
            for k in range(max(len(a),len(b)))]

def mul(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out

def bernstein(power,left,right):
    n=len(power)-1
    cp=[sum(power[j]*comb(j,k)*left**(j-k)*(right-left)**k
            for j in range(k,n+1)) for k in range(n+1)]
    return [sum(cp[j]*F(comb(k,j),comb(n,j)) for j in range(k+1))
            for k in range(n+1)]

def certify(spectra,coeff,error,left=0,right=1):
    s=list(map(R,spectra));a=list(map(R,coeff));e=list(map(R,error))
    left,right=R(left),R(right)
    if len(s)<2 or len(s)!=len(a) or len(s)!=len(e):raise ValueError("dimension")
    if any(x<=0 for x in s) or any(x<0 for x in e):raise ValueError("SPD/error")
    if not 0<=left<right<=1:raise ValueError("interval")
    nu=sum(s,F(0))/len(s)
    q=[[x,nu-x] for x in s]
    N=[F(0)]*len(s);E=[F(0)]*len(s)
    for i in range(len(s)):
        v=[F(1)]
        for j in range(len(s)):
            if i!=j:v=mul(v,q[j])
        N=add(N,[a[i]*x for x in v])
        E=add(E,[e[i]*x for x in v])
    lower=bernstein(add(N,[-x for x in E]),left,right)
    negative_upper=bernstein([-x for x in add(N,E)],left,right)
    if min(lower)>0:return "POS"
    if min(negative_upper)>0:return "NEG"
    return "UNKNOWN"

def run():
    assert certify([1,3],[1,1],["1/10","1/10"])=="POS"
    assert certify([1,3],[-1,-1],["1/10","1/10"])=="NEG"
    assert certify([1,3],[1,0],["11/10",0])=="UNKNOWN"
    assert certify([1,3],[1,0],[0,0])=="POS"
    for args in (([0,3],[1,2],[0,0]),([1,3],[1,2],[-1,0])):
        try:certify(*args)
        except ValueError:pass
        else:raise AssertionError("bad inputs accepted")
    try:certify([1.0,3],[1,2],[0,0])
    except TypeError:pass
    else:raise AssertionError("float inputs silently accepted")
    rng=random.Random(20261013)
    counts=dict(POS=0,NEG=0,UNKNOWN=0);checks=0
    for _ in range(96):
        p=rng.choice([2,3,4,5])
        s=[F(rng.randint(1,18),rng.randint(1,6)) for i in range(p)]
        a=[F(rng.randint(-12,12),12) for i in range(p)]
        e=[F(rng.randint(0,4),24) for i in range(p)]
        L=F(rng.randint(0,3),8);U=F(rng.randint(5,8),8)
        result=certify(s,a,e,L,U);counts[result]+=1
        if result!="UNKNOWN":
            for corners in itertools.product((-1,1),repeat=p):
                for t in (L,(L+U)/2,U):
                    nu=sum(s,F(0))/p
                    val=sum((a[i]+corners[i]*e[i])/(s[i]+(nu-s[i])*t) for i in range(p))
                    assert val>0 if result=="POS" else val<0
                    checks+=1
    return dict(status="PASS",fixed_seed=20261013,fixtures=96,
                exact_corner_checks=checks,decisions=counts,
                floating_eeg_pipeline_certified=False,
                spectral_eigenvalues_assumed_exact=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path)
    args=ap.parse_args();result=run();print(json.dumps(result,sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        data=json.dumps(result,sort_keys=True,indent=2)+"\n"
        args.out.write_text(data)
        print("P4_ROBUST_BERNSTEIN_SHA256",hashlib.sha256(data.encode()).hexdigest())
