#!/usr/bin/env python3
"""Exact rational robust score-sign bound in ORIGINAL covariance coordinates.
For symmetric ||DeltaS||2<=epsS and ||Delta d||2<=epsD and
||Delta v||2<=epsV, certifies v'^T[(1-lambda)S'+lambda tr(S')I/p]^-1 d'
for all lambda in [L,R] WITHOUT fixing covariance eigenvectors.
Bounds must be justified externally; NOT an IEEE754/real EEG guarantee.
"""
import argparse,hashlib,json,random
from fractions import Fraction as F
from math import comb
from pathlib import Path
import sympy as sp

def rat(x):
    if isinstance(x,float):raise TypeError("binary float not exact")
    return F(x)

def bern(coeff,L,R):
    n=len(coeff)-1
    z=[sum(coeff[j]*F(comb(j,k))*L**(j-k)*(R-L)**k
           for j in range(k,n+1)) for k in range(n+1)]
    return [sum(z[j]*F(comb(k,j),comb(n,j))
                for j in range(k+1)) for k in range(n+1)]

def gap_lower(S):
    n=S.rows
    if S!=S.T or any(S[:k,:k].det()<=0 for k in range(1,n+1)):
        raise ValueError("not exact symmetric SPD")
    inv=S.inv()
    upper=max(sum(abs(inv[i,j]) for j in range(n))
              for i in range(n))
    return F(1)/F(upper)

def certify(S,v,d,epsS,epsV,epsD,left=0,right=1):
    n=S.rows
    if n<2 or S.cols!=n or len(v)!=n or len(d)!=n:
        raise ValueError("dimension mismatch")
    if any(isinstance(z,float) for z in list(S)+list(v)+list(d)):
        raise TypeError("float input rejected")
    S=sp.Matrix([[sp.Rational(t) for t in row] for row in S.tolist()])
    v=sp.Matrix([sp.Rational(x) for x in v])
    d=sp.Matrix([sp.Rational(x) for x in d])
    epsS,epsV,epsD,L,R=map(rat,(epsS,epsV,epsD,left,right))
    if not 0<=L<R<=1 or min(epsS,epsV,epsD)<0:
        raise ValueError("invalid radius or shrinkage interval")
    lower=gap_lower(S)
    nu=F(sp.trace(S))/n
    mu=min(lower,nu)
    if epsS>=mu:
        return {"status":"UNKNOWN","reason":"no_robust_SPD_gap"}
    t=sp.symbols("t")
    A=(1-t)*S+t*sp.trace(S)*sp.eye(n)/n
    numer=sp.Poly((v.T*A.adjugate()*d)[0],t,domain="QQ")
    denom=sp.Poly(A.det(),t,domain="QQ")
    ncoef=[F(numer.nth(k)) for k in range(max(numer.degree(),0)+1)]
    dcoef=[F(denom.nth(k)) for k in range(max(denom.degree(),0)+1)]
    Bn=bern(ncoef,L,R);Bd=bern(dcoef,L,R)
    if max(Bd)<=0:raise AssertionError("SPD determinant upper bound")
    pos=min(Bn)/max(Bd) if min(Bn)>0 else F(0)
    neg=-max(Bn)/max(Bd) if max(Bn)<0 else F(0)
    # l1 bounds the Euclidean norms exactly, yielding a rational certificate
    nv=sum((abs(F(x)) for x in v),F(0))
    nd=sum((abs(F(x)) for x in d),F(0))
    delta=(epsS*nv*nd/(mu*(mu-epsS))+
           (nv*epsD+epsV*nd+epsV*epsD)/(mu-epsS))
    status="POS" if pos>delta else "NEG" if neg>delta else "UNKNOWN"
    return {"status":status,"mu_lower":str(mu),
            "whole_interval_base_margin_lower":str(max(pos,neg)),
            "whole_interval_score_error_upper":str(delta),
            "covariance_eigenvectors_fixed":False,
            "all_admissible_symmetric_operator_norm_perturbations":True,
            "not_full_EEG_pipeline_or_IEEE754_bound":True}

def score(S,v,d,t):
    n=S.rows
    return (v.T*((1-t)*S+t*sp.trace(S)*sp.eye(n)/n).inv()*d)[0]

def run(n=120):
    rng=random.Random(20261013)
    counts=dict(POS=0,NEG=0,UNKNOWN=0)
    checks=0;examples=[]
    for z in range(n):
        p=rng.choice([2,3,4])
        G=sp.Matrix([[rng.randint(-2,2) for _ in range(p)] for _ in range(p)])
        S=G.T*G+sp.Rational(rng.randint(2,7))*sp.eye(p)
        d=sp.Matrix([sp.Rational(rng.randint(1,5),3)*rng.choice([-1,1])
                     for _ in range(p)])
        orient=rng.choice([-1,1]) if z%5<4 else 0
        v=orient*d+sp.Matrix([sp.Rational(rng.randint(-2,2),12)
                              for _ in range(p)])
        es=F(rng.choice([0,1,2,3]),100)
        ev=F(rng.randint(0,2),100)
        ed=F(rng.randint(0,2),100)
        x=certify(S,v,d,es,ev,ed)
        counts[x["status"]]+=1
        if x["status"]!="UNKNOWN":
            for _ in range(4):
                E=sp.Matrix([[sp.Rational(rng.randint(-1,1))*sp.Rational(es,p)
                              for _ in range(p)] for _ in range(p)])
                E=(E+E.T)/2
                assert max(sum(abs(E[i,j]) for j in range(p))
                           for i in range(p))<=es
                dv=sp.Matrix([sp.Rational(rng.randint(-1,1))*sp.Rational(ev,p)
                              for _ in range(p)])
                dd=sp.Matrix([sp.Rational(rng.randint(-1,1))*sp.Rational(ed,p)
                              for _ in range(p)])
                for t in map(sp.Rational,(0,sp.Rational(1,4),
                                           sp.Rational(1,2),sp.Rational(3,4),1)):
                    y=score(S+E,v+dv,d+dd,t)
                    assert y>0 if x["status"]=="POS" else y<0
                    checks+=1
        if len(examples)<5:
            examples.append({"index":z,"status":x["status"],
                             "margin":x.get("whole_interval_base_margin_lower"),
                             "uncertainty":x.get("whole_interval_score_error_upper")})
    assert counts==dict(POS=46,NEG=65,UNKNOWN=9)
    assert checks==2220
    return {"status":"PASS","synthetic_seed":20261013,
            "sample_count":n,"decisions":counts,
            "exact_covariance_perturbation_oracle_checks":checks,
            "examples":examples,"uses_rotating_eigenspaces":True,
            "external_input_bounds_required":True,
            "no_real_EEG_or_machine_rounding_claim":True}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();z=run()
    payload=json.dumps(z,indent=2,sort_keys=True)+"\n"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(payload)
    print("P4_COVARIANCE_COORDINATE_GUARANTEE_PASS",json.dumps(
        {"decisions":z["decisions"],"exact_oracle":z["exact_covariance_perturbation_oracle_checks"]}))
    print("SHA256",hashlib.sha256(payload.encode()).hexdigest())
