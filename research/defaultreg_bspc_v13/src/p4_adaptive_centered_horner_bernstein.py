#!/usr/bin/env python3
"""Exact rational interval sign certificates under matched leaf budgets.

Compares classical power-basis Horner, midpoint-shifted Horner,
Bernstein, and selective centered-Horner -> Bernstein fallbacks.
The hybrid never infers a sign from a numerical grid and never
conflates rational-input algebra with a certified floating EEG pipeline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from fractions import Fraction as Q
from math import comb
from pathlib import Path
from p4_exact_rational_bernstein_audit import numerator,certify,eval_poly
from p4_exact_two_certifiers_benchmark import horner,imul

def shifted_power(coeff, center):
    center = Q(center)
    return [sum(coeff[j] * comb(j,k) * center**(j-k)
                for j in range(k,len(coeff))) for k in range(len(coeff))]

def centered_horner(coeff, left, right):
    l,r = Q(left),Q(right)
    if not Q(0)<=l<r<=Q(1):
        raise ValueError("outside fraction lambda domain")
    c=(l+r)/2
    h=(r-l)/2
    centered=shifted_power(coeff,c)
    value=(centered[-1],centered[-1])
    for a in reversed(centered[:-1]):
        lower,upper=imul(value,(-h,h))
        value=(lower+a,upper+a)
    lo,hi=value
    if lo>0:return "strict_positive"
    if hi<0:return "strict_negative"
    if all(q==0 for q in coeff):return "identically_tied"
    return "unresolved"

def selective_hybrid(coeff,left,right):
    center=centered_horner(coeff,left,right)
    if center!="unresolved":return center,False
    return certify(coeff,left,right),True

def sign_certified(label):
    return label in ("strict_positive","strict_negative")

def validate_one(poly,l,r,answers):
    signs={a for a in answers if sign_certified(a)}
    if len(signs)>1:raise AssertionError("sound rational enclosures disagree")
    for ans in signs:
        sign=1 if ans=="strict_positive" else -1
        for j in range(25):
            x=l+(r-l)*Q(j,24)
            if sign*eval_poly(poly,x)<=0:
                raise AssertionError("sign certificate counterexample")
    if all(x==0 for x in poly):
        assert set(answers)=={"identically_tied"}

def test_falsification():
    poly=[Q(1),Q(-2)]
    for l,r in [(Q(0),Q(1)),(Q(0),Q(1,2)),(Q(1,2),Q(1))]:
        a=horner(poly,l,r); b=centered_horner(poly,l,r)
        c=certify(poly,l,r); d,_=selective_hybrid(poly,l,r)
        validate_one(poly,l,r,[a,b,c,d])
        assert all(x=="unresolved" for x in (a,b,c,d))
    test=[(Q(1),Q(2)),(Q(-1),Q(-2)),(Q(0),Q(0))]
    for vec in test:
        validate_one(list(vec),Q(0),Q(1),
                     [horner(list(vec),Q(0),Q(1)),
                      centered_horner(list(vec),Q(0),Q(1)),
                      certify(list(vec),Q(0),Q(1)),
                      selective_hybrid(list(vec),Q(0),Q(1))[0]])
    assert centered_horner([Q(0),Q(0)],0,1)=="identically_tied"

def benchmark():
    test_falsification()
    report={
        "status":"EXECUTED_EXACT_RATIONAL_CENTERED_HORNER_SELECTIVE_BERNSTEIN",
        "benchmarks":"matched deterministic 960 interval fixture with p=2,3,6,9,12",
        "score_labels":["power_interval_horner","centered_interval_horner","exact_bernstein","selective_centered_then_bernstein"],
        "not_a_formal_float_machine_proof":True,
        "no_new_EEG_AUC_or_held_target_claim":True,
        "timing_run_is_hardware_dependent":True,
        "per_dimension":[]
    }
    for p in (2,3,6,9,12):
        row={
            "p":p,"intervals":0,
            "power_certified":0,"centered_certified":0,
            "bernstein_certified":0,"selective_certified":0,
            "centered_only_vs_bernstein":0,
            "bernstein_only_vs_centered":0,
            "selective_fallbacks":0,
            "power_seconds":0.,"centered_seconds":0.,
            "bernstein_seconds":0.,"selective_seconds":0.
        }
        spectrum=[Q(i*i+1) for i in range(1,p+1)]
        nu=sum(spectrum)/p
        for pattern in range(12):
            weights=[Q(((-1)**(i+pattern))*(i+1),
                       (i+3)*(pattern+2)) for i in range(p)]
            poly=numerator(spectrum,weights,nu)
            for k in range(16):
                l,r=Q(k,16),Q(k+1,16)
                t=time.perf_counter()
                a=horner(poly,l,r)
                row["power_seconds"]+=time.perf_counter()-t
                t=time.perf_counter()
                b=centered_horner(poly,l,r)
                row["centered_seconds"]+=time.perf_counter()-t
                t=time.perf_counter()
                c=certify(poly,l,r)
                row["bernstein_seconds"]+=time.perf_counter()-t
                t=time.perf_counter()
                d,fallback=selective_hybrid(poly,l,r)
                row["selective_seconds"]+=time.perf_counter()-t
                validate_one(poly,l,r,[a,b,c,d])
                row["intervals"]+=1
                row["power_certified"]+=int(sign_certified(a))
                row["centered_certified"]+=int(sign_certified(b))
                row["bernstein_certified"]+=int(sign_certified(c))
                row["selective_certified"]+=int(sign_certified(d))
                row["centered_only_vs_bernstein"]+=int(sign_certified(b) and not sign_certified(c))
                row["bernstein_only_vs_centered"]+=int(sign_certified(c) and not sign_certified(b))
                row["selective_fallbacks"]+=int(fallback)
                assert int(sign_certified(d))>=int(sign_certified(b))
                assert int(sign_certified(d))>=int(sign_certified(c))
        assert row["intervals"]==192
        report["per_dimension"].append(row)
        print("P4_ADAPTIVE_EXACT_RATIONAL_DIMENSION",p,row,flush=True)
    report["total_intervals"]=sum(x["intervals"] for x in report["per_dimension"])
    report["all_interval_counts_match"]=report["total_intervals"]==960
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    report=benchmark()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n"
    args.out.write_text(payload,encoding="utf-8")
    print("P4_ADAPTIVE_RATIONAL_COMPARISON_SHA256",
          hashlib.sha256(payload.encode()).hexdigest(),flush=True)

if __name__=="__main__":
    main()
