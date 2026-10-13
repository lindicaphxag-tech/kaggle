#!/usr/bin/env python3
"""Exact rational interval-Bernstein vs interval-Horner benchmark.

Mathematically sound on deliberately rational synthetic inputs only.
Computational timing is machine-dependent, NOT a proven runtime bound.
Neither method certifies original floating-point EEG preprocessing.
"""
import json,time,sys
from math import comb
from fractions import Fraction as Q
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from p4_exact_rational_bernstein_audit import numerator,certify,eval_poly

def imul(a,b):
    v=[a[i]*b[j] for i in range(2) for j in range(2)]
    return min(v),max(v)
def horner(coeff,left,right):
    acc=(coeff[-1],coeff[-1])
    for v in reversed(coeff[:-1]):
        lo,hi=imul(acc,(left,right));acc=(lo+v,hi+v)
    if acc[0]>0:return "strict_positive"
    if acc[1]<0:return "strict_negative"
    if all(x==0 for x in coeff):return "identically_tied"
    return "unresolved"
def centered_horner(coeff,left,right):
    """Exact Taylor shift to midpoint, followed by exact interval Horner."""
    center=(left+right)/2
    radius=(right-left)/2
    shifted=[sum(coeff[k]*comb(k,j)*center**(k-j)
                 for k in range(j,len(coeff))) for j in range(len(coeff))]
    return horner(shifted,-radius,radius)

def verify_sign(poly,label,l,r):
    if label not in ("strict_positive","strict_negative"):return
    sign=Q(1) if label=="strict_positive" else Q(-1)
    # These exact grid checks are a redundancy test; the certificate
    # follows from the exact interval / Bernstein enclosure itself.
    for j in range(9):
        x=l+(r-l)*Q(j,8)
        assert sign*eval_poly(poly,x)>0

def run():
    report={"status":"EXACT_RATIONAL_SYNTHETIC_TWO_CERTIFIER_COMPARISON",
       "input_scope":"deterministic rational SPD spectra / rational contrast weights",
       "certifier_baselines":["exact_power_interval_horner","exact_centered_Horner","exact_Bernstein"],
       "real_EEG_float_input_certified":False,
       "says_any_eeg_auc_is_improved":False,
       "machine_timing_is_complexity_theorem":False,
       "per_dimension":[]}
    for p in (2,3,6,9,12):
        rec={"dimension":p,"cases":0,"intervals":0,
             "bernstein_certified":0,"horner_certified":0,
             "bernstein_only":0,"horner_only":0,
             "centered_horner_certified":0,"bernstein_only_vs_centered":0,
             "centered_only_vs_bernstein":0,"centered_horner_seconds":0.,
             "uncertifiable_sign_change_witnesses":0,
             "unresolved_without_sign_change_witness":0,
             "identically_tied":0,"bernstein_seconds":0.,
             "horner_seconds":0.}
        s=[Q(i*i+1) for i in range(1,p+1)]
        nu=sum(s)/p
        for pattern in range(12):
            a=[Q(((-1)**(i+pattern))*(i+1),(i+3)*(pattern+2)) for i in range(p)]
            poly=numerator(s,a,nu)
            rec["cases"]+=1
            # Compare identical 16 uniform rational leaf intervals.
            for k in range(16):
                lo,hi=Q(k,16),Q(k+1,16)
                t=time.perf_counter();one=certify(poly,lo,hi)
                rec["bernstein_seconds"]+=time.perf_counter()-t
                t=time.perf_counter();two=horner(poly,lo,hi)
                rec["horner_seconds"]+=time.perf_counter()-t
                t=time.perf_counter();three=centered_horner(poly,lo,hi)
                rec["centered_horner_seconds"]+=time.perf_counter()-t
                rec["intervals"]+=1
                verify_sign(poly,one,lo,hi)
                verify_sign(poly,two,lo,hi)
                verify_sign(poly,three,lo,hi)
                if one.startswith("strict_") and three.startswith("strict_") and one!=three:
                    raise AssertionError("Bernstein and centered Horner contradictory")
                if three.startswith("strict_"):rec["centered_horner_certified"]+=1
                if one=="unresolved" and three=="unresolved":
                    # Opposite signed rational evaluations prove that no
                    # sound whole-interval strict-sign certificate can hold.
                    vals=[eval_poly(poly,lo+(hi-lo)*Q(j,128)) for j in range(129)]
                    if min(vals)<0<max(vals):
                        rec["uncertifiable_sign_change_witnesses"]+=1
                    else:
                        rec["unresolved_without_sign_change_witness"]+=1
                if one.startswith("strict_") and three=="unresolved":rec["bernstein_only_vs_centered"]+=1
                if three.startswith("strict_") and one=="unresolved":rec["centered_only_vs_bernstein"]+=1
                if one!="unresolved" and two!="unresolved" and one!=two:
                    raise AssertionError("sound enclosures give contradictory signs")
                if one=="identically_tied":rec["identically_tied"]+=1
                if one.startswith("strict_"):rec["bernstein_certified"]+=1
                if two.startswith("strict_"):rec["horner_certified"]+=1
                if one.startswith("strict_") and two=="unresolved":rec["bernstein_only"]+=1
                if two.startswith("strict_") and one=="unresolved":rec["horner_only"]+=1
        assert rec["intervals"]==192
        assert rec["unresolved_without_sign_change_witness"]==0
        report["per_dimension"].append(rec)
        print("P4_EXACT_TWO_METHODS",p,rec,flush=True)
    report["interval_count"]=sum(x["intervals"] for x in report["per_dimension"])
    report["never_assumed_Bernstein_dominates"]=True
    return report
if __name__=="__main__":
    r=run()
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("P4_EXACT_TWO_CERTIFIER_BENCHMARK_DONE",r["interval_count"],flush=True)
