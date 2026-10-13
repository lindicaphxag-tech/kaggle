#!/usr/bin/env python3
"""Retrospective synthetic exact-rational rank certificates AND counterexamples.
Fixed positive spectral uncertainty boxes in a common eigencoordinate frame.
No empirical EEG transfer result, covariance estimator error bound, or IEEE
floating-point guarantee is inferred from these generated fixtures.
"""
import argparse, hashlib, itertools, json, random
from fractions import Fraction as Q
from pathlib import Path
from p4_joint_spectral_coefficient_box_certificate import certify, exact_score

def sample(seed, n, hard=False):
    r=random.Random(seed); out=[]
    for z in range(n):
        p=r.choice([3,4,5] if hard else [2,3,4,5])
        s=[]
        for _ in range(p):
            if hard:
                lo=Q(r.randint(2,24),r.randint(4,12))
                hi=lo+Q(r.randint(0,7),r.randint(24,64))
            else:
                lo=Q(r.randint(1,16),r.randint(1,7))
                hi=lo+Q(r.randint(0,7),r.randint(2,12))
            s.append((lo,hi))
        if hard:
            a=[Q(r.randint(-25,25),30) for _ in range(p)]
            if z%4==0: a[0]+=2
            if z%4==1: a[0]-=2
            e=[Q(r.randint(0,4),80) for _ in range(p)]
            L,R=Q(0),Q(1)
        else:
            if z%3==0:a=[Q(r.randint(1,14),13) for _ in range(p)]
            elif z%3==1:a=[-Q(r.randint(1,14),13) for _ in range(p)]
            else:a=[Q(r.randint(-13,13),13) for _ in range(p)]
            e=[Q(r.randint(0,5),20) for _ in range(p)]
            L,R=Q(r.randint(0,2),8),Q(r.randint(6,8),8)
        out.append((s,a,e,L,R))
    return out

def bern(s,a,e,L,R):
    args=([[str(u),str(v)] for u,v in s],
          [str(x) for x in a],[str(x) for x in e],str(L),str(R))
    return {'POS':1,'NEG':-1,'UNKNOWN':0}[certify(*args)['status']]

def direct(s,a,e,L,R):
    p=len(s);nu=(sum(x for x,y in s)/p,sum(y for x,y in s)/p)
    lower=upper=Q(0)
    for i,(sl,su) in enumerate(s):
        den_lo=min((1-t)*sl+t*nu[0] for t in (L,R))
        den_hi=max((1-t)*su+t*nu[1] for t in (L,R))
        vals=[v/d for v in (a[i]-e[i],a[i]+e[i]) for d in (den_lo,den_hi)]
        lower+=min(vals);upper+=max(vals)
    return 1 if lower>0 else -1 if upper<0 else 0

def recurse(s,a,e,L,R,budget,lambda_splits):
    b=bern(s,a,e,L,R); d=direct(s,a,e,L,R)
    if b and d and b!=d: raise AssertionError('unsound conflict')
    decision=b or d
    if decision or not budget:return decision
    if lambda_splits:
        mid=(L+R)/2
        x=recurse(s,a,e,L,mid,budget-1,lambda_splits-1)
        if not x:return 0
        y=recurse(s,a,e,mid,R,budget-1,lambda_splits-1)
        return x if x==y else 0
    widths=[(hi-lo)/(lo+Q(1,1000)) for lo,hi in s]
    i=max(range(len(s)),key=lambda k:widths[k])
    if widths[i]==0:return 0
    lo,hi=s[i];mid=(lo+hi)/2
    left=list(s);right=list(s)
    left[i]=(lo,mid);right[i]=(mid,hi)
    x=recurse(left,a,e,L,R,budget-1,0)
    if not x:return 0
    y=recurse(right,a,e,L,R,budget-1,0)
    return x if x==y else 0

def witness(s,a,e,L,R):
    plus=minus=zero=None;p=len(s)
    for sbits in itertools.product((0,1),repeat=p):
        S=[s[i][b] for i,b in enumerate(sbits)]
        for abits in itertools.product((-1,1),repeat=p):
            A=[a[i]+abits[i]*e[i] for i in range(p)]
            for t in (L,(3*L+R)/4,(L+R)/2,(L+3*R)/4,R):
                v=exact_score(S,A,t)
                proof={'spectrum':list(map(str,S)),'coefficients':list(map(str,A)),
                       'lambda':str(t),'exact_score':str(v)}
                if v==0:zero=proof
                if v>0 and plus is None:plus=proof
                if v<0 and minus is None:minus=proof
                if zero:return {'type':'EXACT_ZERO','zero':zero}
                if plus and minus:return {'type':'OPPOSITE_SIGNS',
                                          'positive':plus,'negative':minus}
    return None

def verify_witness(w,s,a,e,L,R):
    for key in ('positive','negative','zero'):
        if key not in w:continue
        z=w[key]; S=list(map(Q,z['spectrum']));A=list(map(Q,z['coefficients']))
        t=Q(z['lambda']);val=Q(z['exact_score'])
        assert L<=t<=R
        assert all(lo<=x<=hi for x,(lo,hi) in zip(S,s))
        assert all(x-y<=u<=x+y for u,x,y in zip(A,a,e))
        assert exact_score(S,A,t)==val
        assert (val>0 if key=='positive' else val<0 if key=='negative' else val==0)

def main():
    fixtures=sample(20261013,126)+sample(20261014,96,hard=True)
    counts={'ROBUST_POS':0,'ROBUST_NEG':0,'FALSIFIED':0,'UNKNOWN':0}
    baselines={'whole_bernstein':0,'whole_direct_rational':0,
               'lambda_refinement':0,'extra_spectral_rescue':0}
    proofs=[]; certs=[]
    for idx,(s,a,e,L,R) in enumerate(fixtures):
        b=bern(s,a,e,L,R); d=direct(s,a,e,L,R)
        if b and d:assert b==d
        baselines['whole_bernstein']+=bool(b)
        baselines['whole_direct_rational']+=bool(d)
        sign=recurse(s,a,e,L,R,4,4)
        baselines['lambda_refinement']+=bool(sign)
        if not sign:
            sign=recurse(s,a,e,L,R,8,0)
            if sign:baselines['extra_spectral_rescue']+=1
        if sign:
            counts['ROBUST_POS' if sign>0 else 'ROBUST_NEG']+=1
            assert witness(s,a,e,L,R) is None
            certs.append({'fixture':idx,'sign':sign,
                          'full_box_proven_by_exact_interval_enclosures':True})
            continue
        w=witness(s,a,e,L,R)
        if w:
            verify_witness(w,s,a,e,L,R)
            counts['FALSIFIED']+=1
            proofs.append({'fixture':idx,'witness':w})
        else:counts['UNKNOWN']+=1
    assert len(fixtures)==222
    assert sum(counts.values())==len(fixtures)
    assert baselines['lambda_refinement']==184
    assert baselines['extra_spectral_rescue']==2
    assert counts=={'ROBUST_POS':95,'ROBUST_NEG':91,'FALSIFIED':36,'UNKNOWN':0}
    return {'status':'PASS','fixture_seeds':[20261013,20261014],
            'fixture_counts':[126,96],'total':len(fixtures),
            'baseline_certified':baselines,'final':counts,
            'falsifications':proofs,'certifications':certs,
            'whole_input_box_guarantees_are_conditional':True,
            'no_independent_EEG_acquisition':True,
            'no_floating_pipeline_bound':True,
            'all_falsifications_have_exact_admissible_witnesses':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();z=main();data=json.dumps(z,indent=2,sort_keys=True)+'\n'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(data);print('P4_EXACT_DECISION_PASS',json.dumps(
        {'baseline_certified':z['baseline_certified'],'final':z['final']},
        sort_keys=True))
    print('SHA256',hashlib.sha256(data.encode()).hexdigest())
