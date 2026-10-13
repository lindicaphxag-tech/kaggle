"""Exact incremental AUC event sweep over rational-input rank polynomials.

All roots still use SymPy's established exact real-root machinery. This
implements only bundled event updates, without re-evaluating every pair
at every cell. No floating EEG or new root-isolation claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from functools import cmp_to_key
from pathlib import Path
import sympy as sp
from p4_exact_rank_event_sweep_prototype import (
    lam, numerator, _cmp_exact, _sign_exact, event_map,
)


def incremental_event_map(spectra, pair_coeffs):
    polys = [numerator(spectra, a) for a in pair_coeffs]
    if not polys:
        raise ValueError('at least one positive-negative pair required')
    events=[]
    for idx, poly in enumerate(polys):
        if poly.is_zero:
            continue
        for root, multiplicity in poly.real_roots(multiple=False):
            if _cmp_exact(root, 0)>0 and _cmp_exact(root, 1)<0:
                events.append((root,idx,multiplicity))
    events.sort(key=cmp_to_key(lambda a,b:_cmp_exact(a[0],b[0])))
    bundles=[]
    for root,idx,multiplicity in events:
        if not bundles or _cmp_exact(root,bundles[-1][0])!=0:
            bundles.append((root,[]))
        bundles[-1][1].append((idx,multiplicity))
    points=[sp.Integer(0)]+[x for x,_ in bundles]+[sp.Integer(1)]
    N=sp.Integer(len(polys))
    left=[(sp.Integer(0),sp.simplify(sum(_sign_exact(p.eval(0)) for p in polys)/N))]
    right=(sp.Integer(1),sp.simplify(sum(_sign_exact(p.eval(1)) for p in polys)/N))
    probe=(points[0]+points[1])/2
    credits=[_sign_exact(p.eval(probe)) for p in polys]
    current=sp.simplify(sum(credits)/N)
    cells=[]
    boundaries=left
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        cells.append(((a,b),current))
        if i>=len(bundles):
            break
        root, affected=bundles[i]
        event_sum=current*N
        new_sum=event_sum
        for pair,multiplicity in affected:
            before=credits[pair]
            if before not in (sp.Integer(0),sp.Integer(1)):
                raise AssertionError('interior pair sign unexpectedly tied')
            event_sum+=sp.Rational(1,2)-before
            after=1-before if multiplicity%2 else before
            new_sum+=after-before
            credits[pair]=after
        boundaries.append((root,sp.simplify(event_sum/N)))
        current=sp.simplify(new_sum/N)
    boundaries.append(right)
    return polys,boundaries,cells


def constructed_coefficients(n, dimension=3):
    """Fixed distinct rational event locations; the same SPD spectrum for all pairs."""
    s=[sp.Rational(v) for v in [1,2,4,7,11,16][:dimension]]
    q=[si+(sum(s)/len(s)-si)*lam for si in s]
    basis=[sp.prod(q[j] for j in range(len(s)) if j!=i) for i in range(len(s))]
    matrix=sp.Matrix([[sp.Poly(b,lam).nth(k) for b in basis] for k in range(len(s))])
    assert matrix.det()!=0
    inverse=matrix.inv()
    out=[]
    for k in range(n):
        a=sp.Rational(2*k+1,4*n+10)
        b=sp.Rational(2*k+2,4*n+10)
        poly=sp.Poly((-1)**k*(lam-a)*(lam-b)*
                     sp.prod(lam+sp.Rational(j+1,10) for j in range(dimension-3)),lam)
        out.append([sp.simplify(v) for v in inverse*sp.Matrix([poly.nth(i) for i in range(len(s))])])
    return s,out


def checks():
    cases=[
        ([1,3],[[1,-2]]),
        ([1,3],[[0,0],[1,-2],[-1,2]]),
        ([1,3],[[0,0],[1,-2],[1,-2]]),
    ]
    for s,pairs in cases:
        assert incremental_event_map(s,pairs)==event_map(s,pairs)
    # Construct an even-multiplicity touching event and a coincident bundle.
    s,pairs=constructed_coefficients(4)
    assert incremental_event_map(s,pairs)==event_map(s,pairs)
    repeated=[sp.Rational(1,3),sp.Rational(1,3)]
    basis=[sp.prod([sp.Rational(si)+(sp.Rational(7,3)-sp.Rational(si))*lam
                    for j,si in enumerate(s) if j!=i]) for i in range(3)]
    mat=sp.Matrix([[sp.Poly(b,lam).nth(k) for b in basis] for k in range(3)])
    p=sp.Poly((lam-repeated[0])**2,lam)
    touch=[sp.simplify(v) for v in mat.inv()*sp.Matrix([p.nth(i) for i in range(3)])]
    assert incremental_event_map(s,pairs+[touch,touch])==event_map(s,pairs+[touch,touch])


def run():
    checks()
    rows=[]
    for dimension,n in [(3,k) for k in (4,8,16,32)]+[(6,k) for k in (4,8,16)]:
        s,pairs=constructed_coefficients(n,dimension)
        slow=[];fast=[]
        for _ in range(3):
            t=time.perf_counter();a=event_map(s,pairs);slow.append(time.perf_counter()-t)
            t=time.perf_counter();b=incremental_event_map(s,pairs);fast.append(time.perf_counter()-t)
            assert a==b
        rows.append({'dimension':dimension,'pairs':n,'events':len(a[1])-2,
                     'cells':len(a[2]),'naive_median_seconds':statistics.median(slow),
                     'incremental_median_seconds':statistics.median(fast),
                     'exact_outputs_equal':True})
        print('P4_INCREMENTAL_EXACT_EVENT_ROW',rows[-1],flush=True)
    return {'status':'EXECUTED_EXACT_RATIONAL_INCREMENTAL_EVENT_BENCHMARK',
            'rows':rows,'baseline':'same SymPy real_roots; full AUC reevaluation per event',
            'method':'same roots; bundled sign-credit updates',
            'fixed_dimensions':[3,6], 'fixed_pair_counts_by_dimension':{'3':[4,8,16,32],'6':[4,8,16]},
            'time_is_machine_dependent':True,
            'no_new_root_isolator_or_float_EEG_certificate':True,
            'no_unseen_subject_gain':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();args.out.write_bytes(data)
    print('P4_INCREMENTAL_EXACT_EVENT_SHA',hashlib.sha256(data).hexdigest(),flush=True)
