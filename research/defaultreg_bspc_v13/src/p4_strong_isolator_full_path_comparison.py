#!/usr/bin/env python3
"""Fair exact AUC full-path comparison with shared rational isolator.

Both methods isolate every polynomial root through identical SymPy
rational intervals; nonrational roots fall back to exact real_roots.
Only full reevaluation versus bundled incident-pair update differs.
All outputs include endpoint and event half-credit ties exactly.
"""
import argparse,hashlib,json,statistics,time
from functools import cmp_to_key
from pathlib import Path
import sympy as sp
from p4_exact_rank_event_sweep_prototype import numerator,_cmp_exact,_sign_exact,event_map
from p4_incremental_exact_event_sweep import constructed_coefficients

EPS=sp.Rational(1,10**8)

def bundles(spectra,pair_coeffs):
    polys=[numerator(spectra,a) for a in pair_coeffs]
    events=[];fallback=0
    for idx,p in enumerate(polys):
        if p.is_zero:continue
        intervals=p.intervals(eps=EPS)
        roots=None
        for (lo,hi),multiplicity in intervals:
            # Certified disjoint rational bounds already rule out an
            # interior event; endpoint ties are handled by direct eval.
            if hi<=0 or lo>=1:continue
            if lo==hi:
                root=lo
            else:
                if roots is None:roots=p.real_roots(multiple=False);fallback+=1
                match=[r for r,m in roots if m==multiplicity and
                       _cmp_exact(r,lo)>=0 and _cmp_exact(r,hi)<=0]
                if len(match)!=1:raise AssertionError('non-unique algebraic root in isolating interval')
                root=match[0]
            if _cmp_exact(root,0)>0 and _cmp_exact(root,1)<0:
                events.append((root,idx,multiplicity))
    events.sort(key=cmp_to_key(lambda a,b:_cmp_exact(a[0],b[0])))
    packed=[]
    for root,idx,m in events:
        if not packed or _cmp_exact(root,packed[-1][0])!=0:packed.append((root,[]))
        packed[-1][1].append((idx,m))
    return polys,packed,fallback

def full(spectra,coeffs):
    polys,events,fallback=bundles(spectra,coeffs)
    points=[sp.Integer(0)]+[x for x,_ in events]+[sp.Integer(1)]
    n=sp.Integer(len(polys))
    boundaries=[(x,sp.simplify(sum(_sign_exact(p.eval(x)) for p in polys)/n)) for x in points]
    cells=[((a,b),sp.simplify(sum(_sign_exact(p.eval((a+b)/2)) for p in polys)/n))
           for a,b in zip(points[:-1],points[1:])]
    return polys,boundaries,cells,fallback

def incremental(spectra,coeffs):
    polys,events,fallback=bundles(spectra,coeffs)
    points=[sp.Integer(0)]+[x for x,_ in events]+[sp.Integer(1)]
    n=sp.Integer(len(polys))
    boundaries=[(sp.Integer(0),sp.simplify(sum(_sign_exact(p.eval(0)) for p in polys)/n))]
    credits=[_sign_exact(p.eval(points[1]/2)) for p in polys]
    running=sp.simplify(sum(credits)/n);cells=[]
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        cells.append(((a,b),running))
        if i==len(events):break
        event,incident=events[i]
        at=running*n;next_sum=at
        for idx,m in incident:
            old=credits[idx]
            if old not in (sp.Integer(0),sp.Integer(1)):raise AssertionError('unexpected interior tie')
            at+=sp.Rational(1,2)-old
            new=1-old if m%2 else old
            next_sum+=new-old;credits[idx]=new
        boundaries.append((event,sp.simplify(at/n)))
        running=sp.simplify(next_sum/n)
    boundaries.append((sp.Integer(1),sp.simplify(sum(_sign_exact(p.eval(1)) for p in polys)/n)))
    return polys,boundaries,cells,fallback

def test():
    spectra,pairs=constructed_coefficients(4,3)
    spectra2,pairs2=constructed_coefficients(4,6)
    for s,p in [(spectra,pairs),(spectra2,pairs2),([1,3],[[1,-2],[0,0]])]:
        a=full(s,p);b=incremental(s,p)
        assert a[:3]==b[:3]==event_map(s,p)
    assert full([1,3],[[0,0],[1,-2],[-1,2]])[:3]==incremental([1,3],[[0,0],[1,-2],[-1,2]])[:3]
    x=sp.Symbol('lambda',real=True);s=list(map(sp.Rational,[1,2,4]));avg=sum(s)/3
    basis=[sp.prod(s[j]+(avg-s[j])*x for j in range(3) if j!=i) for i in range(3)]
    mat=sp.Matrix([[sp.Poly(q,x).nth(k) for q in basis] for k in range(3)])
    polynomials=[sp.Poly((x-sp.Rational(1,3))**2,x),sp.Poly(x*(x-1),x),
                 sp.Poly(x*x-sp.Rational(1,2),x)]
    coeffs=[[sp.simplify(v) for v in mat.inv()*sp.Matrix([p.nth(i) for i in range(3)])]
            for p in polynomials]
    assert full(s,coeffs)[:3]==incremental(s,coeffs)[:3]==event_map(s,coeffs)

def run():
    test();rows=[]
    fixtures=[(3,16,'crossing'),(3,32,'crossing'),(6,8,'crossing'),
              (6,16,'crossing'),(3,32,'no_event'),(6,16,'no_event')]
    for dim,n,kind in fixtures:
        s,pairs=constructed_coefficients(n,dim)
        if kind=='no_event':pairs=[[sp.Rational(1+i+k,i+k+2) for i in range(dim)] for k in range(n)]
        a=full(s,pairs);b=incremental(s,pairs)
        assert a[:3]==b[:3]==event_map(s,pairs)
        timing={'full':[],'incremental':[]}
        for rep in range(4):
            for method in (('full','incremental') if rep%2==0 else ('incremental','full')):
                t=time.perf_counter()
                current=full(s,pairs) if method=='full' else incremental(s,pairs)
                timing[method].append(time.perf_counter()-t)
                assert current[:3]==a[:3]
        row={'dimension':dim,'pairs':n,'fixture':kind,'distinct_events':len(a[1])-2,
             'full_recompute_median_seconds':statistics.median(timing['full']),
             'incremental_median_seconds':statistics.median(timing['incremental']),
             'same_rational_root_isolator_and_exact_output':True,
             'nonrational_polynomial_fallback_count':a[3]}
        rows.append(row);print('P4_STRONG_ISOLATOR_FULL_PATH_ROW',json.dumps(row,sort_keys=True),flush=True)
    return {'status':'EXECUTED_EXACT_RATIONAL_STRONG_ISOLATOR_FULL_AUC_PATH_COMPARISON',
            'rows':rows,'sympy_version':sp.__version__,
            'root_engine':'Poly.intervals width 1e-8; exact real_roots only for nonrational intervals',
            'identical_root_engine_both_methods':True,'floating_EEG_not_certified':True,
            'constructed_input_only':True,'not_new_root_isolation_algorithm':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();args.out.write_bytes(data)
    print('P4_STRONG_ISOLATOR_FULL_PATH_SHA',hashlib.sha256(data).hexdigest(),flush=True)
