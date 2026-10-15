#!/usr/bin/env python3
"""Compare exact-rational root isolation engines on P4 rank polynomials.

SymPy's rational isolating intervals and exact real_roots are established
strong algorithms. This times root isolation itself, not the incremental
AUC sweep; no claim of new isolator, real EEG or float certification.
"""
import argparse,hashlib,json,statistics,time
from pathlib import Path
import sympy as sp
from p4_incremental_exact_event_sweep import constructed_coefficients
from p4_exact_rank_event_sweep_prototype import numerator,_cmp_exact

EPS=sp.Rational(1,10**8)

def roots_via_intervals(poly):
    if poly.is_zero:return []
    return poly.intervals(eps=EPS)

def verify(poly, intervals, exact):
    if poly.is_zero:
        assert not intervals and not exact
        return 0
    assert len(intervals)==len(exact)
    for ((lo,hi),multiplicity),(root,m2) in zip(intervals,exact):
        assert multiplicity==m2 and _cmp_exact(root,lo)>=0 and _cmp_exact(root,hi)<=0
        assert hi-lo<=EPS or lo==hi
    internal=sum(_cmp_exact(root,0)>0 and _cmp_exact(root,1)<0 for root,_ in exact)
    assert internal==sum(_cmp_exact(lo,0)>0 and _cmp_exact(hi,1)<0 for (lo,hi),_ in intervals)
    assert poly.count_roots(0,1)>=internal
    return internal

def fixture(dimension,pairs,kind):
    spectrum,coeffs=constructed_coefficients(pairs,dimension)
    if kind=='no_event':
        coeffs=[[sp.Rational(1+i+k,i+k+2) for i in range(dimension)] for k in range(pairs)]
    if kind=='mixed':
        coeffs=[[sp.Rational(((k+3)*(i+5))%13-6,(i+1)*(k+2)) for i in range(dimension)]
                for k in range(pairs)]
    return [numerator(spectrum,x) for x in coeffs]

def run():
    rows=[]
    cases=[(3,8,'crossing'),(3,16,'crossing'),(6,8,'crossing'),
           (6,16,'crossing'),(3,8,'mixed'),(6,8,'mixed'),
           (3,16,'no_event'),(6,16,'no_event')]
    for dimension,pairs,kind in cases:
        polys=fixture(dimension,pairs,kind)
        intv=[roots_via_intervals(x) for x in polys]
        exact=[[] if x.is_zero else x.real_roots(multiple=False) for x in polys]
        inside=sum(verify(x,a,b) for x,a,b in zip(polys,intv,exact))
        times={'rational_intervals':[],'exact_real_roots':[]}
        for rep in range(4):
            for name in (('rational_intervals','exact_real_roots') if rep%2==0 else
                         ('exact_real_roots','rational_intervals')):
                t=time.perf_counter()
                current=[roots_via_intervals(x) for x in polys] if name=='rational_intervals' else [
                    [] if x.is_zero else x.real_roots(multiple=False) for x in polys]
                times[name].append(time.perf_counter()-t)
                assert current==(intv if name=='rational_intervals' else exact)
        row={'dimension':dimension,'pairs':pairs,'fixture':kind,
             'distinct_interior_root_incidences':inside,
             'rational_interval_median_seconds':statistics.median(times['rational_intervals']),
             'exact_real_roots_median_seconds':statistics.median(times['exact_real_roots']),
             'all_isolating_intervals_contain_exact_root_with_multiplicity':True,
             'interval_eps':str(EPS),
             'timing_scope':'full_real_line_per_polynomial_root_engine_only'}
        rows.append(row);print('P4_STURM_ROOT_COMPARATOR_ROW',json.dumps(row,sort_keys=True),flush=True)
    return {'status':'EXECUTED_RATIONAL_ROOT_ISOLATION_STRONG_BASELINE',
            'rows':rows,'sympy_version':sp.__version__,
            'algorithms':'SymPy exact rational isolating intervals versus SymPy real_roots',
            'not_end_to_end_auc_path_timing':True,'no_float_input_error_bound':True,
            'no_root_isolator_novelty_claim':True,'exploratory_constructed_families':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode();args.out.write_bytes(data)
    print('P4_STURM_ROOT_COMPARATOR_SHA',hashlib.sha256(data).hexdigest(),flush=True)
