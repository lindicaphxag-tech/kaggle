#!/usr/bin/env python3
"""Fixed-seed fair exact-rational source AUC event-path benchmark; synthetic only.
Same SymPy real-roots and incremental event updater for both arms.
"""
import json,math,random,statistics,time
from pathlib import Path
import sympy as sp
from scipy.stats import wilcoxon
from p4_certificate_pruned_incident_auc_20261015 import T,event_sweep
S=(sp.Integer(1),sp.Integer(2),sp.Integer(4))
nu=sum(S)/3
q=[v+(nu-v)*T for v in S]
basis=[sp.Poly(sp.prod(q[j] for j in range(3) if i!=j),T) for i in range(3)]
B=sp.Matrix([[poly.nth(j) for poly in basis] for j in range(3)])
Binv=B.inv()
def generate(seed,regime):
    rng=random.Random(seed);polys=[]
    probability={'stable':0.,'sparse':.12,'mixed':.35,'dense':.90}[regime]
    for i in range(16):
        if rng.random()<probability:
            root=sp.Rational(rng.randint(10,75),100)
            if rng.random()<.3:
                other=sp.Rational(rng.randint(76,95),100)
                p=sp.Poly((T-root)*(T-other),T)
            else:
                other=sp.Rational(rng.randint(10,60),100)
                p=sp.Poly((T-root)*(T+other),T)
        else:
            a=sp.Rational(rng.randint(1,9),10);b=sp.Rational(rng.randint(1,9),10)
            p=sp.Poly((T+a)*(T+b),T)
        if rng.randint(0,1):p=-p
        coeff=Binv*sp.Matrix([p.nth(j) for j in range(3)])
        if sp.expand(sum(coeff[j]*basis[j].as_expr() for j in range(3))-p.as_expr())!=0:
            raise AssertionError('not realizable by shrinkage LDA')
        polys.append(p)
    return polys
def main():
    stats={};rows=[]
    for regime in ('stable','sparse','mixed','dense'):
        for seed in range(101,113):
            polys=generate(seed,regime)
            baseline=event_sweep(polys,False)
            optimized=event_sweep(polys,True)
            for key in ('max_auc','cells','boundaries'):
                if baseline[key]!=optimized[key]:raise AssertionError('exact path mismatch')
            ts={'base':[],'new':[]}
            for j in range(3):
                for mode in (('base','new') if j%2 else ('new','base')):
                    tic=time.perf_counter()
                    check=event_sweep(polys,mode=='new')
                    ts[mode].append(time.perf_counter()-tic)
                    if check['boundaries']!=baseline['boundaries']:raise AssertionError('unstable paths')
            a=statistics.median(ts['base']);b=statistics.median(ts['new'])
            rows.append({'regime':regime,'seed':seed,'pairs':16,'events':optimized['distinct_events'],
                         'calls_saved':baseline['root_calls']-optimized['root_calls'],
                         'reference_seconds':a,'new_seconds':b,'speedup':a/b})
        subset=[r for r in rows if r['regime']==regime]
        stats[regime]={'n':len(subset),'median_speedup':statistics.median(r['speedup'] for r in subset),
                       'wins':sum(r['speedup']>1 for r in subset),
                       'one_sided_p':float(wilcoxon([math.log(r['speedup']) for r in subset],alternative='greater').pvalue)}
        print('P4_FEASIBLE_FAMILY',regime,json.dumps(stats[regime],sort_keys=True),flush=True)
    ordered=sorted(stats,key=lambda k:stats[k]['one_sided_p'])
    lower=0.
    for pos,key in enumerate(ordered):
        lower=max(lower,(len(ordered)-pos)*stats[key]['one_sided_p'])
        stats[key]['holm_adjusted_p']=min(1.,lower)
    obj={'status':'PASS_ALL_EXACT_OUTPUTS','stats':stats,'rows':rows,
         'inputs':'48 rational spectral-diagonal shrinkage-LDA synthetic experiments',
         'strong_baseline':'same incremental sweep and same exact root backend, without prefilter',
         'limitations':['not real EEG','same CPU, short timing intervals','no external published SOTA baseline',
           'Bernstein positivity and event sweeps are classical ingredients']}
    path=Path('p4_certificate_benchmark_results.json')
    path.write_text(json.dumps(obj,indent=2)+'\n')
    print('P4_EXACT_CERTIFICATE_BENCHMARK_PASS',str(path),flush=True)
if __name__=='__main__':main()
