#!/usr/bin/env python3
"""Exact-rational robustness sweep against centered Horner.
Contrast formula corrected after a duplicate-pattern audit; exploratory.
"""
import argparse,json,hashlib
from fractions import Fraction as Q
from pathlib import Path
from p4_exact_rational_bernstein_audit import numerator,certify,eval_poly
from p4_exact_two_certifiers_benchmark import centered_horner

FAMILIES=('linear','quadratic','alternating')
DIMENSIONS=(2,3,6,9,12)
PARTITIONS=(8,16)
PATTERNS=tuple(range(12))

def spectrum(p,name):
    if name=='linear':return [Q(i+1) for i in range(p)]
    if name=='quadratic':return [Q((i+1)**2+1) for i in range(p)]
    return [Q(2**(i%4)+i+1) for i in range(p)]

def run():
    rows=[]
    for fam in FAMILIES:
      for p in DIMENSIONS:
       s=spectrum(p,fam);nu=sum(s)/p
       for bins in PARTITIONS:
        rec={'family':fam,'dimension':p,'partition':bins,'intervals':0,
             'centered':0,'bernstein':0,'bernstein_only':0,'centered_only':0,
             'opposite_sign_rational_witness':0,'unresolved_without_witness':0}
        normalized_polynomials=[]
        for t in PATTERNS:
         a=[Q((-1)**(i+t)*(i+1+t*(i+1)**2),
              (i+3+((i+2)*t)%7)*(t+2)) for i in range(p)]
         poly=numerator(s,a,nu)
         nonzero=next(v for v in poly if v)
         normalized_polynomials.append(tuple(v/nonzero for v in poly))
         for k in range(bins):
          l,r=Q(k,bins),Q(k+1,bins)
          x=centered_horner(poly,l,r);y=certify(poly,l,r)
          assert not (x.startswith('strict_') and y.startswith('strict_') and x!=y)
          rec['intervals']+=1
          rec['centered']+=x.startswith('strict_')
          rec['bernstein']+=y.startswith('strict_')
          rec['bernstein_only']+=y.startswith('strict_') and x=='unresolved'
          rec['centered_only']+=x.startswith('strict_') and y=='unresolved'
          if x=='unresolved' and y=='unresolved':
           vals=[eval_poly(poly,l+(r-l)*Q(j,128)) for j in range(129)]
           witness=min(vals)<0<max(vals)
           rec['opposite_sign_rational_witness']+=witness
           rec['unresolved_without_witness']+=not witness
        assert len(set(normalized_polynomials))==len(PATTERNS)
        rows.append(rec)
        print('P4_SWEEP',fam,p,bins,rec['bernstein_only'],rec['centered_only'],flush=True)
    total={k:sum(r[k] for r in rows) for k in ('intervals','centered','bernstein','bernstein_only','centered_only','opposite_sign_rational_witness','unresolved_without_witness')}
    return {'status':'EXACT_RATIONAL_CORRECTED_EXPLORATORY_MULTI_FAMILY_SWEEP',
            'families':FAMILIES,'dimensions':DIMENSIONS,'partitions':PARTITIONS,
            'patterns':PATTERNS,'rows':rows,'total':total,
            'contrast_formula_repaired_after_first_sweep_duplicate_audit':True,
            'all_12_polynomials_distinct_up_to_scale_within_each_spectrum':True,
            'rational_inputs_only':True,'no_floating_eeg_pipeline_certified':True,
            'no_accuracy_superiority_claim':True}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    z=ap.parse_args();r=run();z.out.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(r,indent=2,sort_keys=True)+'\n').encode();z.out.write_bytes(data)
    print('P4_MULTI_FAMILY_SHA',hashlib.sha256(data).hexdigest(),r['total'],flush=True)
