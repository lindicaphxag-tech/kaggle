#!/usr/bin/env python3
"""Continuous fit-selected whole-search score with blocked verification.
No threshold is applied before score ranking. A source-corpus maximum must
be calibrated against source-corpus maximum on independent null records.
Only numerical-content correspondence; not physical patient identity.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import p2_provguard_v4_calibrated as v4

def search_score(source,target,p=v4.Policy(),endpoint='clock'):
    if endpoint not in ('clock','any'):raise ValueError('endpoint')
    s={k:v4.arr(v) for k,v in source.items()}
    t={k:v4.arr(v) for k,v in target.items()}
    sk=sorted(s);tk=sorted(t)
    k=p.min_channels if endpoint=='clock' else 1
    if min(len(sk),len(tk))<k:return {'score':0.,'status':'INSUFFICIENT_CHANNELS'}
    if len({len(x) for x in s.values()})!=1 or len({len(x) for x in t.values()})!=1:
        raise ValueError('uniform channel lengths required for current scorer')
    X=np.vstack([s[x] for x in sk]);Y=np.vstack([t[x] for x in tk])
    best=None
    for lag in range(-p.max_lag,p.max_lag+1):
        a=max(0,-lag);b=max(0,lag);n=min(X.shape[1]-a,Y.shape[1]-b)
        if n<p.min_overlap:continue
        h=int(n*(1-p.verify_fraction));e=h-p.guard_samples
        if e<64 or n-h<64:continue
        fit=v4._corr_matrix(X[:,a:a+e],Y[:,b:b+e])
        if endpoint=='any':
            i,j=np.unravel_index(np.argmax(fit),fit.shape)
            matches=((int(i),int(j)),);rank=float(fit[i,j])
        else:
            ii,jj=linear_sum_assignment(-fit)
            matches=tuple(sorted(zip(map(int,ii),map(int,jj)),key=lambda x:-fit[x[0],x[1]])[:k])
            if len(matches)!=k:continue
            rank=float(min(fit[i,j] for i,j in matches))
        score=(rank,-abs(lag),-lag)
        if best is None or score>best[0]:best=(score,lag,matches,(a,b,n,e,h))
    if best is None:return {'score':0.,'status':'INSUFFICIENT_OVERLAP'}
    _,lag,matches,(a,b,n,e,h)=best
    hold=v4._corr_matrix(X[:,a+h:a+n],Y[:,b+h:b+n])
    return {'score':float(min(hold[i,j] for i,j in matches)),
            'status':'SCORED','endpoint':endpoint,'lag':int(lag),
            'fit_score':best[0][0],
            'matches':[(sk[i],tk[j]) for i,j in matches]}
def corpus_score(sources,target,p=v4.Policy(),endpoint='clock'):
    if not sources:raise ValueError('empty corpus')
    row={name:search_score(s,target,p,endpoint) for name,s in sources.items()}
    name=max(row,key=lambda n:(row[n]['score'],n))
    return {'score':row[name]['score'],'best_source':name,'per_source':row}
def calibrate(sources,target,null_records,p=v4.Policy(),endpoint='clock'):
    z=corpus_score(sources,target,p,endpoint)
    null=[corpus_score(sources,t,p,endpoint)['score'] for t in null_records]
    if not null:raise ValueError('empty independent null')
    value=(1+sum(x>=z['score'] for x in null))/(len(null)+1)
    return dict(score=z['score'],null_scores=null,p_value=value,n_null=len(null),
                decision_0p05=bool(value<=.05),
                assumptions=['exchangeable null recording-level calibrators',
                'same complete corpus search on all calibrators',
                'no repeated acquisition or clinical identity overlap',
                'threshold and family size predeclared'])
