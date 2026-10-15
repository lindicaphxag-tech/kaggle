#!/usr/bin/env python3
"""Experimental ProvGuard v3: fit-only selection, blocked holdout, trusted-policy witness.
Numerical content predicate, NOT physical/patient identity proof. Prototype.
"""
from dataclasses import dataclass,asdict
from scipy.optimize import linear_sum_assignment
import hashlib,numpy as np
@dataclass(frozen=True)
class Policy:
    max_lag:int=100
    min_channels:int=4
    train_threshold:float=.985
    verify_threshold:float=.97
    min_overlap:int=230
    verify_fraction:float=.25
    guard_samples:int=32
def arr(v):
    x=np.asarray(v,dtype=np.float64).ravel()
    if not np.isfinite(x).all():raise ValueError('nonfinite')
    return x
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(arr(a),dtype='<f8').tobytes()).hexdigest()
def corr(a,b):
    if len(a)<8 or len(a)!=len(b):return 0.
    a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/d if d>1e-12 else 0.,-1.,1.))
def blocks(x,y,lag,p):
    a=max(0,-lag);b=max(0,lag);n=min(len(x)-a,len(y)-b)
    if n<p.min_overlap:return None
    v=int(n*(1-p.verify_fraction));end=v-p.guard_samples
    if end<64 or n-v<64:return None
    return a,b,n,end,v,abs(corr(x[a:a+end],y[b:b+end])),abs(corr(x[a+v:a+n],y[b+v:b+n]))
def match(source,target,p=Policy()):
    if not(0<=p.max_lag<=10000 and 2<=p.min_channels and
           0<p.train_threshold<=1 and 0<p.verify_threshold<=1 and
           0<p.verify_fraction<.5 and p.min_overlap>=120 and
           0<=p.guard_samples<p.min_overlap//3):raise ValueError('policy')
    s={k:arr(v) for k,v in source.items()}
    t={k:arr(v) for k,v in target.items()}
    ss=sorted(s);tt=sorted(t);abstain={'status':'ABSTAIN','policy':asdict(p),'witnesses':[]}
    if min(len(ss),len(tt))<p.min_channels:return abstain
    best=None
    for lag in range(-p.max_lag,p.max_lag+1):
        w=np.zeros((len(ss),len(tt)));cache={}
        for i,sc in enumerate(ss):
            for j,tc in enumerate(tt):
                b=blocks(s[sc],t[tc],lag,p)
                if b is not None and b[5]>=p.train_threshold:
                    w[i,j]=b[5];cache[(i,j)]=b
        ii,jj=linear_sum_assignment(-w)
        hits=[(int(i),int(j)) for i,j in zip(ii,jj) if w[i,j]>0]
        if len(hits)<p.min_channels:continue
        score=(len(hits),min(w[i,j] for i,j in hits),sum(w[i,j] for i,j in hits))
        if best is None or score>best[0]:best=(score,lag,hits,cache)
    if best is None:return abstain
    _,lag,hits,cache=best
    # Verification never chooses a second lag or rematches to rescue failed fit-only selection.
    if any(cache[(i,j)][6]<p.verify_threshold for i,j in hits):return abstain
    ww=[]
    for i,j in hits:
        a,b,n,end,v,c,h=cache[(i,j)]
        ww.append(dict(source_channel=ss[i],target_channel=tt[j],lag=lag,
           source_span=[a,a+n],target_span=[b,b+n],train_relative=[0,end],
           verify_relative=[v,n],train_abs_corr=round(c,10),verify_abs_corr=round(h,10),
           source_sha256_f64=digest(s[ss[i]]),target_sha256_f64=digest(t[tt[j]])))
    return dict(status='NUMERIC_WITNESS',policy=asdict(p),lag=lag,
                matched_channels=len(ww),witnesses=ww)
def verify(source,target,cert,p=Policy()):
    try:
        if cert['status']!='NUMERIC_WITNESS' or cert['policy']!=asdict(p):return False
        lag=cert['lag'];ww=cert['witnesses']
        if type(lag)!=int or abs(lag)>p.max_lag or len(ww)<p.min_channels or len(ww)!=cert['matched_channels']:return False
        if len(set(w['source_channel'] for w in ww))!=len(ww) or len(set(w['target_channel'] for w in ww))!=len(ww):return False
        for w in ww:
            src=arr(source[w['source_channel']]);tgt=arr(target[w['target_channel']])
            if w['lag']!=lag:return False
            b=blocks(src,tgt,lag,p)
            if b is None:return False
            a,y,n,end,v,c,h=b
            if c<p.train_threshold or h<p.verify_threshold:return False
            if (w['source_span']!=[a,a+n] or w['target_span']!=[y,y+n] or
                w['train_relative']!=[0,end] or w['verify_relative']!=[v,n]):return False
            if abs(w['train_abs_corr']-c)>1e-8 or abs(w['verify_abs_corr']-h)>1e-8:return False
            if w['source_sha256_f64']!=digest(src) or w['target_sha256_f64']!=digest(tgt):return False
        return True
    except (ValueError,KeyError,TypeError,IndexError,AttributeError,ZeroDivisionError):return False
def identified_components(ids,all_confirmed_reuse_edges):
    ids=list(ids);parent={x:x for x in ids}
    if len(ids)!=len(parent):raise ValueError('duplicate IDs')
    def root(x):
        while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for a,b in all_confirmed_reuse_edges:
        if a not in parent or b not in parent:raise ValueError('unknown ID')
        parent[root(a)]=root(b)
    d={}
    for x in ids:d.setdefault(root(x),[]).append(x)
    return list(d.values())
def valid_split(ids,all_confirmed_reuse_edges,train,test):
    train=set(train);test=set(test)
    if train&test or not train|test<=set(ids):return False
    return all(not(train&set(g) and test&set(g)) for g in identified_components(ids,all_confirmed_reuse_edges))
