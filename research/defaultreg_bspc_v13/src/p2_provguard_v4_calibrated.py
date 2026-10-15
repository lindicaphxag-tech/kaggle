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
def _corr_matrix(X,Y):
    X=X-X.mean(axis=1,keepdims=True);Y=Y-Y.mean(axis=1,keepdims=True)
    denom=np.linalg.norm(X,axis=1)[:,None]*np.linalg.norm(Y,axis=1)[None,:]
    return np.divide(np.abs(X@Y.T),denom,out=np.zeros_like(denom),where=denom>1e-12)

def match(source,target,p=Policy()):
    """Training-only lag/channel selection with batched Pearson for uniform channel lengths."""
    if not(0<=p.max_lag<=10000 and 2<=p.min_channels and
           0<p.train_threshold<=1 and 0<p.verify_threshold<=1 and
           0<p.verify_fraction<.5 and p.min_overlap>=120 and
           0<=p.guard_samples<p.min_overlap//3):raise ValueError('policy')
    s={k:arr(v) for k,v in source.items()}
    t={k:arr(v) for k,v in target.items()}
    ss=sorted(s);tt=sorted(t)
    abstain={'status':'ABSTAIN','policy':asdict(p),'witnesses':[]}
    if min(len(ss),len(tt))<p.min_channels:return abstain
    uniform=(len({len(x) for x in s.values()})==1 and
             len({len(x) for x in t.values()})==1)
    if uniform:
        X=np.vstack([s[k] for k in ss]);Y=np.vstack([t[k] for k in tt])
    best=None
    for lag in range(-p.max_lag,p.max_lag+1):
        if uniform:
            a=max(0,-lag);b=max(0,lag);n=min(X.shape[1]-a,Y.shape[1]-b)
            if n<p.min_overlap:continue
            v=int(n*(1-p.verify_fraction));e=v-p.guard_samples
            if e<64 or n-v<64:continue
            m=_corr_matrix(X[:,a:a+e],Y[:,b:b+e])
            w=np.where(m>=p.train_threshold,m,0.)
        else:
            w=np.zeros((len(ss),len(tt)))
            for i,sc in enumerate(ss):
                for j,tc in enumerate(tt):
                    block=blocks(s[sc],t[tc],lag,p)
                    if block is not None and block[5]>=p.train_threshold:
                        w[i,j]=block[5]
        ii,jj=linear_sum_assignment(-w)
        hits=[(int(i),int(j)) for i,j in zip(ii,jj) if w[i,j]>0]
        if len(hits)<p.min_channels:continue
        score=(len(hits),min(w[i,j] for i,j in hits),sum(w[i,j] for i,j in hits))
        if best is None or score>best[0]:best=(score,lag,hits)
    if best is None:return abstain
    _,lag,hits=best
    ww=[]
    for i,j in hits:
        a,b,n,end,v,c,h=blocks(s[ss[i]],t[tt[j]],lag,p)
        if c<p.train_threshold or h<p.verify_threshold:return abstain
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

# Statistical calibration utility: do NOT fit null calibration on transformed replicas of
# one recording, or on EEG samples treated as independent persons.
def verified_score(cert):
    if cert.get('status')!='NUMERIC_WITNESS':return 0.0
    return min(float(w['verify_abs_corr']) for w in cert['witnesses'])

def null_pvalue(score,recording_level_null_scores):
    """Whole-search rank p-value under fixed pipeline and null-score exchangeability."""
    z=np.asarray(recording_level_null_scores,dtype=float).ravel()
    if z.size==0 or not np.isfinite(z).all() or not 0<=score<=1 or np.any((z<0)|(z>1)):
        raise ValueError('invalid null calibration')
    return (1+int(np.sum(z>=score)))/(len(z)+1)

def familywise_accept(scores,null_scores,alpha=.05):
    """Bonferroni under null exchangeability, fixed family size. Reject all zero scores."""
    scores=list(scores)
    if not scores or not 0<alpha<1:raise ValueError('family')
    return [bool(v>0 and null_pvalue(v,null_scores)<=alpha/len(scores)) for v in scores]

# Distinct endpoint for content-safe splitting: ONE digitally reused channel is enough.
# A cross-channel asynchronous splice is NEGATIVE for coherent-clock identity but
# POSITIVE for ANY-CHANNEL content reuse. Never mix their labels or confusion matrices.
def any_content_witness(source,target,p=Policy()):
    if p.min_channels<2:raise ValueError('policy')
    s={k:arr(v) for k,v in source.items()};t={k:arr(v) for k,v in target.items()}
    ss=sorted(s);tt=sorted(t)
    failed=dict(status='ABSTAIN',endpoint='ANY_CHANNEL_CONTENT',policy=asdict(p))
    if not ss or not tt:return failed
    uniform=(len({len(x) for x in s.values()})==1 and len({len(x) for x in t.values()})==1)
    if uniform:
        X=np.vstack([s[k] for k in ss]);Y=np.vstack([t[k] for k in tt])
    best=None
    for lag in range(-p.max_lag,p.max_lag+1):
        if uniform:
            a=max(0,-lag);b=max(0,lag);n=min(X.shape[1]-a,Y.shape[1]-b)
            if n<p.min_overlap:continue
            v=int(n*(1-p.verify_fraction));e=v-p.guard_samples
            if e<64 or n-v<64:continue
            m=_corr_matrix(X[:,a:a+e],Y[:,b:b+e])
            i,j=np.unravel_index(int(np.argmax(m)),m.shape)
            score=float(m[i,j])
            if score<p.train_threshold:continue
            z=(score,-abs(lag),lag,int(i),int(j))
        else:
            z=None
            for i,sc in enumerate(ss):
                for j,tc in enumerate(tt):
                    row=blocks(s[sc],t[tc],lag,p)
                    if row is None or row[5]<p.train_threshold:continue
                    v=(row[5],-abs(lag),lag,i,j)
                    if z is None or v>z:z=v
        if z is not None and (best is None or z>best):best=z
    if best is None:return failed
    _,_,lag,i,j=best
    a,b,n,e,v,fit,hold=blocks(s[ss[i]],t[tt[j]],lag,p)
    if hold<p.verify_threshold:return failed
    return dict(status='NUMERIC_WITNESS',endpoint='ANY_CHANNEL_CONTENT',
        policy=asdict(p),lag=lag,source_channel=ss[i],target_channel=tt[j],
        source_span=[a,a+n],target_span=[b,b+n],train_relative=[0,e],
        verify_relative=[v,n],train_abs_corr=round(fit,10),
        verify_abs_corr=round(hold,10),
        source_sha256_f64=digest(s[ss[i]]),target_sha256_f64=digest(t[tt[j]]))

def verify_any_content(source,target,w,p=Policy()):
    try:
        if w['status']!='NUMERIC_WITNESS' or w['endpoint']!='ANY_CHANNEL_CONTENT' or w['policy']!=asdict(p):return False
        lag=w['lag']
        if type(lag)!=int or abs(lag)>p.max_lag:return False
        x=arr(source[w['source_channel']]);y=arr(target[w['target_channel']])
        result=blocks(x,y,lag,p)
        if result is None:return False
        a,b,n,e,v,fit,hold=result
        return bool(fit>=p.train_threshold and hold>=p.verify_threshold and
            w['source_span']==[a,a+n] and w['target_span']==[b,b+n] and
            w['train_relative']==[0,e] and w['verify_relative']==[v,n] and
            abs(w['train_abs_corr']-fit)<1e-8 and abs(w['verify_abs_corr']-hold)<1e-8 and
            w['source_sha256_f64']==digest(x) and w['target_sha256_f64']==digest(y))
    except (KeyError,ValueError,TypeError,IndexError,AttributeError):return False
