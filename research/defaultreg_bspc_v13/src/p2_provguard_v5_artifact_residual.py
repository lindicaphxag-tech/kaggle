#!/usr/bin/env python3
"""ProvGuard V5 candidate: matched-channel common-mode residual witness.
Distinguishes shared coherent channel-specific content from a dominant common
artifact. Only a NUMERICAL predicate, not a physiological source identity proof.
"""
import numpy as np
import p2_provguard_v4_calibrated as core

def _corr(x,y):
    x=x-x.mean();y=y-y.mean()
    d=np.linalg.norm(x)*np.linalg.norm(y)
    return float(abs(x@y/d)) if d>1e-12 else 0.

def residual_evidence(source,target,base):
    xs=[];ys=[]
    for w in base['witnesses']:
        a=core.arr(source[w['source_channel']])
        b=core.arr(target[w['target_channel']])
        s=w['source_span'][0];t=w['target_span'][0];v,n=w['verify_relative']
        xs.append(a[s+v:s+n]);ys.append(b[t+v:t+n])
    X=np.stack(xs);Y=np.stack(ys)
    X=X-X.mean(axis=0,keepdims=True);Y=Y-Y.mean(axis=0,keepdims=True)
    return [_corr(a,b) for a,b in zip(X,Y)]

def match(source,target,p=core.Policy(),residual_threshold=.90):
    if not(0<residual_threshold<=1):raise ValueError('bad threshold')
    base=core.match(source,target,p)
    if base['status']!='NUMERIC_WITNESS':
        return dict(status='ABSTAIN',reason='NO_RAW_COHERENCE')
    if not core.verify(source,target,base,p):raise ValueError('invalid numerical base')
    residual=residual_evidence(source,target,base)
    if not residual or min(residual)<residual_threshold:
        return dict(status='ABSTAIN',reason='COMMON_MODE_RESIDUAL_FAILED',
                    residual_scores=residual,raw_channel_count=len(residual))
    return dict(status='DUAL_WITNESS',base=base,residual_scores=residual,
                residual_threshold=residual_threshold)

def verify(source,target,cert,p=core.Policy(),residual_threshold=.90):
    if cert.get('status')!='DUAL_WITNESS' or cert.get('residual_threshold')!=residual_threshold:
        return False
    base=cert['base']
    if not core.verify(source,target,base,p):return False
    scores=residual_evidence(source,target,base)
    return bool(len(scores)==len(cert.get('residual_scores',[])) and
        min(scores)>=residual_threshold and
        all(abs(a-b)<1e-8 for a,b in zip(scores,cert['residual_scores'])))
