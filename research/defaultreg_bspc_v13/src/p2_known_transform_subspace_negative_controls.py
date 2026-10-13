#!/usr/bin/env python3
"""Known-transform synthetic EEG-like signal-product association.
No patient/acquisition identity inference. Null calibration is synthetic only.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import beta

def make_record(rng,c=7,n=192):
    z=rng.standard_normal((c,n));x=np.zeros_like(z)
    for k,w in enumerate([.32,.24,.18,.14,.08,.04]):x[:,k:]+=w*z[:,:n-k]
    phase=rng.uniform(-np.pi,np.pi,size=(c,1))
    hz=rng.uniform(3,13,size=(c,1))
    return x+.35*np.sin(2*np.pi*hz*np.arange(n)[None,:]/n+phase)

def transform(x,kind,rng):
    if kind=='identical':return x.copy()
    if kind=='permutation_gain':
        perm=rng.permutation(len(x))
        gain=rng.choice([-1.,1.],size=(len(x),1))*rng.uniform(.7,1.5,size=(len(x),1))
        return gain*x[perm]+rng.normal(0,2,size=(len(x),1))
    if kind=='rereference':return x-x.mean(axis=0,keepdims=True)
    if kind=='mix':return rng.standard_normal((len(x),len(x)))@x
    if kind=='shift':return np.roll(x,3,axis=1)
    if kind=='noise':return x+rng.standard_normal(x.shape)*.08
    raise ValueError(kind)

def rowspace(x):
    x=x-x.mean(axis=1,keepdims=True)
    _u,s,vh=np.linalg.svd(x,full_matrices=False)
    k=int(np.sum(s>s[0]*1e-10)) if s[0]>0 else 0
    if not k:raise ValueError('rank-zero input')
    return vh[:k].T

def subspace_score(x,y):
    a=rowspace(x);b=rowspace(y)
    return 1-min(1,float(np.linalg.norm(b-a@(a.T@b),'fro')/np.sqrt(b.shape[1])))

def pearson_score(x,y):
    return abs(float(np.corrcoef(x.reshape(-1),y.reshape(-1))[0,1]))

def ci(k,n):
    return [0. if not k else float(beta.ppf(.025,k,n-k+1)),
            1. if k==n else float(beta.ppf(.975,k+1,n-k))]

def run(ncal=120,ntest=120):
    rng=np.random.default_rng(20261013)
    negatives=[(subspace_score(make_record(rng),make_record(rng)),
                pearson_score(make_record(rng),make_record(rng))) for _ in range(ncal)]
    # Values exceed all fitted negative controls, but no population FPR guarantee.
    thresholds=np.asarray(negatives).max(axis=0)
    tst=np.random.default_rng(20261014)
    kinds=['identical','permutation_gain','rereference','mix','shift','noise']
    rows={k:{'subspace_accept':0,'pearson_accept':0,'cases':0} for k in kinds}
    fp=[0,0]
    for _ in range(ntest):
        x=make_record(tst);other=make_record(tst)
        fp[0]+=int(subspace_score(x,other)>thresholds[0])
        fp[1]+=int(pearson_score(x,other)>thresholds[1])
        for kind in kinds:
            y=transform(x,kind,tst);v=rows[kind]
            v['cases']+=1
            v['subspace_accept']+=int(subspace_score(x,y)>thresholds[0])
            v['pearson_accept']+=int(pearson_score(x,y)>thresholds[1])
    for v in rows.values():
        v['subspace_rate']=v['subspace_accept']/v['cases']
        v['pearson_rate']=v['pearson_accept']/v['cases']
    return dict(status='DISJOINT_SYNTHETIC_PRODUCT_CHECK',seed_cal=20261013,
                seed_test=20261014,n_calibration=ncal,n_test=ntest,
                thresholds=[float(x) for x in thresholds],
                heldout_false_links_subspace=fp[0],heldout_false_links_pearson=fp[1],
                synthetic_FPR_interval_subspace=ci(fp[0],ntest),
                by_transform=rows,no_real_eeg_FPR_calibration=True,
                no_patient_identity_claim=True,unmatched_means_UNKNOWN=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path)
    args=ap.parse_args();r=run();print(json.dumps(r,sort_keys=True))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        data=json.dumps(r,sort_keys=True,indent=2)+'\n';args.out.write_text(data)
        print('P2_SYNTH_KNOWN_TRANSFORM_SHA',hashlib.sha256(data.encode()).hexdigest())
