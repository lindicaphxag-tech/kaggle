#!/usr/bin/env python3
"""Exact rational joint spectral/score-box Bernstein rank certificate.
Scope: positive eigenvalue intervals and independently bounded score
coefficients in fixed spectral coordinates, not the EEG float pipeline.
"""
from fractions import Fraction as Q
from math import comb
import itertools,random,argparse,json,hashlib
from pathlib import Path

def rat(v):
    if isinstance(v,float): raise TypeError('binary float is not exact input')
    return Q(v)

def interval(lo,hi=None):
    a=rat(lo); b=a if hi is None else rat(hi)
    if a>b: raise ValueError('inverted interval')
    return (a,b)

def add(x,y): return (x[0]+y[0],x[1]+y[1])

def mul(x,y):
    z=[a*b for a in x for b in y]
    return min(z),max(z)

def bern_mul(a,b):
    m=len(a)-1;n=len(b)-1
    out=[interval(0) for _ in range(m+n+1)]
    for i in range(m+1):
        for j in range(n+1):
            weight=Q(comb(m,i)*comb(n,j),comb(m+n,i+j))
            ab=mul(a[i],b[j])
            out[i+j]=add(out[i+j],(weight*ab[0],weight*ab[1]))
    return out

def certify(spectral_bounds,coefficients,errors,left='0',right='1'):
    p=len(spectral_bounds)
    if p<2 or len(coefficients)!=p or len(errors)!=p:
        raise ValueError('dimension mismatch')
    s=[interval(*pair) for pair in spectral_bounds]
    a=list(map(rat,coefficients));e=list(map(rat,errors))
    L,R=rat(left),rat(right)
    if not (0<=L<R<=1):raise ValueError('invalid lambda interval')
    if any(lo<=0 for lo,hi in s):raise ValueError('spectrum not SPD')
    if any(v<0 for v in e):raise ValueError('negative error')
    nu=(sum((x[0] for x in s),Q(0))/p,
        sum((x[1] for x in s),Q(0))/p)
    q=[]
    for lo,hi in s:
        qleft=interval((1-L)*lo+L*nu[0],(1-L)*hi+L*nu[1])
        qright=interval((1-R)*lo+R*nu[0],(1-R)*hi+R*nu[1])
        assert qleft[0]>0 and qright[0]>0
        q.append([qleft,qright])
    coefficients_N=[interval(0) for _ in range(p)]
    for i in range(p):
        basis=[interval(1)]
        for j in range(p):
            if j!=i: basis=bern_mul(basis,q[j])
        box=interval(a[i]-e[i],a[i]+e[i])
        coefficients_N=[add(old,mul(box,bi)) for old,bi in zip(coefficients_N,basis)]
    if all(v[0]>0 for v in coefficients_N):status='POS'
    elif all(v[1]<0 for v in coefficients_N):status='NEG'
    else:status='UNKNOWN'
    return {'status':status,
            'bernstein_bounds':[[str(lo),str(hi)] for lo,hi in coefficients_N],
            'dimension':p,'eigenvalue_input_boxes_external':True}

def exact_score(s,a,t):
    nu=sum(s,Q(0))/len(s)
    return sum((x/((1-t)*v+t*nu) for v,x in zip(s,a)),Q(0))

def regression(n=126):
    assert certify([[1,1],[3,3]],[1,1],[0,0])['status']=='POS'
    assert certify([[1,2],[2,5]],[-1,-1],[0,0])['status']=='NEG'
    assert certify([[1,2],[2,5]],[0,0],[1,1])['status']=='UNKNOWN'
    assert certify([['1/100000','1/1000'],[4,9]],[1,1],[0,0])['status']=='POS'
    for bad in [([[0,1],[2,3]],[1,1],[0,0]),
                ([[1,2],[2,3]],[1,1],[-1,0]),
                ([[1,2],[2,3]],[1,1.0],[0,0])]:
        try:certify(*bad)
        except (TypeError,ValueError):pass
        else:raise AssertionError('should fail closed')
    rng=random.Random(20261013);counts={'POS':0,'NEG':0,'UNKNOWN':0}
    checks=0
    for z in range(n):
        p=rng.choice([2,3,4,5])
        spectral=[]
        for i in range(p):
            lo=Q(rng.randint(1,16),rng.randint(1,7))
            hi=lo+Q(rng.randint(0,7),rng.randint(2,12))
            spectral.append([str(lo),str(hi)])
        if z%3==0:a=[Q(rng.randint(1,14),13) for _ in range(p)]
        elif z%3==1:a=[-Q(rng.randint(1,14),13) for _ in range(p)]
        else:a=[Q(rng.randint(-13,13),13) for _ in range(p)]
        eps=[Q(rng.randint(0,5),20) for _ in range(p)]
        L=Q(rng.randint(0,2),8);R=Q(rng.randint(6,8),8)
        decision=certify(spectral,[str(v) for v in a],
                         [str(v) for v in eps],str(L),str(R))['status']
        counts[decision]+=1
        if decision!='UNKNOWN':
            for s_bits in itertools.product((0,1),repeat=p):
                s=[Q(spectral[i][s_bits[i]]) for i in range(p)]
                for a_bits in itertools.product((-1,1),repeat=p):
                    aa=[a[i]+a_bits[i]*eps[i] for i in range(p)]
                    for t in [L,(3*L+R)/4,(L+R)/2,(L+3*R)/4,R]:
                        value=exact_score(s,aa,t)
                        assert value>0 if decision=='POS' else value<0
                        checks+=1
    return {'status':'PASS','synthetic_joint_boxes':n,
            'decisions':counts,'exact_corner_oracle_checks':checks,
            'seed':20261013,'uncertainty_scope':'rational spectral boxes and score coefficient boxes',
            'not_float_EEG_pipeline_proof':True,'no_real_EEG_transfer_gain':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args()
    r=regression();data=json.dumps(r,indent=2,sort_keys=True)+'\n'
    print(data)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data)
        print('P4_BOX_SHA256',hashlib.sha256(data.encode()).hexdigest())
