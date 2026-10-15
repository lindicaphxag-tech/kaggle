#!/usr/bin/env python3
"""P4 exact rational Sturm check for roots of source-rank numerator polynomials.
This is a CLASSICAL Sturm construction, NOT a new mathematical theorem.
No end-to-end floating EEG preprocessing guarantee.
"""
from fractions import Fraction as F
def norm(a):
    a=[F(t) for t in a]
    while a and a[-1]==0:a.pop()
    return tuple(a)
def evalp(a,x):
    z=F(0)
    for t in a[::-1]:z=z*x+t
    return z
def deriv(a):return norm([i*a[i] for i in range(1,len(a))])
def remain(a,b):
    a=list(a)
    if not b:raise ZeroDivisionError
    while len(a)>=len(b) and a:
        j=len(a)-len(b);q=a[-1]/b[-1]
        for i,c in enumerate(b):a[i+j]-=q*c
        while a and a[-1]==0:a.pop()
    return norm(a)
def chain(a):
    a=norm(a)
    if len(a)<=1:return [a]
    seq=[a,deriv(a)]
    while seq[-1]:
        r=norm([-x for x in remain(seq[-2],seq[-1])])
        if not r:break
        seq.append(r)
    return seq
def var(seq,x):
    signs=[1 if z>0 else -1 for q in seq if (z:=evalp(q,x))!=0]
    return sum(a!=b for a,b in zip(signs,signs[1:]))
def distinct_roots(coeff,left=F(0),right=F(1)):
    p=norm(coeff);left=F(left);right=F(right)
    if not p or not left<right or evalp(p,left)==0 or evalp(p,right)==0:
        raise ValueError('invalid polynomial, interval or endpoint root')
    s=chain(p)
    return var(s,left)-var(s,right)
if __name__=='__main__':
    assert distinct_roots([F(9,256),F(-3,8),F(11,8),-2,1])==2
    assert distinct_roots([1,0,1])==0
    assert distinct_roots([F(1,4),-1,1])==1
    print('P4_EXACT_STURM_ROOT_TOPOLOGY_PASS synthetic, no novelty claim')
