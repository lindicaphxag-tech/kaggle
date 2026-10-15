#!/usr/bin/env python3
"""Source rank-path complete exact AUC with certified no-event screening.
Exact rational tests only. Classical Bernstein positivity; no new theorem claim.
"""
import sympy as sp
from math import comb
T=sp.Symbol('t',real=True)

def bernstein_coefficients(poly):
    if poly.is_zero:return (sp.Integer(0),)
    d=poly.degree()
    a=[poly.nth(j) for j in range(d+1)]
    return tuple(sum(a[j]*sp.Rational(comb(k,j),comb(d,j)) for j in range(k+1)) for k in range(d+1))

def cannot_have_interior_root(poly):
    if poly.is_zero:return False
    if poly.eval(0)*poly.eval(1)<0:return False
    coeff=[poly.nth(j) for j in range(poly.degree()+1)]
    if (all(x>=0 for x in coeff) and any(x>0 for x in coeff)) or (all(x<=0 for x in coeff) and any(x<0 for x in coeff)):
        return True
    if poly.degree()==2:
        aa=poly.nth(2);bb=poly.nth(1);cc=poly.nth(0)
        u=-bb/(2*aa);vals=[cc,aa+bb+cc]
        if 0<u<1:vals.append(poly.eval(u))
        if min(vals)>0 or max(vals)<0:return True
        if min(vals)<0 and max(vals)>0:return False
        if 0<u<1 and poly.eval(u)==0:return False
    B=bernstein_coefficients(poly)
    return (all(x>=0 for x in B) and any(x>0 for x in B)) or (all(x<=0 for x in B) and any(x<0 for x in B))

def credit(poly,value):
    s=sp.sign(poly.eval(value))
    if s==1:return sp.Integer(1)
    if s==0:return sp.Rational(1,2)
    if s==-1:return sp.Integer(0)
    raise ArithmeticError('undecidable exact sign')

def event_sweep(polys,prune=True):
    polys=list(polys)
    if not polys:raise ValueError('empty')
    found=[];root_calls=0;skipped=0
    for i,p in enumerate(polys):
        if p.is_zero or p.degree()<1:continue
        if prune and cannot_have_interior_root(p):
            skipped+=1;continue
        root_calls+=1
        for root,multiplicity in p.real_roots(multiple=False):
            if sp.sign(root)==1 and sp.sign(1-root)==1:
                found.append((root,i,multiplicity))
    found.sort(key=lambda x:x[0])
    bundles=[]
    for root,i,m in found:
        if not bundles or sp.simplify(root-bundles[-1][0])!=0:bundles.append((root,[(i,m)]))
        else:bundles[-1][1].append((i,m))
    points=[sp.Integer(0)]+[b[0] for b in bundles]+[sp.Integer(1)]
    N=sp.Integer(len(polys))
    state=[credit(p,(points[0]+points[1])/2) for p in polys]
    active=sum(state)/N
    cells=[];boundaries=[(sp.Integer(0),sum(credit(p,0) for p in polys)/N)]
    for j,(left,right) in enumerate(zip(points[:-1],points[1:])):
        cells.append(((left,right),active))
        if j==len(bundles):break
        root,incident=bundles[j];at=active*N;next_total=at
        for idx,m in incident:
            before=state[idx]
            if before not in (sp.Integer(0),sp.Integer(1)):raise ArithmeticError('unexpected interior tie')
            at+=sp.Rational(1,2)-before
            after=1-before if m%2 else before
            next_total+=after-before
            state[idx]=after
        boundaries.append((root,at/N))
        active=next_total/N
    boundaries.append((sp.Integer(1),sum(credit(p,1) for p in polys)/N))
    return {'max_auc':max([c for _,c in boundaries]+[c for _,c in cells]),
            'boundaries':boundaries,'cells':cells,
            'n_pairs':len(polys),'root_calls':root_calls,
            'skipped_no_event':skipped,'distinct_events':len(bundles)}
if __name__=='__main__':
    x=T
    cases=[sp.Poly(x-sp.Rational(1,3),x),sp.Poly(-((x-sp.Rational(1,2))**2),x),sp.Poly(0,x),sp.Poly(1+x*x,x)]
    a=event_sweep(cases,False);b=event_sweep(cases,True)
    assert all(a[k]==b[k] for k in ['max_auc','boundaries','cells'])
    print('P4_CERTIFIED_PRUNED_EXACT_SMOKE_PASS')
