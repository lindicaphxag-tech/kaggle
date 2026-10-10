"""Exact Euclidean projection onto the reachable actuator actions WITH two clips.

Our external Reacher model has an engineered inner clip followed by the
official Gymnasium physical action clip. A bare parallelogram projection would
ignore the latter, making an ostensibly omniscient comparison too weak.

For 2D, let K=C([-L,L]^2), and f(y)=componentwise clip(y,[-B,B]).
Partition K by x and y thresholds -B,+B into 9 convex polygons. On each
piece, f is affine, so its image is a convex polygon whose vertices are
images of the clipped polygon vertices. Exactly project target onto each of
9 projected polygons and take the smallest distance. This is a global
one-step Euclidean projection over the COMPLETE reachability set f(K).
No learned controller, source authorization, or global future-reward
optimality is implied.
"""
from __future__ import annotations

import math
from typing import Sequence

Vec=tuple[float,float]
Mat=tuple[Vec,Vec]
EPS=1e-9


def mm(a:Mat,v:Vec)->Vec:
    return (a[0][0]*v[0]+a[0][1]*v[1],
            a[1][0]*v[0]+a[1][1]*v[1])


def clip_scalar(x:float,bound:float)->float:
    return min(bound,max(-bound,x))


def clip_polygon(poly:Sequence[Vec],axis:int, direction:int,threshold:float)->list[Vec]:
    """Keep direction*(x_axis-threshold) <= 0."""
    if not poly:return []
    def signed(p):return direction*(p[axis]-threshold)
    kept=[]
    prev=poly[-1]
    prev_val=signed(prev)
    for curr in poly:
        curr_val=signed(curr)
        inside_curr=curr_val<=EPS
        inside_prev=prev_val<=EPS
        if inside_curr!=inside_prev:
            weight=prev_val/(prev_val-curr_val)
            kept.append((prev[0]+weight*(curr[0]-prev[0]),
                         prev[1]+weight*(curr[1]-prev[1])))
        if inside_curr:
            kept.append(curr)
        prev,prev_val=curr,curr_val
    return kept


def _cross(o:Vec,a:Vec,b:Vec)->float:
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])


def hull(points:Sequence[Vec])->list[Vec]:
    """Monotone chain; zero-area images remain point or segment."""
    xs=sorted(set((round(float(x),14),round(float(y),14)) for x,y in points))
    if len(xs)<=1:return xs
    lo=[]
    for p in xs:
        while len(lo)>=2 and _cross(lo[-2],lo[-1],p)<=1e-13:lo.pop()
        lo.append(p)
    hi=[]
    for p in reversed(xs):
        while len(hi)>=2 and _cross(hi[-2],hi[-1],p)<=1e-13:hi.pop()
        hi.append(p)
    return lo[:-1]+hi[:-1]


def nearest_convex_polygon(target:Vec,vertices:Sequence[Vec])->Vec:
    if not vertices:raise ValueError("no reachable vertices")
    if len(vertices)>=3 and all(
        _cross(vertices[i],vertices[(i+1)%len(vertices)],target)>=-1e-11
        for i in range(len(vertices))
    ):
        return target
    choices=list(vertices)
    for i in range(len(vertices)):
        a=vertices[i]
        b=vertices[(i+1)%len(vertices)]
        dx,dy=b[0]-a[0],b[1]-a[1]
        den=dx*dx+dy*dy
        if den<=0:continue
        t=max(0.,min(1.,((target[0]-a[0])*dx+(target[1]-a[1])*dy)/den))
        choices.append((a[0]+t*dx,a[1]+t*dy))
    return min(choices,key=lambda p:math.dist(target,p))


def output_reachability_polygons(C:Mat, *,
                                 inner_bound:float=1.,
                                 outer_bound:float=1.) -> list[list[Vec]]:
    if any(not math.isfinite(x) for row in C for x in row) or \
       not all(math.isfinite(x) and x>0 for x in (inner_bound,outer_bound)):
        raise ValueError("finite invertible geometry with positive clipping required")
    det=C[0][0]*C[1][1]-C[0][1]*C[1][0]
    if abs(det)<1e-12:raise ValueError("singular interface source")
    original=[mm(C,z) for z in ((-inner_bound,-inner_bound),
             (inner_bound,-inner_bound),(inner_bound,inner_bound),(-inner_bound,inner_bound))]
    polys=[]
    for xr in (-1,0,1):
        for yr in (-1,0,1):
            poly=list(original)
            for axis,region in ((0,xr),(1,yr)):
                if region==-1:
                    poly=clip_polygon(poly,axis,1,-outer_bound)
                elif region==1:
                    poly=clip_polygon(poly,axis,-1,outer_bound)
                else:
                    poly=clip_polygon(poly,axis,-1,-outer_bound)
                    poly=clip_polygon(poly,axis,1,outer_bound)
                if not poly:break
            if poly:
                projected=[(clip_scalar(z[0],outer_bound),
                            clip_scalar(z[1],outer_bound)) for z in poly]
                image=hull(projected)
                if image:polys.append(image)
    if not polys:raise AssertionError("nonempty source action image must exist")
    return polys


def optimal_clipped_output(target:Vec,polygons:Sequence[Sequence[Vec]])->dict:
    """Exact best per-step output attainable after BOTH clips."""
    if not all(math.isfinite(x) for x in target):
        raise ValueError("nonfinite reference command")
    projections=[nearest_convex_polygon(target,p) for p in polygons if p]
    selected=min(projections,key=lambda p:math.dist(p,target))
    return {"action":selected,"minimum_euclidean_error":math.dist(target,selected),
            "oracle_scope":"exact instantaneous action tracking after two clips, NOT optimal future task reward"}


if __name__=="__main__":
    import json
    C=((1.,-.8),(0.,1.))
    polys=output_reachability_polygons(C)
    print(json.dumps(optimal_clipped_output((.8,.8),polys),indent=2))
