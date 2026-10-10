"""Bounded-precision ROBUST UPSTREAM control under uncertain robot source faults.

The earlier proposed minimax over a common physical output y in an actuator
reachable set K is sound ONLY if a unique physical y can actually be
commanded identically in all indistinguishable source worlds. Normally a
blackbox sends ONE *upstream* action u, and EACH unknown source world
produces a different physical F_w(u). A union of reachable physical output
polygons is NOT enough to certify an upstream controller.

Correct problem for a hidden-world observation fiber W:

    rho* = min_{u in [-R0,R0]x[-R1,R1]}
                 max_{w in W} ||F_w(u) - desired_w||_infinity,

    F_w(u) = clip(C_w clip(P_w u, -L_w,+L_w), -B_w,+B_w).

This algorithm uses finite interval branch-and-bound on original COMMON
upstream u, not relaxed independently chosen per-world physical outputs.
Every branch holds a SOUND lower bound under exact real arithmetic
(standard interval range inclusion + global operator Lipschitz bound)
and each sampled upstream u yields a feasible upper bound and a
verifiable witness after the full clipped source maps.

It returns 3 values for requested error tolerance:
- SAT_WITH_UPSTREAM_WITNESS iff physically feasible sampled u satisfies it;
- UNSAT_WITH_LOWER_BOUND iff lower bound exceeds the allowed error;
- UNKNOWN_BUDGET_OR_PRECISION otherwise (NEVER authorized).

Branch&bound optimization is exponential in general. No guarantee of
fast convergence on arbitrary source maps. Polynomial/interval proofs
assume source matrices, hidden-world set, source clips and physical bounds
are TRUSTED EXACT models. Python float arithmetic here is not a directed
rounding / formally proved IEEE interval implementation. Use margin as
a conservative numerical tolerance; never claim machine-level soundness
of proof near its boundary or cryptographic source provenance.

In particular, this helps eliminate a potential overclaim in earlier
SemRepair "exact action oracle": it had allowed an unattainable
physical output to be selected independently of a bounded common input.
"""
from __future__ import annotations
from heapq import heappush,heappop
from itertools import count
from math import isfinite

def _square_matrix(mat):
    if not isinstance(mat,(tuple,list)) or len(mat)!=2 or any(
        not isinstance(row,(tuple,list)) or len(row)!=2 or any(
            type(z) not in (float,int) or not isfinite(z) for z in row)
        for row in mat):
        raise ValueError("two-by-two finite physical source matrix required")
    return tuple(tuple(float(z) for z in row) for row in mat)

def _v2(x,positive=False):
    if not isinstance(x,(tuple,list)) or len(x)!=2 or any(
        type(z) not in (float,int) or not isfinite(z) or
        (positive and z<=0) for z in x):
        raise ValueError("finite bounded 2D action coordinate required")
    return tuple(float(z) for z in x)

def _clip(x,radius):
    return min(radius,max(-radius,x))

def _mul(mat,u):
    return tuple(sum(mat[i][j]*u[j] for j in range(2)) for i in range(2))

def physical_command(world,u):
    z=_mul(world["P"],u)
    z=tuple(_clip(z[i],world["L"][i]) for i in range(2))
    y=_mul(world["C"],z)
    return tuple(_clip(y[i],world["B"][i]) for i in range(2))

def _world(x):
    if not isinstance(x,dict) or set(x)!={"P","C","L","B","desired"}:
        raise ValueError("strict source-world schema")
    w={"P":_square_matrix(x["P"]),
       "C":_square_matrix(x["C"]),
       "L":_v2(x["L"],True),"B":_v2(x["B"],True),
       "desired":_v2(x["desired"])}
    return w

def _interval_mul(M,lo,hi):
    result=[]
    for row in M:
        low=sum(row[j]* (lo[j] if row[j]>=0 else hi[j]) for j in range(2))
        high=sum(row[j]*(hi[j] if row[j]>=0 else lo[j]) for j in range(2))
        result.append((low,high))
    return tuple(z[0] for z in result),tuple(z[1] for z in result)

def _box_world_lower(w,lo,hi):
    """Guaranteed source F_w(box) enclosure via interval propagation."""
    pl,ph=_interval_mul(w["P"],lo,hi)
    ml=tuple(_clip(pl[i],w["L"][i]) for i in range(2))
    mh=tuple(_clip(ph[i],w["L"][i]) for i in range(2))
    cl,ch=_interval_mul(w["C"],ml,mh)
    yl=tuple(_clip(cl[i],w["B"][i]) for i in range(2))
    yh=tuple(_clip(ch[i],w["B"][i]) for i in range(2))
    return max(
        (yl[i]-w["desired"][i] if w["desired"][i]<yl[i] else
         w["desired"][i]-yh[i] if w["desired"][i]>yh[i] else 0.)
        for i in range(2)
    )

def _lipschitz(w):
    pinf=max(sum(abs(x) for x in row) for row in w["P"])
    cinf=max(sum(abs(x) for x in row) for row in w["C"])
    return pinf*cinf

def _loss(worlds,u):
    return max(abs(z-w["desired"][i])
               for w in worlds
               for i,z in enumerate(physical_command(w,u)))

def _cell_lower(worlds,lo,hi):
    center=tuple((a+b)/2 for a,b in zip(lo,hi))
    radius=max((hi[i]-lo[i])/2 for i in range(2))
    e=_loss(worlds,center)
    interval_lb=max(_box_world_lower(w,lo,hi) for w in worlds)
    lips_lb=max(0.,e-max(_lipschitz(w) for w in worlds)*radius)
    # Under ideal real arithmetic both bounds independently valid.
    return max(interval_lb,lips_lb),e,center

def solve_common_upstream_minimax(
    *,worlds,input_radii=(1.,1.),
    accepted_error:float=.01,accuracy:float=.001,
    max_nodes:int=10000,numerical_margin:float=1e-10
):
    if not isinstance(worlds,(tuple,list)) or not worlds or len(worlds)>8:
        raise ValueError("1 to 8 possible unknown interface worlds required")
    ws=tuple(_world(x) for x in worlds)
    R=_v2(input_radii,True)
    for label,x in (("accepted error",accepted_error),
                    ("accuracy",accuracy),
                    ("numerical margin",numerical_margin)):
        if type(x) not in (float,int) or not isfinite(x) or \
           (x<=0 if label=="accuracy" else x<0):
            raise ValueError(label+" must be finite nonnegative")
    if type(max_nodes) is not int or max_nodes<1 or max_nodes>100000:
        raise ValueError("bounded positive branch budget <=100000")
    lo=(-R[0],-R[1]);hi=(R[0],R[1])
    initial_lb,initial_ub,u=_cell_lower(ws,lo,hi)
    best=initial_ub
    witness=u
    heap=[];ticket=count()
    heappush(heap,(initial_lb,next(ticket),lo,hi))
    expanded=0
    while heap and expanded<max_nodes:
        lower=heap[0][0]
        if best-lower<=accuracy or lower>accepted_error+numerical_margin:
            break
        lb,_,low,high=heappop(heap)
        if lb>=best:
            continue
        span=[high[i]-low[i] for i in range(2)]
        axis=0 if span[0]>=span[1] else 1
        center=(low[axis]+high[axis])/2
        for side in (0,1):
            a=list(low);b=list(high)
            if side==0:b[axis]=center
            else:a[axis]=center
            a=tuple(a);b=tuple(b)
            cell_lb,cell_best,candidate=_cell_lower(ws,a,b)
            if cell_best<best:
                best=cell_best
                witness=candidate
            if cell_lb<best:
                heappush(heap,(cell_lb,next(ticket),a,b))
        expanded+=1
    remaining_min=min((node[0] for node in heap),default=best)
    lower=min(best,remaining_min)
    exact_witness_error=_loss(ws,witness)
    if exact_witness_error>best+1e-8:
        raise AssertionError("source-modeled common upstream witness not replayable")
    if best+numerical_margin<=accepted_error:
        decision="SAT_WITH_COMMON_UPSTREAM_WITNESS"
    elif lower-numerical_margin>accepted_error:
        decision="UNSAT_WITH_INTERVAL_LOWER_BOUND"
    else:
        decision="UNKNOWN_BUDGET_OR_PRECISION"
    return {
        "decision":decision,
        "number_of_hidden_worlds":len(ws),
        "unique_common_upstream_witness":list(witness),
        "per_world_ACTUAL_physical_commands":[list(physical_command(w,witness)) for w in ws],
        "actual_common_upstream_max_error":exact_witness_error,
        "global_minimax_lower_under_assumed_exact_model":lower,
        "global_minimax_upper_under_assumed_exact_model":best,
        "optimality_bracket_width":best-lower,
        "expanded_interval_boxes":expanded,
        "retained_boxes":len(heap),
        "source_model_exact_real_arithmetic_assumed":True,
        "floating_point_directed_rounding_not_verified":True,
        "full_future_robot_task_optimality":False,
        "generic_program_synthesis":False,
        "can_authorize_policy_command":decision=="SAT_WITH_COMMON_UPSTREAM_WITNESS",
        "claim":"bounded-input common-upstream robust two-stage action, not independent output projection",
    }


def run_indistinguishable_interface_stress():
    I=((1.,0.),(0.,1.))
    neg=((-1.,0.),(0.,1.))
    base=dict(C=I,L=(2.,2.),B=(2.,2.),desired=(.7,0.))
    worlds=[dict(P=I,**base),dict(P=neg,**base)]
    uncertain=solve_common_upstream_minimax(
        worlds=worlds,input_radii=(1.,1.),accepted_error=.2,max_nodes=400)
    each_world_independently_can_achieve=(True,True)
    repaired=solve_common_upstream_minimax(
        worlds=[worlds[0],dict(P=I,**base)],
        input_radii=(1.,1.),accepted_error=.01,max_nodes=10000)
    return {
        "protocol":"SemRepair-common-upstream-robust-hidden-actuator-v1",
        "independently_reachable_targets_per_world":each_world_independently_can_achieve,
        "same_upstream_input_across_worlds_obligatory":True,
        "uncertain_source_worlds":uncertain,
        "source_worlds_verified_to_share_correct_sign":repaired,
        "vendor_fault_is_author_constructed":True,
        "actual_robot_task_episodes":0,
    }

if __name__=="__main__":
    import json
    print(json.dumps(run_indistinguishable_interface_stress(),indent=2))
