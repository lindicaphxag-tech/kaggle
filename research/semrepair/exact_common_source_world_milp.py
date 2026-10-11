"""Global mixed-integer source-world actuator recovery with ONE common upstream u.

For each indistinguishable version w: F_w(u)=clip(C_w clip(P_w u,L_w),B_w).
Solve min_{u in true upstream box} max_w ||F_w(u)-desired_w||_infinity.
Both clip layers are encoded EXACTLY in real-arithmetic disjunctive mixed-
integer linear constraints. A primal control is physically replayed through
every original authored source map, and the MILP DUAL bound is a global
lower bound on achievable tracking error, subject to numerical tolerances.

SAT / UNSAT / UNKNOWN are distinct. Do NOT treat the MILP gap as a
certified hardware-level proof: HiGHS is floating point, coefficient
models, hidden worlds, desired commands and clip caps are first-party
assumptions without external cryptographic or actuator calibration proof.
This is a 2D finite-world method, not global task-return optimization or
novel MILP mathematics. A real MIP may time out; in that case abstain.
"""
from __future__ import annotations
from math import isfinite
import numpy as np
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix


def _v2(v,positive=False):
    if not isinstance(v,(list,tuple)) or len(v)!=2 or any(
        type(x) not in (int,float) or not isfinite(x) or
        (positive and x<=0) for x in v):
        raise ValueError("finite two-vector required")
    return tuple(float(x) for x in v)


def _mat(m):
    if not isinstance(m,(list,tuple)) or len(m)!=2:
        raise ValueError("two-by-two source matrix required")
    return (_v2(m[0]),_v2(m[1]))


def _mat_range(P,bounds):
    return tuple((
        sum(min(P[i][j]*bounds[j][0],P[i][j]*bounds[j][1]) for j in range(2)),
        sum(max(P[i][j]*bounds[j][0],P[i][j]*bounds[j][1]) for j in range(2))
    ) for i in range(2))


def actual_physical_command(w,u):
    t=[sum(w["P"][i][j]*u[j] for j in range(2)) for i in range(2)]
    z=[max(-w["L"][i],min(w["L"][i],t[i])) for i in range(2)]
    c=[sum(w["C"][i][j]*z[j] for j in range(2)) for i in range(2)]
    return [max(-w["B"][i],min(w["B"][i],c[i])) for i in range(2)]


class _LinearMIP:
    def __init__(self):
        self.bounds=[];self.integrality=[];self.rows=[]
    def var(self,lower,upper,binary=False):
        if not (isfinite(lower) and isfinite(upper) and upper>=lower):
            raise ValueError("invalid finite source MILP coordinate bounds")
        k=len(self.bounds)
        self.bounds.append((float(lower),float(upper)))
        self.integrality.append(int(binary))
        return k
    def row(self,d,lo=-np.inf,hi=np.inf):
        self.rows.append((dict(d),float(lo),float(hi)))
    def eq(self,d,target=0.):self.row(d,target,target)
    def le(self,d,limit):self.row(d,hi=limit)
    def if_binary(self,selector,d,rhs):
        # Tight valid big-M from the ACTUAL variable interval bounds.
        mx=sum(a*(self.bounds[k][1] if a>=0 else self.bounds[k][0])
               for k,a in d.items())
        M=max(0.,mx-rhs)
        c=dict(d);c[selector]=c.get(selector,0.)+M
        self.le(c,rhs+M)
    def clamp(self,t,y,L):
        low,mid,high=[self.var(0,1,True) for _ in range(3)]
        self.eq({low:1.,mid:1.,high:1.},1.)
        for b,conditions in (
            (low,[({t:1.},-L),({y:1.},-L),({y:-1.},L)]),
            (mid,[({t:1.},L),({t:-1.},L),
                  ({y:1.,t:-1.},0.),({t:1.,y:-1.},0.)]),
            (high,[({t:-1.},-L),({y:1.},L),({y:-1.},-L)]),
        ):
            for coeff,target in conditions:self.if_binary(b,coeff,target)
    def optimize(self,obj,max_seconds,gap):
        I=[];J=[];V=[]
        for i,(d,_,_) in enumerate(self.rows):
            for j,v in d.items():I.append(i);J.append(j);V.append(v)
        A=coo_matrix((V,(I,J)),
                     shape=(len(self.rows),len(self.bounds))).tocsr()
        c=np.zeros(len(self.bounds));c[obj]=1.
        lb=np.array([v[0] for v in self.bounds])
        ub=np.array([v[1] for v in self.bounds])
        r=milp(c,integrality=np.array(self.integrality),
               bounds=Bounds(lb,ub),
               constraints=[LinearConstraint(
                   A,[row[1] for row in self.rows],
                   [row[2] for row in self.rows])],
               options={"time_limit":max_seconds,"mip_rel_gap":gap,
                        "presolve":True})
        return r,len(self.bounds),len(self.rows),sum(self.integrality)


def solve_common_upstream_milp(
    *,worlds,input_radii=(1.,1.),tolerance=.01,
    seconds=20,numeric_margin=1e-6,solver_gap=1e-9
):
    if not isinstance(worlds,(tuple,list)) or not 1<=len(worlds)<=8:
        raise ValueError("one through eight compatible source worlds required")
    if type(seconds) not in (int,float) or not 0<seconds<=120:
        raise ValueError("time budget in (0,120]")
    if type(tolerance) not in (int,float) or not isfinite(tolerance) or tolerance<0:
        raise ValueError("nonnegative source error tolerance required")
    if type(numeric_margin) not in (int,float) or not 0<numeric_margin<.01:
        raise ValueError("strict positive uncertainty margin required")
    if type(solver_gap) not in (int,float) or not 0<=solver_gap<=.1:
        raise ValueError("solver gap must be finite bounded")
    R=_v2(input_radii,True)
    ws=[]
    for item in worlds:
        if not isinstance(item,dict) or set(item)!={"P","C","L","B","desired"}:
            raise ValueError("full source world identity required")
        ws.append(dict(P=_mat(item["P"]),C=_mat(item["C"]),
                       L=_v2(item["L"],True),
                       B=_v2(item["B"],True),
                       desired=_v2(item["desired"])))
    M=_LinearMIP()
    u=[M.var(-r,r) for r in R]
    maximum=max(max(abs(v) for v in w["desired"])+max(w["B"])
                for w in ws)+1.
    rho=M.var(0.,maximum)
    for w in ws:
        pr=_mat_range(w["P"],[(-r,r) for r in R])
        t=[M.var(*interval) for interval in pr]
        z=[M.var(-limit,limit) for limit in w["L"]]
        cr=_mat_range(w["C"],[(-v,v) for v in w["L"]])
        c=[M.var(*interval) for interval in cr]
        y=[M.var(-limit,limit) for limit in w["B"]]
        for i in range(2):
            M.eq({t[i]:1.,u[0]:-w["P"][i][0],
                  u[1]:-w["P"][i][1]})
            M.clamp(t[i],z[i],w["L"][i])
            M.eq({c[i]:1.,z[0]:-w["C"][i][0],
                  z[1]:-w["C"][i][1]})
            M.clamp(c[i],y[i],w["B"][i])
            M.le({y[i]:1.,rho:-1.},w["desired"][i])
            M.le({y[i]:-1.,rho:-1.},-w["desired"][i])
    result,nvar,nrow,nbin=M.optimize(rho,seconds,solver_gap)
    witness=(float(result.x[u[0]]),float(result.x[u[1]])) \
            if result.x is not None else None
    physical=[actual_physical_command(w,witness) for w in ws] \
             if witness is not None else None
    actual=max(abs(physical[k][i]-w["desired"][i])
               for k,w in enumerate(ws) for i in range(2)) \
               if witness is not None else None
    lower=(float(result.mip_dual_bound)
           if getattr(result,"mip_dual_bound",None) is not None else None)
    upper=float(result.fun) if result.fun is not None else None
    if actual is not None and actual+numeric_margin<=tolerance:
        verdict="SAT_REPLAYED_COMMON_UPSTREAM_INPUT"
    elif lower is not None and lower-numeric_margin>tolerance:
        verdict="UNSAT_GLOBAL_DUAL_BOUND"
    else:
        verdict="UNKNOWN_NO_AUTHORIZATION"
    return {
        "decision":verdict,"number_of_source_worlds":len(ws),
        "actual_shared_upstream_control":list(witness) if witness else None,
        "replayed_per_world_physical_commands":physical,
        "replayed_true_max_error":actual,
        "global_primal_objective_upper":upper,
        "global_mixed_integer_dual_lower":lower,
        "global_MIP_gap":upper-lower
            if upper is not None and lower is not None else None,
        "expanded_source_CLIP_binary_variables":nbin,
        "MILP_total_variables":nvar,"MILP_constraints":nrow,
        "solver_status":int(result.status),"solver_message":str(result.message),
        "numerical_margin":numeric_margin,
        "explicit_UPSTREAM_control_box":list(R),
        "all_source_models_and_world_set_assumed_exact":True,
        "floating_MILP_not_outward_rounded_formal_software_proof":True,
        "real_robot_rollouts":0,
        "proof_scope":"two-layer piecewise-linear clip, bounded shared upstream control, up to 8 finite source worlds; NOT cumulative robot return",
    }


def two_source_identity_regression():
    I=((1.,0.),(0.,1.));reverse=((-1.,0.),(0.,1.))
    world=lambda P:dict(P=P,C=I,L=(2.,2.),B=(2.,2.),
                        desired=(.7,0.))
    return {
        "two_individually_repairable_but_indistinguishable_versions":
            solve_common_upstream_milp(
                worlds=[world(I),world(reverse)],tolerance=.2),
        "version_identity_known":
            solve_common_upstream_milp(worlds=[world(I)],tolerance=.01),
    }


if __name__=="__main__":
    import json
    print(json.dumps(two_source_identity_regression(),indent=2,sort_keys=True))
