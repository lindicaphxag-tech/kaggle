"""Dimension-independent AUTHORISED-CANDIDATE synthesis with uncertainty proofs.

Research goal: NOT merely detecting a two-axis compensating shear and
blindly replacing both interfaces. Given independent externally supplied
candidate software patches (including changing only a clipping contract),
choose the MINIMUM permitted intervention that certifiably restores the
full multidimensional requested-action domain despite bounded numerical
calibration drift. On an actual 7-torque Pusher, "clip-only" may be
strictly lower source-change cost than rewriting both erroneous maps if
a trustworthy component-level saturation authority CAP allows it.

Finite catalog exact-optimality: exhaustive enumeration (minimum catalog
cost) is exact relative to the given *approved* finite catalog; it is
NOT a globally optimal program patch synthesizer, not new polytope math,
and does NOT cryptographically authenticate the catalog signers.

Model, with vector input u in product([-R_j,R_j]):
   t=(C+ΔC) clip((P+ΔP)u, -L_i,+L_i).
The programmed matrices and independent coordinatewise coefficient drift
limits are trusted SOURCE-MODEL assumptions.

Sufficient robust certification for all u:
    B_in[i]=sum_j (abs(P_ij)+dP_ij)*R_j <= L_i
    B_out[i]=sum_j [
        abs((CP-I)_ij) +
        sum_k( abs(C_ik)*dP_kj + dC_ik*abs(P_kj) + dC_ik*dP_kj)
    ] * R_j <= eps_i

For ΔP=ΔC=0 and epsilon=0, this becomes NECESSARY and SUFFICIENT
for invertible square C and full-dimensional u-box.

This bounds actuator interface semantics, not success/safety of
nonlinear full robot trajectories; actual trusted signer roots and
field calibration origins are NOT established by this code.
"""
from __future__ import annotations

from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Mapping,Sequence


def canonical_hash(x):
    return sha256(dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _matrix(a,n,name,nonnegative=False):
    if not isinstance(a,(tuple,list)) or len(a)!=n:
        raise ValueError(f"{name}: must be finite square matrix")
    result=[]
    for row in a:
        if not isinstance(row,(tuple,list)) or len(row)!=n:
            raise ValueError(f"{name}: must be {n}x{n}")
        out=[]
        for item in row:
            if type(item) not in (int,float) or not isfinite(item) or \
               (nonnegative and item<0):
                raise ValueError(f"{name}: entries must be finite and allowed")
            out.append(float(item))
        result.append(tuple(out))
    return tuple(result)


def _vector(v,n,name,strict):
    if not isinstance(v,(tuple,list)) or len(v)!=n:
        raise ValueError(f"{name}: expected length {n}")
    ans=[]
    for z in v:
        if type(z) not in (int,float) or not isfinite(z) or \
           (strict and z<=0) or (not strict and z<0):
            raise ValueError(f"{name}: finite nonnegative bounds required")
        ans.append(float(z))
    return tuple(ans)


def certify_multichannel_robust_repair(
    *,producer,controller,action_radii,internal_clip_limits,
    producer_abs_error,controller_abs_error,output_tolerance,
    roundoff_margin=1e-11,
):
    """One-sided sound output-envelope bounds; fail closed at uncertain edge."""
    if not isinstance(action_radii,(tuple,list)) or not 2<=len(action_radii)<=32:
        raise ValueError("number of channels must be in [2,32]")
    n=len(action_radii)
    P=_matrix(producer,n,"producer")
    C=_matrix(controller,n,"controller")
    dP=_matrix(producer_abs_error,n,"producer drift",True)
    dC=_matrix(controller_abs_error,n,"controller drift",True)
    R=_vector(action_radii,n,"requested policy action radii",True)
    L=_vector(internal_clip_limits,n,"middleware clip",True)
    tol=_vector(output_tolerance,n,"output action tolerance",False)
    if type(roundoff_margin) not in (int,float) or not isfinite(roundoff_margin) or roundoff_margin<0:
        raise ValueError("invalid numerical soundness margin")
    M=float(roundoff_margin)
    CP=tuple(tuple(sum(C[i][k]*P[k][j] for k in range(n))
                   for j in range(n)) for i in range(n))
    stage=tuple(sum((abs(P[i][j])+dP[i][j])*R[j] for j in range(n))
                for i in range(n))
    uncert=tuple(tuple(abs(CP[i][j]-(1. if i==j else 0.))+
        sum(abs(C[i][k])*dP[k][j]+dC[i][k]*abs(P[k][j])+
            dC[i][k]*dP[k][j] for k in range(n))
                        for j in range(n)) for i in range(n))
    outgoing=tuple(sum(uncert[i][j]*R[j] for j in range(n))
                   for i in range(n))
    stage_ok=all(stage[i]+M<=L[i] for i in range(n))
    tolerance_ok=all(outgoing[i]+M<=tol[i] for i in range(n))
    status=("CONDITIONAL_ROBUST_VALID" if stage_ok and tolerance_ok
            else "REFUSE_ACTION_CONTRACT_UNPROVEN")
    return {
        "decision":status,"certified":bool(stage_ok and tolerance_ok),
        "dimension":n,"stage_worst_case":list(stage),
        "max_allowed_internal_clip":list(L),
        "outgoing_abs_error_upper_bound":list(outgoing),
        "allowed_action_output_error":list(tol),
        "stage_saturation_absent_over_full_box":stage_ok,
        "output_error_covered_over_full_box":tolerance_ok,
        "numeric_margin":M,
        "certificate_sha256":canonical_hash({
            "n":n,"P":P,"C":C,"dP":dP,"dC":dC,
            "R":R,"L":L,"tol":tol,"margin":M,
            "stage":stage,"outgoing":outgoing}),
        "claim":"only within fixed linear source model and asserted trusted elementwise coefficient uncertainty",
    }


def choose_authority_constrained_patch(
    *,catalog:Sequence[Mapping],action_radii,producer_abs_error,
    controller_abs_error,output_tolerance,
    licensed_component_edits:Sequence[str],
    hard_clip_caps,roundoff_margin=1e-11,
):
    """Find provably least declared COST in finite externally provided catalog.

    Every entry must include:
      name, producer, controller, clip, edits (list), cost.
    A clip increase is NOT inherently safe: the externally specified
    component's allowed *maximum* is a separate gate. Candidate selection
    is NOT signature verification or actual patch installation.
    """
    if not catalog or not isinstance(catalog,(tuple,list)):
        raise ValueError("nonempty finite candidate patch catalog required")
    n=len(action_radii)
    cap=_vector(hard_clip_caps,n,"external component absolute clip caps",True)
    if not isinstance(licensed_component_edits,(tuple,list)) or \
       not all(x in ("producer","controller","clip") for x in licensed_component_edits):
        raise ValueError("unknown licensed source modification")
    allowed=set(licensed_component_edits)
    reports=[]
    names=set()
    for item in catalog:
        if not isinstance(item,Mapping) or set(item)!={"name","producer","controller","clip","edits","cost"}:
            raise ValueError("strict candidate patch schema")
        label=item["name"]
        if not isinstance(label,str) or not label or label in names:
            raise ValueError("duplicate or empty candidate name")
        names.add(label)
        edits=item["edits"]
        if not isinstance(edits,(tuple,list)) or len(set(edits))!=len(edits) or \
           any(x not in ("producer","controller","clip") for x in edits):
            raise ValueError("invalid source change scope")
        cost=item["cost"]
        if type(cost) not in (int,float) or not isfinite(cost) or cost<0:
            raise ValueError("nonnegative externally declared patch cost required")
        L=_vector(item["clip"],n,"candidate clip",True)
        inside_caps=all(L[i]<=cap[i] for i in range(n))
        licensed=set(edits)<=allowed
        proof=certify_multichannel_robust_repair(
            producer=item["producer"],controller=item["controller"],
            action_radii=action_radii,internal_clip_limits=L,
            producer_abs_error=producer_abs_error,
            controller_abs_error=controller_abs_error,
            output_tolerance=output_tolerance,
            roundoff_margin=roundoff_margin)
        status=("CERTIFIED_CATALOG_CANDIDATE" if proof["certified"] and
                inside_caps and licensed else
                "REFUSE_LICENSE_OR_CLIP_CAP" if not inside_caps or not licensed
                else "REFUSE_ROBUST_SEMANTICS")
        reports.append({"name":label,"cost":float(cost),"edits":list(edits),
                        "licensed":licensed,"inside_hard_caps":inside_caps,
                        "decision":status,"proof":proof})
    viable=[r for r in reports if r["decision"]=="CERTIFIED_CATALOG_CANDIDATE"]
    if not viable:
        return {"decision":"REFUSE_NO_AUTHORIZED_CERTIFIED_CANDIDATE",
                "selected":None,"candidate_evaluations":reports,
                "catalog_optimality_proven":True,"live_patch_installed":False,
                "trust_root_signatures_verified_here":False}
    best=min(viable,key=lambda x:(x["cost"],x["name"]))
    return {
        "decision":"CANDIDATE_ONLY_NEEDS_REAL_SIGNED_SOURCE_ACTIVATION",
        "selected":best["name"],"minimum_declared_catalog_cost":best["cost"],
        "candidate_evaluations":reports,
        "catalog_optimality_proven":True,
        "live_patch_installed":False,
        "trust_root_signatures_verified_here":False,
        "optimality_scope":"finite licensed catalog only; NOT all possible source programs",
        "selected_proof_sha256":best["proof"]["certificate_sha256"],
    }


def _identity(n):
    return [[float(i==j) for j in range(n)] for i in range(n)]


def _matmul(a,b):
    n=len(a)
    return [[sum(a[i][k]*b[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)]


def example_native_seven_joint_catalog():
    n=7
    base=_identity(n)
    broken=_identity(n)
    broken[0][1]=0.8
    compensating=_identity(n)
    compensating[0][1]=-0.8
    C=compensating
    P=broken
    old=[2.]*n
    enough=[3.65]+[2.05]*(n-1)
    # Set signed author edit budget for full atomic rewrite > clip-only edit.
    catalog=[
      {"name":"A-no-change","producer":P,"controller":C,
       "clip":old,"edits":[],"cost":0},
      {"name":"B-controller-only","producer":P,"controller":base,
       "clip":old,"edits":["controller"],"cost":2},
      {"name":"C-producer-only","producer":base,"controller":C,
       "clip":old,"edits":["producer"],"cost":2},
      {"name":"D-atomic-matrices","producer":base,"controller":base,
       "clip":enough,"edits":["producer","controller","clip"],"cost":5},
      {"name":"E-clip-only","producer":P,"controller":C,
       "clip":enough,"edits":["clip"],"cost":1},
    ]
    d=[[0.]*n for _ in range(n)]
    # This is deterministic proof of a modelled 7-channel source package,
    # not an independent real 7-source software publisher!
    return {
       "premise":"author-constructed seven-source-action contract, 7D physical action scaling from published Pusher",
       "two_stage_fault_is_modelled":True,
       "producer_controller_compose_to_identity":_matmul(C,P)==base,
       "if_outer_hardware_does_not_allow_clip_increase_then_refuse_clip_only":True,
       "licensed_all_costs":choose_authority_constrained_patch(
          catalog=catalog,action_radii=[2.]*n,
          producer_abs_error=d,controller_abs_error=d,
          output_tolerance=[1e-9]*n,
          hard_clip_caps=[3.7]+[2.2]*(n-1),
          licensed_component_edits=["producer","controller","clip"]),
       "clip_change_not_licensed":choose_authority_constrained_patch(
          catalog=catalog,action_radii=[2.]*n,
          producer_abs_error=d,controller_abs_error=d,
          output_tolerance=[1e-9]*n,
          hard_clip_caps=[3.7]+[2.2]*(n-1),
          licensed_component_edits=["producer","controller"]),
       "strict_motor_clip_caps":choose_authority_constrained_patch(
          catalog=catalog,action_radii=[2.]*n,
          producer_abs_error=d,controller_abs_error=d,
          output_tolerance=[1e-9]*n,
          hard_clip_caps=[2.]*n,
          licensed_component_edits=["producer","controller","clip"]),
    }


if __name__=="__main__":
    print(dumps(example_native_seven_joint_catalog(),indent=2,sort_keys=True))
