"""Hidden-label-free reconstruction of two linear action ABI interfaces.

Given requested u, a tap after the producer p=P*u and a tap at the controller
t=C*p, recover P and C independently using two linearly independent probes.
This is textbook matrix identification, not new algebra or source attestation.
A third held-out probe checks linear-model generalization and rejects bad data.

Crucially, exterior output alone identifies CP, not P and C individually.
The evaluator owns true interfaces but only passes tap NUMBERS to the solver.
A verified patch origin is still required before live software installation.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

Vec=tuple[float,float]
Mat=tuple[Vec,Vec]
I:Mat=((1.,0.),(0.,1.))
PROBE_E1:Vec=(0.67,0.)
PROBE_E2:Vec=(0.,-0.53)
CHECK:Vec=(0.31,-0.27)


@dataclass(frozen=True)
class TapSample:
    requested: Vec
    producer: Vec
    controller: Vec


def mmul(a:Mat,b:Mat)->Mat:
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))


def mv(a:Mat,v:Vec)->Vec:
    return (a[0][0]*v[0]+a[0][1]*v[1],a[1][0]*v[0]+a[1][1]*v[1])


def inverse(a:Mat,rank_tolerance:float=1e-10)->Mat:
    det=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    if abs(det)<=rank_tolerance:
        raise ValueError("insufficiently independent probes or noninvertible observed producer map")
    return ((a[1][1]/det,-a[0][1]/det),(-a[1][0]/det,a[0][0]/det))


def columns(v1:Vec,v2:Vec)->Mat:
    return ((v1[0],v2[0]),(v1[1],v2[1]))


def max_diff(a:Vec,b:Vec)->float:
    return max(abs(a[i]-b[i]) for i in range(2))


def is_identity(a:Mat, tol:float)->bool:
    return all(abs(a[i][j]-I[i][j])<=tol for i in range(2) for j in range(2))


def identify_repair_from_numeric_taps(train: Sequence[TapSample],
                                      check: TapSample, *,
                                      error_tolerance:float=1e-7)->dict:
    """Return only a *candidate* repair bundle, not a legal patch certificate."""
    if not isfinite(error_tolerance) or error_tolerance<=0:
        raise ValueError("finite positive error tolerance required")
    if len(train)!=2:
        raise ValueError("two independent probe samples required")
    for row in tuple(train)+(check,):
        if any(not isfinite(v) for v in row.requested+row.producer+row.controller):
            raise ValueError("nonfinite tap")
    u=columns(train[0].requested,train[1].requested)
    p=columns(train[0].producer,train[1].producer)
    t=columns(train[0].controller,train[1].controller)
    try:
        producer=mmul(p,inverse(u))
        controller=mmul(t,inverse(p))
    except ValueError:
        return {"decision":"REFUSE_RANK_DEFICIENT","candidate_patch":None}
    predicted_p=mv(producer,check.requested)
    predicted_t=mv(controller,predicted_p)
    if max_diff(predicted_p,check.producer)>error_tolerance or \
            max_diff(predicted_t,check.controller)>error_tolerance:
        return {"decision":"REFUSE_NONLINEAR_OR_INCONSISTENT",
                "candidate_patch":None,
                "heldout_producer_error":max_diff(predicted_p,check.producer),
                "heldout_controller_error":max_diff(predicted_t,check.controller)}
    if is_identity(producer,error_tolerance) and is_identity(controller,error_tolerance):
        patch="noop"
    elif is_identity(producer,error_tolerance):
        patch="controller-only"
    elif is_identity(controller,error_tolerance):
        patch="producer-only"
    else:
        patch="atomic"
    return {
        "decision":"CANDIDATE_ONLY_NEEDS_TRUSTED_SOURCE_ATTESTATION",
        "candidate_patch":patch,
        "producer_matrix":producer,"controller_matrix":controller,
        "effective_matrix":mmul(controller,producer),
        "heldout_producer_error":max_diff(predicted_p,check.producer),
        "heldout_controller_error":max_diff(predicted_t,check.controller),
        "limited_claim":"finite two-axis linear ABI identification with honest numeric taps, not physical safety or authorized source code",
    }


def nominal_probe(world_producer:Mat, world_controller:Mat, request:Vec)->TapSample:
    # Called by the evaluator. The solver receives this numeric sample only,
    # not the hidden two matrices or symbolic world ID.
    observed_producer=mv(world_producer,request)
    observed_controller=mv(world_controller,observed_producer)
    return TapSample(request,observed_producer,observed_controller)


def blind_closed_loop_planar_assay()->dict:
    """Actual MuJoCo task executions after hidden-label-free ABI factorization."""
    from hashlib import sha256
    import json
    from .native_planar_task_repair import (
        actual_mujoco_model, simulate, frozen_targets, CASES,
        IDENTITY, SWAP, NEGATE, ROT_POS, ROT_NEG,
    )
    # The 0.25 shear pair is deliberately outside the original signed-
    # permutation catalog while remaining an invertible linear ABI.
    SHEAR:Mat=((1.,.25),(0.,1.))
    SHEAR_INV:Mat=((1.,-.25),(0.,1.))
    scenarios=(("reference",I,I,"noop"),)+tuple(
        (name,tuple(tuple(float(v) for v in row) for row in prod),
              tuple(tuple(float(v) for v in row) for row in ctrl),patch)
        for name,prod,ctrl,patch in CASES)+(
            ("unseen-invertible-shear",SHEAR,SHEAR_INV,"atomic"),)
    model=actual_mujoco_model()
    goals=frozen_targets()
    scores={}
    rows=[]
    for name,p,c,oracle_patch in scenarios:
        tapped=[nominal_probe(p,c,u) for u in (PROBE_E1,PROBE_E2)]
        hold=nominal_probe(p,c,CHECK)
        inferred=identify_repair_from_numeric_taps(tapped,hold)
        candidate=inferred.get("candidate_patch")
        if candidate!=oracle_patch:
            raise AssertionError("hidden-label-free controller interface inference failed")
        label="no-implementation-labels-in-solver"
        # Explicitly separate proposed patch structure from *legal authorization*.
        solved=0
        baseline=0
        for j,target in enumerate(goals):
            predicted=simulate(model,target,p,c,candidate)
            untouched=simulate(model,target,p,c,"untouched")
            ref=simulate(model,target,I,I,"untouched")
            solved+=int(predicted["success"])
            baseline+=int(untouched["success"])
            rows.append({"case":name,"target_index":j,"blind_candidate_patch":candidate,
                         "oracle_patch_for_scoring_only":oracle_patch,
                         "blind_distance":predicted["distance"],
                         "baseline_distance":untouched["distance"],
                         "reference_distance":ref["distance"],
                         "blind_success":predicted["success"],
                         "untouched_success":untouched["success"],
                         "source_attested":False})
        scores[name]={"blind_candidate_success":solved,"untouched_success":baseline,
                      "trials":len(goals),"candidate":candidate,"heldout_tap_error":max(
                          inferred["heldout_producer_error"],inferred["heldout_controller_error"])}
    if not all(x["blind_candidate_success"]==16 for x in scores.values()):
        raise AssertionError("blind linear ABI diagnosis fails frozen tasks")
    return {"protocol":"semrepair-blind-two-tap-factorization-native-mujoco-v1",
            "task_controller":"frozen hand-coded PD, NOT pretrained VLA",
            "target_count":len(goals),"source_case_count":len(scenarios),
            "independent_robot_families":1,
            "unseen_mapping_family":"invertible shear composed with inverse",
            "probe_menu":[list(PROBE_E1),list(PROBE_E2),list(CHECK)],
            "input_oracle_labels_visible_to_solver":False,
            "source_attestation_implemented":False,
            "patch_installation_performed":False,
            "results":scores,"rows":rows,
            "claim_scope":"honest internal numeric interface taps and exact linear source model, one simulated controller/plant family; no real robot, no trusted source, no VLA"}
