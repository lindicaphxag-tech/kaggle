"""A repair can change the physically REACHABLE action set, not just its label.

Assume motor t = C clip(P u, -L, L), with invertible P,C and an independent
source implementation which can replace C with identity (and P with identity).
Even an omniscient black-box actor with UNBOUNDED upstream u cannot produce
a desired torque t outside C[-L,L]^2. It follows that perfect black-box
precompensation is impossible at those targets, whereas a source repair may
restore t if |t|_infty<=L.

The exact optimal black-box output for a Euclidean torque tracking loss is the
projection of t onto the four edges of a 2D parallelogram C[-L,L]^2.
No Monte Carlo or weak black-box heuristic is used.

Tests step an actual articulated two-hinge MuJoCo motor from the SAME initial
state under each candidate torque, comparing final joint velocities.
This is a controlled actuator mechanism and NOT learned-policy success.
"""
from __future__ import annotations
from math import dist, isfinite
from hashlib import sha256
import itertools
import json

import mujoco

from .observable_interface_factorization import I, inverse, mv, Mat, Vec
from .two_link_cross_controller_benchmark import make_model, INIT

LIMIT=9.
PRODUCER:Mat=((1.,.8),(0.,1.))
CONTROLLER:Mat=((1.,-.8),(0.,1.))
TORQUES=(-8.,-6.,-3.,0.,3.,6.,8.)
EPS=1e-9


def _point_to_edge(x:Vec,a:Vec,b:Vec)->Vec:
    dx,dy=b[0]-a[0],b[1]-a[1]
    denom=dx*dx+dy*dy
    if denom<=0:
        raise ValueError("degenerate actuator reachable edge")
    t=max(0.,min(1.,((x[0]-a[0])*dx+(x[1]-a[1])*dy)/denom))
    return (a[0]+t*dx,a[1]+t*dy)


def exact_blackbox_torque_oracle(target:Vec, *, controller:Mat=CONTROLLER,
                                 limit:float=LIMIT) -> dict:
    """True best possible black-box actuator output given C and clipping.

    Solver is clairvoyant, has no observation cost, and upstream input can be
    arbitrary, so it upper-bounds *any* realistic black-box precompensator.
    """
    if len(target)!=2 or any(not isfinite(v) for v in target):
        raise ValueError("invalid desired torque")
    if not isfinite(limit) or limit<=0:
        raise ValueError("invalid internal saturation")
    recovered=mv(inverse(controller),target)
    if max(abs(z) for z in recovered)<=limit+EPS:
        return {"feasible":True,"closest_actuator_torque":target,
                "min_l2_torque_error":0.,
                "unconstrained_required_inner_torque":recovered}
    corners=[mv(controller,corner) for corner in
             ((-limit,-limit),(limit,-limit),(limit,limit),(-limit,limit))]
    closest=[_point_to_edge(target,corners[i],corners[(i+1)%4])
             for i in range(4)]
    optimum=min(closest,key=lambda output:dist(output,target))
    return {"feasible":False,"closest_actuator_torque":optimum,
            "min_l2_torque_error":dist(optimum,target),
            "unconstrained_required_inner_torque":recovered}


def _mujoco_step(model,torque:Vec)->Vec:
    data=mujoco.MjData(model)
    data.qpos[:2]=INIT
    mujoco.mj_forward(model,data)
    data.ctrl[:]=torque
    mujoco.mj_step(model,data)
    if max(abs(float(data.qfrc_actuator[i])-float(torque[i]))
           for i in range(2)) > 1e-8:
        raise AssertionError("requested actuator output differs from applied MuJoCo motor force")
    return (float(data.qvel[0]),float(data.qvel[1]))


def run_reachability_assay() -> dict:
    # The producer/controller maps exactly compensate BEFORE clipping.
    from .observable_interface_factorization import mmul
    composite=mmul(CONTROLLER,PRODUCER)
    if any(abs(composite[i][j]-I[i][j])>1e-12
           for i in range(2) for j in range(2)):
        raise AssertionError("not an exact low-amplitude compensating pair")
    model=make_model()
    rows=[]
    for command in itertools.product(TORQUES,repeat=2):
        desired=(float(command[0]),float(command[1]))
        oracle=exact_blackbox_torque_oracle(desired)
        corrected=desired       # after source patches P=C=I, within [-L,L]
        ref_vel=_mujoco_step(model,desired)
        source_vel=_mujoco_step(model,corrected)
        blackbox_vel=_mujoco_step(model,oracle["closest_actuator_torque"])
        if dist(ref_vel,source_vel)>EPS:
            raise AssertionError("valid source repair did not reproduce reference")
        if oracle["feasible"] and dist(blackbox_vel,ref_vel)>EPS:
            raise AssertionError("blackbox oracle must match when reachable")
        rows.append({
            "requested_torque":desired,"blackbox_reachable":oracle["feasible"],
            "oracle_min_torque_error":oracle["min_l2_torque_error"],
            "blackbox_oracle_torque":oracle["closest_actuator_torque"],
            "source_torque":corrected,
            "blackbox_velocity_gap":dist(blackbox_vel,ref_vel),
            "source_velocity_gap":dist(source_vel,ref_vel),
            "blackbox_is_omniscient_and_unbounded_input":True})
    unreachable=[r for r in rows if not r["blackbox_reachable"]]
    if not unreachable:
        raise AssertionError("selected set fails to exercise internal saturation obstruction")
    assert all(r["oracle_min_torque_error"]>EPS and
               r["blackbox_velocity_gap"]>EPS for r in unreachable)
    return {
        "protocol":"semrepair-native-two-link-internal-clipping-reachable-set-v1",
        "model":"2-link native MuJoCo hinge arm; exact full-information action oracle",
        "source_interface_pair":"P shear .8, C inverse shear -.8",
        "source_limited_producer_torque":LIMIT,
        "test_torque_grid":list(TORQUES),
        "grid_conditions_not_independent":len(rows),
        "unreachable_target_count_even_for_omniscient_blackbox":len(unreachable),
        "reachable_target_count":len(rows)-len(unreachable),
        "max_omniscient_blackbox_torque_error":max(r["oracle_min_torque_error"] for r in rows),
        "max_blackbox_first_step_joint_velocity_gap":max(r["blackbox_velocity_gap"] for r in rows),
        "max_source_repair_joint_velocity_gap":max(r["source_velocity_gap"] for r in rows),
        "source_patch_was_authenticated":False,
        "source_patch_was_installed_as_software":False,
        "task_level_success_measured":False,
        "source_model_families":1,"learned_policy_count":0,
        "input_sha256":sha256(json.dumps({
           "producer":PRODUCER,"controller":CONTROLLER,
           "limit":LIMIT,"grid":TORQUES},sort_keys=True).encode()).hexdigest(),
        "rows":rows,
        "limitation":"analytic exact 2D clipped-actuation parallelogram and single native MuJoCo response; "
                     "not a multi-step robot control outcome, unknown fault learning or verified source authority",
    }


if __name__=="__main__":
    print(json.dumps(run_reachability_assay(),indent=2,sort_keys=True))
