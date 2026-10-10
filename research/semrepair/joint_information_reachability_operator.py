"""Joint observable-world and actuator-reachability minimax repair certificate.

For each public observation h, a blackbox must select the SAME physical
action z for all possible hidden source/controller worlds x in its
information fiber H^{-1}(h), subject to z in K(h), where K(h) is a compact
finite UNION of convex actuator polytopes (e.g. full two-clip Pusher action
reachability region).

Its tight worst-case L_infinity tracking error is

  rho*(h) = min_{z in K(h)} max_{x in H^-1(h)} ||z - desired(x)||_infinity.

The algorithm computes rho* by bisection of the EXACT monotonic predicate:
a convex reachability polygon intersects the intersection of all
L_infinity balls of radius r around the target commands. The latter is
an axis-aligned rectangle with bounds
 max_x(target[x,j]-r) <= z[j] <= min_x(target[x,j]+r).
All distinct polygons in the entire physically reachable union are
checked without convex-hulling across gaps. This is a numerical exact
feasibility oracle to requested tolerance, NOT closed-form new convex
optimization theory, and NOT global future task reward.

This UNIFIES two previously separate recovery failure mechanisms:
  (i) ACTUATION obstruction: a fixed fully observed ideal action lies
      outside current source action image K.
  (ii) INFORMATION obstruction: distinct latent source histories require
       different ideal actions but are indistinguishable to current-only
       blackbox observations, even if every target action is reachable.

True source authority may enlarge K. Trustworthy historical anchor
evidence may refine H into singleton fibers. Both interventions can be
needed; neither alone is assumed to fix all worlds.

The certificate is conditional on an honest world set and exact reachable
source polytopes. It does not attest observations, decide code owner keys,
solve nonlinear dynamics, or infer hidden worlds from raw images.
"""
from __future__ import annotations

from math import isfinite
from typing import Sequence

from .strong_clipped_actuator_oracle import (
    clip_polygon,output_reachability_polygons,hull
)


def _validate_targets(targets,polygons):
    if not isinstance(targets,(tuple,list)) or not targets:
        raise ValueError("nonempty source world target set required")
    if not isinstance(polygons,(tuple,list)) or not polygons:
        raise ValueError("finite nonempty reachable polygon union required")
    target=[]
    for a in targets:
        if not isinstance(a,(tuple,list)) or len(a)!=2 or not all(
           type(z) in (int,float) and isfinite(z) for z in a):
            raise ValueError("invalid hidden-world target action")
        target.append((float(a[0]),float(a[1])))
    poly=[]
    for polygon in polygons:
        if not isinstance(polygon,(tuple,list)) or not polygon:
            raise ValueError("empty/unavailable actuator region")
        xs=[]
        for a in polygon:
            if not isinstance(a,(tuple,list)) or len(a)!=2 or not all(
                type(z) in (int,float) and isfinite(z) for z in a):
                raise ValueError("invalid physical actuator vertex")
            xs.append((float(a[0]),float(a[1])))
        poly.append(xs)
    return target,poly


def _intersection_witness(targets,polygons,radius):
    low=[max(w[j]-radius for w in targets) for j in range(2)]
    high=[min(w[j]+radius for w in targets) for j in range(2)]
    if any(low[j]>high[j]+1e-11 for j in range(2)):
        return None
    for initial in polygons:
        poly=initial
        for axis in range(2):
            poly=clip_polygon(poly,axis,-1,low[axis])
            if not poly:break
            poly=clip_polygon(poly,axis,1,high[axis])
            if not poly:break
        if poly:
            # Selecting the average of a convex clipped region is guaranteed
            # feasible in that SAME convex polygon (do not average across
            # separate physical reachable components).
            witness=(sum(v[0] for v in poly)/len(poly),
                     sum(v[1] for v in poly)/len(poly))
            violation=max(abs(witness[j]-w[j]) for w in targets for j in range(2))
            if violation<=radius+5e-9:
                return witness
    return None


def robust_blackbox_action_error_floor(*,indistinguishable_desired_actions,
                                       reachable_actuator_polygons,
                                       requested_error_tolerance:float|None=None,
                                       absolute_search_accuracy:float=1e-8)->dict:
    targets,polygons=_validate_targets(
        indistinguishable_desired_actions,reachable_actuator_polygons)
    if not (isinstance(absolute_search_accuracy,(float,int)) and
            isfinite(absolute_search_accuracy) and
            0<absolute_search_accuracy<=.01):
        raise ValueError("invalid numerical search accuracy")
    if requested_error_tolerance is not None and \
       (type(requested_error_tolerance) not in (int,float) or
        not isfinite(requested_error_tolerance) or
        requested_error_tolerance<0):
        raise ValueError("invalid claimed error tolerance")
    info_floor=max(
        (max(a[j] for a in targets)-min(a[j] for a in targets))/2.
        for j in range(2))
    lo=info_floor
    # Every vertex is reachable, so any vertex provides a valid upper bound.
    first=polygons[0][0]
    hi=max(abs(first[j]-w[j]) for w in targets for j in range(2))
    if hi<lo:hi=lo
    if _intersection_witness(targets,polygons,hi+1e-10) is None:
        raise AssertionError("supplied physical polygons are inconsistent")
    if _intersection_witness(targets,polygons,lo) is not None:
        hi=lo
    else:
        for _ in range(70):
            if hi-lo<=absolute_search_accuracy:break
            mid=(hi+lo)/2
            if _intersection_witness(targets,polygons,mid) is not None:
                hi=mid
            else:
                lo=mid
    witness=_intersection_witness(targets,polygons,hi+5e-10)
    if witness is None:
        raise AssertionError("failed to recover actual reachable worst-world minimax witness")
    worst=max(abs(witness[j]-w[j]) for w in targets for j in range(2))
    if worst>hi+5e-9:raise AssertionError("claimed actuator minimax witness is invalid")

    meets=(requested_error_tolerance is None or
           hi<=requested_error_tolerance+absolute_search_accuracy)
    return {
        "decision":"CONDITIONALLY_ACTION_COMPENSABLE"
                   if meets else "REFUSE_UNREACHABLE_OR_UNIDENTIFIABLE_COMMAND",
        "world_count":len(targets),
        "physical_region_components":len(polygons),
        "source_world_observation_equivalence_assumed":True,
        "unrestricted_blackbox_control_has_same_observation_in_each_world":True,
        "tight_numerical_worst_case_minimax_error_lower":lo,
        "tight_numerical_worst_case_minimax_error_upper":hi,
        "ambiguity_information_only_lower_bound":info_floor,
        "reachable_physical_action_witness":list(witness),
        "witness_actual_worst_world_linf_error":worst,
        "search_accuracy":absolute_search_accuracy,
        "requested_tolerance":requested_error_tolerance,
        "can_meet_requested_tolerance":bool(meets),
        "proof_scope":"fixed two-dimensional actuator image union, finite indistinguishable source-world targets and L_infinity action error",
        "limitations":"numeric bisection; trusted world set and action reachable union; NOT hardware task optimality or sensor authentication",
    }


def four_way_orthogonal_source_and_history_repair():
    """Construct an ACTUAL two-stage physical source and indistinguishable history case.

    The two target actions share current-only observation but differ due to
    an unavailable source chunk anchor. Both demand vertical actuator torque
    1.5, which original internal L=.8 prevents for ANY source command.
    Pusher action box [-2,2]. Independently:
      - grant source clip permission: physical K becomes full 2D box, but
        hidden-world ambiguity remains 0.2;
      - disclose verified anchor: target is unique, yet original K still
        cannot deliver vertical 1.5 (>=0.7 mismatch);
      - permit BOTH: each now attainable exactly.
    """
    C=((1.,-3.5),(0.,1.))
    compromised=output_reachability_polygons(C,inner_bound=.8,outer_bound=2.)
    # Without source saturation, for external desired within [-2,2] all
    # requested actions are attainable through invertible P and C.
    restored=output_reachability_polygons(C,inner_bound=9.05,outer_bound=2.)
    desired=((1.3,1.5),(1.7,1.5))
    options={
      "neither_repaired":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=desired,
          reachable_actuator_polygons=compromised),
      "source_clip_only":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=desired,
          reachable_actuator_polygons=restored),
      "historical_anchor_only_A":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=[desired[0]],
          reachable_actuator_polygons=compromised),
      "historical_anchor_only_B":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=[desired[1]],
          reachable_actuator_polygons=compromised),
      "both_source_and_history_A":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=[desired[0]],
          reachable_actuator_polygons=restored),
      "both_source_and_history_B":robust_blackbox_action_error_floor(
          indistinguishable_desired_actions=[desired[1]],
          reachable_actuator_polygons=restored),
    }
    return {
        "protocol":"mixed-source-saturation-and-history-observation-2d-conditional-counterexample-v1",
        "faults_both_author_constructed":True,
        "real_LeRobot_callsite_or_Pusher_task_episodes_in_this_demo":0,
        "worlds_with_same_blackbox_current_observation":len(desired),
        "original_L":.8,"source_repair_L":9.05,
        "physical_robot_outer_action_clip":2.,
        "world_target_actions":desired,
        "all_modes":options,
        "result_claim":"actuator reachability and history-information obstructions can coexist; repairing one cannot always eliminate the other",
    }


if __name__=="__main__":
    from json import dumps
    print(dumps(four_way_orthogonal_source_and_history_repair(),indent=2,sort_keys=True))
