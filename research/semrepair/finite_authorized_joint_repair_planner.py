"""Minimum licensed information-plus-actuation repair over true reachable sets.

Inputs are FINITE externally supplied intervention catalogs, two-dimensional
physical reachable polygon unions, and a finite set of world-specific
desired commands under the same current-only observable information.

A candidate plan (a,b) combines ONE action-software interface and ONE
information-evidence source. For each equivalence class of still hidden
worlds it calls the verified physical-reachability Chebyshev minimax
solver. The robust bound for the plan is the max across those classes.
If any class has error exceeding the declared tolerance it CANNOT be
authorized. Otherwise select minimum DECLARED edit/evidence cost.
Exact global optimum relative to this finite licensed catalog only.

This is not new combinatorial optimization or a physical future-task
success theorem. The mechanism-level contribution is integrating real
reachability and observation ambiguity in a verifiable source-repair
decision rather than always rewriting the interface.

External permission fields are ASSERTIONS, NOT cryptographically verified
publisher credentials. No program bytes are written by this module; a
source patch that is selected must still pass the separate source-signing
and drift-certificate activation before execution.
"""
from __future__ import annotations
from hashlib import sha256
from json import dumps
from math import isfinite

from .joint_information_reachability_operator import (
    robust_blackbox_action_error_floor
)
from .strong_clipped_actuator_oracle import output_reachability_polygons


def _check_partition(classes,n):
    if not isinstance(classes,(tuple,list)) or not classes:
        raise ValueError("nonempty observation partition required")
    flat=[]
    for c in classes:
        if not isinstance(c,(tuple,list)) or not c:
            raise ValueError("empty observation class not allowed")
        for i in c:
            if type(i) is not int or i<0 or i>=n:
                raise ValueError("out-of-range world index")
            flat.append(i)
    if sorted(flat)!=list(range(n)):
        raise ValueError("all worlds must occur exactly once across observation classes")
    return [tuple(c) for c in classes]


def _trusted_catalog_item(obj,required,kind):
    if not isinstance(obj,dict) or set(obj)!=required:
        raise ValueError("strict "+kind+" intervention schema required")
    if not isinstance(obj["name"],str) or not obj["name"]:
        raise ValueError("nonblank intervention name")
    if type(obj["cost"]) not in (float,int) or not isfinite(obj["cost"]) or obj["cost"]<0:
        raise ValueError("nonnegative finite declared intervention cost required")
    if type(obj["authorized"]) is not bool:
        raise ValueError("explicit caller-asserted permission bool required")


def minimize_joint_authorized_repair(
    *,desired_actions,action_software_catalog,evidence_catalog,
    max_per_action_error,search_accuracy=1e-8
):
    """Find and witness cheapest finite plan meeting worst-world action error."""
    if not isinstance(desired_actions,(tuple,list)) or not desired_actions or \
       not all(isinstance(x,(tuple,list)) and len(x)==2 and
               all(type(v) in (float,int) and isfinite(v) for v in x)
               for x in desired_actions):
        raise ValueError("nonempty finite 2D desired target action worlds")
    if type(max_per_action_error) not in (int,float) or \
       not isfinite(max_per_action_error) or max_per_action_error<0:
        raise ValueError("finite nonnegative action tolerance required")
    if not action_software_catalog or not evidence_catalog:
        raise ValueError("both intervention authority catalogs required")
    worlds=tuple(tuple(float(v) for v in x) for x in desired_actions)
    action_names=set()
    for x in action_software_catalog:
        _trusted_catalog_item(
            x,{"name","cost","authorized","physical_reachable_polygons"},
            "actuator software")
        if x["name"] in action_names:
            raise ValueError("duplicate source software candidate")
        action_names.add(x["name"])
    evidence_names=set()
    for x in evidence_catalog:
        _trusted_catalog_item(
            x,{"name","cost","authorized","observation_world_partition"},
            "information evidence")
        if x["name"] in evidence_names:
            raise ValueError("duplicate evidence intervention")
        evidence_names.add(x["name"])
        _check_partition(x["observation_world_partition"],len(worlds))

    reports=[]
    for source in action_software_catalog:
        for evidence in evidence_catalog:
            licensed=source["authorized"] and evidence["authorized"]
            report={
                "source_component":source["name"],
                "information_component":evidence["name"],
                "total_declared_cost":float(source["cost"]+evidence["cost"]),
                "licensed":licensed,
                "source_bytes_actually_modified":False,
                "license_roots_cryptographically_verified_here":False,
            }
            if not licensed:
                report.update(decision="REFUSE_MISSING_EXTERNAL_AUTHORITY",
                              worst_world_action_error_upper=None,
                              observation_class_witnesses=[])
                reports.append(report)
                continue
            perclass=[]
            for indices in _check_partition(
                evidence["observation_world_partition"],len(worlds)
            ):
                local=robust_blackbox_action_error_floor(
                    indistinguishable_desired_actions=[worlds[i] for i in indices],
                    reachable_actuator_polygons=source["physical_reachable_polygons"],
                    absolute_search_accuracy=search_accuracy)
                perclass.append({
                    "indistinguishable_world_indices":list(indices),
                    "error_lower":local["tight_numerical_worst_case_minimax_error_lower"],
                    "error_upper":local["tight_numerical_worst_case_minimax_error_upper"],
                    "actual_control_witness":local["reachable_physical_action_witness"],
                })
            achieved=max(x["error_upper"] for x in perclass)
            okay=achieved<=max_per_action_error
            report.update({
                "decision":("CONDITIONALLY_CONTRACT_VALID" if okay else
                            "REFUSE_WRONG_ACTION_OR_INCOMPLETE_EVIDENCE"),
                "worst_world_action_error_upper":achieved,
                "observation_class_witnesses":perclass,
                "conditional_action_tolerance_met":okay,
            })
            reports.append(report)
    acceptable=[r for r in reports if r["decision"]=="CONDITIONALLY_CONTRACT_VALID"]
    best=min(acceptable,key=lambda x:(
        x["total_declared_cost"],
        x["worst_world_action_error_upper"],
        x["source_component"],x["information_component"]
    )) if acceptable else None
    return {
        "decision":("MIN_COST_CONDITIONAL_REPAIR_SELECTED_NOT_INSTALLED"
                    if best is not None else "REFUSE_NO_AUTHORIZED_FEASIBLE_REPAIR"),
        "selected_source_component":best["source_component"] if best else None,
        "selected_information_component":best["information_component"] if best else None,
        "minimum_catalog_cost":best["total_declared_cost"] if best else None,
        "worst_case_action_error_upper_bound":(
            best["worst_world_action_error_upper"] if best else None),
        "action_tolerance":float(max_per_action_error),
        "world_count":len(worlds),"number_of_catalog_plans":len(reports),
        "source_configuration_installed":False,
        "external_signer_and_telemetry_authentication":False,
        "optimality_is_finite_catalog_only":True,
        "plan_proof_records":reports,
        "limitations":"exact relative to finite candidate catalogs and caller-supplied physical polygons, world classes and authority flags, not original VLA task success",
    }


def two_fault_source_and_history_cost_sweep():
    C=((1.,-3.5),(0.,1.))
    broken=output_reachability_polygons(C,inner_bound=.8,outer_bound=2.)
    source_fixed=output_reachability_polygons(C,inner_bound=9.05,outer_bound=2.)
    actions=[
        {"name":"source_current","cost":0,"authorized":True,
         "physical_reachable_polygons":broken},
        {"name":"one_field_source_clip","cost":2,"authorized":True,
         "physical_reachable_polygons":source_fixed},
    ]
    evidence=[
        {"name":"current_only","cost":0,"authorized":True,
         "observation_world_partition":[[0,1]]},
        {"name":"verified_epoch_anchor","cost":1,"authorized":True,
         "observation_world_partition":[[0],[1]]},
    ]
    worlds=[[-2.,1.5],[-1.6,1.5]]
    result={}
    for key,tolerance in (
        ("loose",.8),("medium",.25),("strict",.05)
    ):
        result[key]=minimize_joint_authorized_repair(
            desired_actions=worlds,action_software_catalog=actions,
            evidence_catalog=evidence,max_per_action_error=tolerance)
    blocked=[dict(x,authorized=(False if x["name"]=="one_field_source_clip"
                                else x["authorized"])) for x in actions]
    result["strict_source_not_licensed"]=minimize_joint_authorized_repair(
        desired_actions=worlds,action_software_catalog=blocked,
        evidence_catalog=evidence,max_per_action_error=.05)
    return {
        "protocol":"authorized-source-information-cartesian-intervention-minimal-cost-v1",
        "worlds":worlds,"costs_are_author_supplied":True,
        "no_real_robot_tasks_in_this_counterexample":True,
        "results":result,
    }


if __name__=="__main__":
    print(dumps(two_fault_source_and_history_cost_sweep(),
                indent=2,sort_keys=True))
