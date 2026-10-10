"""Safe joint authority planning over SAME upstream actions and source-world maps.

This corrects a real methodological weakness of an action-space-only
reachability oracle: physical outputs reachable separately in each
unknown source world need not be obtainable with ONE input sent before
the defective software interface.

Candidate physical-source edits replace typed P/C/L/B mappings for ALL
possible worlds; candidate history-evidence interventions partition
the worlds into distinguishable observation fibers. Within each fiber
the selected policy MUST send the SAME upstream input to all worlds,
verified by the interval branch-and-bound robust controller.

Every candidate result is SAT (actual feasible upstream witness),
UNSAT (interval lower-bound obstruction, conditional on exact model),
UNKNOWN (proof computation exhausted), or UNLICENSED. A chosen plan's
finite-catalog minimum cost is CERTIFIED only after ALL cheaper candidates
are conclusively unlicensed or UNSAT, not silently ignoring unknowns.
No source, calibration, sensor or signer integrity is established here.
"""
from __future__ import annotations

from math import isfinite
from .robust_multiworld_upstream_control import solve_common_upstream_minimax

def choose_joint_source_history_with_common_input(
    *,source_catalog,information_catalog,desired_actions,
    error_tolerance,upstream_radii=(1.,1.),budget_per_fiber=6000
):
    if not source_catalog or not information_catalog:
        raise ValueError("nonempty approved intervention catalogs required")
    if not isinstance(desired_actions,(list,tuple)) or not desired_actions:
        raise ValueError("finite declared hidden-world action targets required")
    n=len(desired_actions)
    def verify_item(item,required):
        if not isinstance(item,dict) or set(item)!=required:
            raise ValueError("strict repair item schema")
        if not isinstance(item["name"],str) or not item["name"]:
            raise ValueError("valid candidate name required")
        if type(item["cost"]) not in (float,int) or not isfinite(item["cost"]) or item["cost"]<0:
            raise ValueError("finite nonnegative declared cost required")
        if type(item["authorized"]) is not bool:
            raise ValueError("Boolean asserted authority required")
    for a in source_catalog:
        verify_item(a,{"name","cost","authorized","per_world_source_model"})
        if len(a["per_world_source_model"])!=n:
            raise ValueError("every source patch must cover every source world")
    for e in information_catalog:
        verify_item(e,{"name","cost","authorized","world_observation_partition"})
        seen=[]
        for cls in e["world_observation_partition"]:
            if not cls:raise ValueError("empty observation class")
            seen.extend(cls)
        if sorted(seen)!=list(range(n)):
            raise ValueError("observation classes must partition all possible worlds")
    if len({a["name"] for a in source_catalog})!=len(source_catalog) or \
       len({e["name"] for e in information_catalog})!=len(information_catalog):
        raise ValueError("duplicate source/evidence repair names")
    results=[]
    for src in source_catalog:
        for ev in information_catalog:
            cost=float(src["cost"]+ev["cost"])
            rec={"source":src["name"],"evidence":ev["name"],
                 "declared_cost":cost,"source_installed":False,
                 "cross_domain_signatures_verified":False}
            if not src["authorized"] or not ev["authorized"]:
                rec.update(decision="UNLICENSED",fibers=[])
                results.append(rec)
                continue
            fibers=[]
            for cls in ev["world_observation_partition"]:
                models=[]
                for i in cls:
                    m=dict(src["per_world_source_model"][i])
                    m["desired"]=desired_actions[i]
                    models.append(m)
                proof=solve_common_upstream_minimax(
                    worlds=models,input_radii=upstream_radii,
                    accepted_error=error_tolerance,
                    max_nodes=budget_per_fiber,accuracy=.001)
                fibers.append({
                    "world_indices":list(cls),"decision":proof["decision"],
                    "same_upstream_witness":proof["unique_common_upstream_witness"],
                    "physical_world_outputs":proof["per_world_ACTUAL_physical_commands"],
                    "robust_lower":proof["global_minimax_lower_under_assumed_exact_model"],
                    "robust_upper":proof["global_minimax_upper_under_assumed_exact_model"],
                    "expanded_boxes":proof["expanded_interval_boxes"],
                })
            if all(x["decision"]=="SAT_WITH_COMMON_UPSTREAM_WITNESS" for x in fibers):
                choice="CERTIFIED_CONDITIONALLY_FEASIBLE"
            elif any(x["decision"]=="UNSAT_WITH_INTERVAL_LOWER_BOUND" for x in fibers):
                choice="CERTIFIED_CONDITIONALLY_INFEASIBLE"
            else:
                choice="UNKNOWN_NEEDS_STRONGER_PROOF"
            rec.update(decision=choice,fibers=fibers)
            results.append(rec)
    feasible=[r for r in results if r["decision"]=="CERTIFIED_CONDITIONALLY_FEASIBLE"]
    best=min(feasible,key=lambda r:(r["declared_cost"],r["source"],r["evidence"])) if feasible else None
    missing_cheaper=[r for r in results if r["decision"]=="UNKNOWN_NEEDS_STRONGER_PROOF"
                     and (best is None or r["declared_cost"]<best["declared_cost"])]
    if missing_cheaper:
        decision="UNKNOWN_CHEAPER_REPAIR_NOT_EXCLUDED"
        selected=None
    elif best is None:
        decision="REFUSE_NO_LICENSED_VERIFIED_SOURCE_ACTION_PLAN"
        selected=None
    else:
        decision="CONDITIONAL_FINITE_MIN_COST_SOURCE_REPAIR_SELECTED_NOT_INSTALLED"
        selected=best
    return {
        "decision":decision,"source_repair":selected["source"] if selected else None,
        "historical_evidence":selected["evidence"] if selected else None,
        "minimum_declared_finite_cost":selected["declared_cost"] if selected else None,
        "source_patch_installed":False,
        "signers_authenticity_established":False,
        "world_count":n,"evaluated_catalog_combinations":len(results),
        "source_world_models_assumed_complete":True,
        "full_robot_dynamics_recovery_proven":False,
        "candidate_records":results,
        "scope":"finite licensed source/evidence catalogs; common pre-interface input; conditional exact source maps",
    }


def run_robust_common_input_joint_repair():
    I=((1.,0.),(0.,1.));neg=((-1.,0.),(0.,1.))
    def model(P,L):
        return dict(P=P,C=I,L=(L,L),B=(2.,2.))
    source=[
        dict(name="unchanged_unknown_sign_clipped",
             authorized=True,cost=0,
             per_world_source_model=[model(I,.4),model(neg,.4)]),
        dict(name="authorized_one_source_clip_expansion",
             authorized=True,cost=2,
             per_world_source_model=[model(I,1.1),model(neg,1.1)]),
    ]
    evidence=[
        dict(name="current_observation_only",cost=0,authorized=True,
             world_observation_partition=[[0,1]]),
        dict(name="version_bound_historical_state",cost=1,authorized=True,
             world_observation_partition=[[0],[1]]),
    ]
    desired=[(.7,.8),(.7,.8)]
    outcomes={}
    for key,tol in (("loose",.8),("intermediate",.5),("strict",.1)):
        outcomes[key]=choose_joint_source_history_with_common_input(
            source_catalog=source,information_catalog=evidence,
            desired_actions=desired,error_tolerance=tol,
            budget_per_fiber=5000)
    restricted=[dict(s,authorized=(False if s["cost"]>0 else True)) for s in source]
    outcomes["strict_no_source_rights"]=choose_joint_source_history_with_common_input(
        source_catalog=restricted,information_catalog=evidence,
        desired_actions=desired,error_tolerance=.1,
        budget_per_fiber=5000)
    return {
        "protocol":"SemRepair-information-authority-upstream-worlds-min-cost-v2",
        "full_source_model_assumed":True,"faults_constructed_not_vendor":True,
        "author_chosen_edit_costs":True,
        "robot_episodes":0,
        "cases":outcomes,
    }

if __name__=="__main__":
    import json
    print(json.dumps(run_robust_common_input_joint_repair(),indent=2))
