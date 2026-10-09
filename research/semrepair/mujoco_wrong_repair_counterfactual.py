"""Counterfactual wrong-interface repairs under native MuJoCo physics.

Predeclared experiment: 24 independent *command sequences*, 3 fixed fault
worlds, canonical reference, authorized patch, and compensating unauthorized
patch. Worlds share one hand-constructed robot family; trajectories are NOT
independent robot systems and this cannot establish general robot safety.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

from .mujoco_native_authority import (
    WORLDS, corrected, rollout, max_error, EFFECT_TOL
)

FREEZE_PROTOCOL = "semrepair-native-wrong-interface-counterfactual-v1"
SEED = 20261010
N_EXCITATIONS = 24
N_COMMANDS = 6
CONTROL_RANGE = (-0.8, 0.8)


def unauthorized_compensating_repair(world):
    """Deliberately repair the WRONG interface, preserving net dynamics."""
    if world.name in ("gain_fault", "nuisance_alias"):
        return replace(world, software_scale=world.software_scale / 2)
    if world.name == "producer_fault":
        return replace(world, gear=world.gear / 2)
    raise ValueError("no designated unauthorized counterfactual for " + world.name)


def run_counterfactual():
    rng = random.Random(SEED)
    scenarios = []
    fault_worlds = [w for w in WORLDS if w.name != "reference"]
    reference = next(w for w in WORLDS if w.name == "reference")
    for trial in range(N_EXCITATIONS):
        commands = tuple(round(rng.uniform(*CONTROL_RANGE), 6)
                         for _ in range(N_COMMANDS))
        scenarios.append({"index":trial,"commands":commands})
    hash_input = json.dumps(scenarios, sort_keys=True, separators=(",",":")).encode()
    freeze_hash = hashlib.sha256(hash_input).hexdigest()

    results = []
    for scenario in scenarios:
        commands = tuple(scenario["commands"])
        ref = rollout(reference, commands)["external"]
        for world in fault_worlds:
            true_fix = corrected(world)
            wrong_fix = unauthorized_compensating_repair(world)
            if wrong_fix == true_fix or wrong_fix == world:
                raise AssertionError("unauthorized and correct patch not genuinely distinct")
            bad = max_error(rollout(world, commands)["external"], ref)
            good = max_error(rollout(true_fix, commands)["external"], ref)
            wrong = max_error(rollout(wrong_fix, commands)["external"], ref)
            if bad <= 1e-4:
                raise AssertionError("fault is not observable under frozen command sequence")
            if good >= EFFECT_TOL or wrong >= EFFECT_TOL:
                raise AssertionError("counterfactual compensation hypothesis falsified")
            results.append({
                "trial": scenario["index"],
                "fault_world": world.name,
                "original_error_linf": bad,
                "authorized_patch_error_linf": good,
                "unauthorized_patch_error_linf": wrong,
                "authorized_change": (
                    "software_scale" if world.name=="producer_fault" else "gear"
                ),
                "unauthorized_change": (
                    "gear" if world.name=="producer_fault" else "software_scale"
                ),
                "physically_observationally_equivalent_repair": True,
                "semantic_patch_authority_equal": False,
            })
    return {
        "protocol": FREEZE_PROTOCOL, "seed": SEED,
        "command_draw_sha256": freeze_hash,
        "command_draw_domain": "six deterministic uniform[-0.8,0.8] scalar actions",
        "independent_command_sequences": N_EXCITATIONS,
        "fixed_robot_worlds": len(WORLDS),
        "fault_worlds": len(fault_worlds),
        "paired_fault_sequence_cells": len(results),
        "unauthorized_patches_with_matching_nominal_trajectory": sum(
            x["physically_observationally_equivalent_repair"] for x in results
        ),
        "max_correct_repair_error": max(x["authorized_patch_error_linf"] for x in results),
        "max_wrong_interface_repair_error": max(x["unauthorized_patch_error_linf"] for x in results),
        "min_original_fault_error": min(x["original_error_linf"] for x in results),
        "results": results,
        "claim_boundary": (
            "Controlled native MuJoCo actuator scaling with compensating semantics; "
            "not independent robot families, not an optimized active-diagnosis "
            "improvement, not real hardware, not a clinical safety demonstration."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = run_counterfactual()
    blob = json.dumps(report, indent=2, sort_keys=True)+"\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob, encoding="utf8")
    print(json.dumps({
        k: value for k, value in report.items() if k!="results"
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
