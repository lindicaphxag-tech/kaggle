#!/usr/bin/env python3
"""Independent (relative to the experiment runner) audit of public 64-state evidence.

Every episode is transcribed from the original GitHub Actions log's
FROZEN_TARGET_MEMORY_EPISODE JSON event; this program uses only the Python
standard library and never calls the policy, simulator, or author's
action-translation algorithm. An audit pass is still author-operated, *not*
third-party replication. The original immutable run and complete ZIP remain
linked inside prospective_pull_stack_64.json.

Usage:
    python research/semrepair/policy-transfer/audit_prospective_64.py
    python research/semrepair/policy-transfer/audit_prospective_64.py --self-test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

DATA_PATH = Path(__file__).with_name("prospective_pull_stack_64.json")
ARMS = ("source", "memory", "projected", "stateless", "naive")
SOURCE_NAMES = {
    "pull_cube": {
        "seed_start": 51001,
        "checkpoint_sha256":
            "74ae6a09b9af5e9e50dc71944f2e99316a8b67b02f3a96ca45df4a6d53dc1bd7",
        "original_log_job_ids": (113441827474, 113441827387, 113441827626, 113441827582),
        "source_policy_observation_width": 35,
        "chunk_artifact_ids": (11567061608, 11566388617, 11566353319, 11566732807),
        "chunk_artifact_zip_sha256": (
            "b02c3e4f0a667006a33637053ea169ff6f0f9794702797cdcf1eca3a90136442",
            "27368d0e3587a76fc3ee88689c1afb930ca14e5437469ebf14c68f00132d7e89",
            "d93c28fb900a3182bb49534f429f24c1bdb4e4c2e381481375f327b841b77935",
            "f13c1a6b2472b138ef0742c2b65db119c07a9741e4f2e7fe8021c27dd80fa11a",
        ),
    },
    "stack_cube": {
        "seed_start": 61001,
        "checkpoint_sha256":
            "e63cc8d8ffdca3d03553a21ea615c759b2b224493a7e7e12bee7efc29d5bad9c",
        "original_log_job_ids": (113441827912, 113441827469, 113441827476, 113441827424),
        "source_policy_observation_width": 48,
        "chunk_artifact_ids": (11567121270, 11566847601, 11566328610, 11566264133),
        "chunk_artifact_zip_sha256": (
            "1ed2df637427df21bd7bc5b28bd1a660842883937987ab8f57ff9d5d14c4c468",
            "f9318d96583c320decb6275a74ed70ee0a639b591d2d6be5681d01be3b9a8c2c",
            "3c8f163fe6e04476548e830bfbcde016eaf99c36bb82a86f7570b09226327217",
            "4f8f8e180e3e40766836773035ecd74a84f3b57ac123636bb2d10fab2600a7b7",
        ),
    },
}


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def finite(value: object) -> bool:
    return (type(value) in (float, int)
            and math.isfinite(value))


def audit(payload: dict) -> dict:
    require(payload.get("schema") == "semrepair_stateful_abi_prospective_64_log_extraction_v1",
            "unexpected artifact schema")
    require(payload.get("original_workflow_run_id") == 37815142810,
            "wrong original experiment identity")
    require(payload.get("original_experiment_commit") ==
            "9ffa86c6d3a86d2cee44b6e13c560033a8b373cf",
            "original executable identity changed")
    require(payload.get("preregistered_protocol_git_blob") ==
            "721509118c490f2e9a5a87a6570922b7106bfbea",
            "preregistered protocol changed")
    require(payload.get("pretrained_model_hf_revision") ==
            "6bdeb28810330ab5425ccd629bb561c58a56ff85",
            "public model revision changed")
    require(payload.get("no_new_policy_training") is True
            and payload.get("full_denominator_original_states") == 64
            and payload.get("planned_arms_per_state") == 5,
            "incompatible cohort or unreported retraining")
    require(payload.get("original_aggregator_job_id") == 113443743282,
            "aggregator identity mismatch")
    require(payload.get("original_aggregated_artifact_zip_sha256") ==
            "2a4701dcdc77f9f4f7cf72137e55172f8a6882001be5019d92b7c5af0c50225c",
            "original public archive digest changed")
    require(set(payload.get("datasets", {})) == set(SOURCE_NAMES),
            "missing/new task evidence")
    require(set(payload.get("summary", {})) == set(SOURCE_NAMES),
            "missing/new summary")
    result = {}
    all_seeds = set()

    for task, spec in SOURCE_NAMES.items():
        dataset = payload["datasets"][task]
        episodes = dataset.get("episodes")
        require(isinstance(episodes, list) and len(episodes) == 32,
                f"{task}: need all 32 original unselected states")
        require(tuple(dataset.get("original_log_job_ids", ())) ==
                spec["original_log_job_ids"],
                f"{task}: original eight-seed job identities not frozen")
        require(dataset.get("source_checkpoint_sha256") ==
                spec["checkpoint_sha256"],
                f"{task}: source PPO checkpoint does not match frozen model identity")
        require(dataset.get("source_policy_observation_width") ==
                spec["source_policy_observation_width"],
                f"{task}: source network input ABI changed")
        require(tuple(dataset.get("original_chunk_artifact_ids", ())) ==
                spec["chunk_artifact_ids"],
                f"{task}: source artifact identifiers were substituted")
        require(tuple(dataset.get("original_chunk_artifact_zip_sha256", ())) ==
                spec["chunk_artifact_zip_sha256"],
                f"{task}: original eight-seed artifact digests were substituted")
        seeds = [e.get("seed") for e in episodes]
        expected = list(range(spec["seed_start"], spec["seed_start"] + 32))
        require(seeds == expected, f"{task}: skipped/duplicated/reordered original seeds")
        require(not (set(seeds) & all_seeds), "tasks re-used source seeds")
        all_seeds.update(seeds)

        success = {a: 0 for a in ARMS}
        refusals = 0
        inexact_count = {"projected": 0, "stateless": 0}
        projected_only = stateless_only = 0

        for episode in episodes:
            flags = episode.get("success_once")
            require(isinstance(flags, dict) and set(flags) == set(ARMS),
                    f"{task}: one planned controller outcome missing")
            require(all(type(flags[a]) is bool for a in ARMS),
                    f"{task}: non-Boolean task success")
            for a in ARMS:
                success[a] += int(flags[a])
            projected_only += int(flags["projected"] and not flags["stateless"])
            stateless_only += int(flags["stateless"] and not flags["projected"])

            diffs = episode.get("initial_obs_diff")
            require(isinstance(diffs, dict) and set(diffs) ==
                    {"memory", "projected", "stateless", "naive"},
                    f"{task}: controller-target observation matching incomplete")
            require(all(finite(value) and 0 <= value <= 5e-4
                        for value in diffs.values()),
                    f"{task}: physical/task initial state mismatch")

            steps = episode.get("steps")
            require(isinstance(steps, dict), f"{task}: no executed-step proof")
            for a in ("source", "projected", "stateless", "naive"):
                require(type(steps.get(a)) is int and 1 <= steps[a] <= 50,
                        f"{task}: missing/incomplete physical execution of {a}")
            if "memory" in steps:
                require(type(steps["memory"]) is int and 1 <= steps["memory"] <= 50,
                        f"{task}: invalid exact mode episode steps")

            rejected = episode.get("refusals", {})
            require(isinstance(rejected, dict)
                    and set(rejected).issubset({"memory"}),
                    f"{task}: unauthorized/missing refusal arm")
            if rejected.get("memory") is not None:
                refusals += 1
                record = rejected["memory"]
                require(isinstance(record, dict)
                        and isinstance(record.get("reason"), str)
                        and type(record.get("step")) is int
                        and 0 <= record["step"] < 50
                        and finite(record.get("required_amp"))
                        and record["required_amp"] > 1,
                        f"{task}: exact mode refusal witness invalid")
                require(not flags["memory"], f"{task}: refusal incorrectly counted as success")
            elif "memory" not in steps:
                raise ValueError(f"{task}: memory run not executed or refused")

            approx = episode.get("approximations", {})
            require(isinstance(approx, dict)
                    and set(approx).issubset({"projected", "stateless"}),
                    f"{task}: invalid approximate-arm claim")
            for a, records in approx.items():
                require(isinstance(records, list),
                        f"{task}: corrupt approximation record")
                for rec in records:
                    require(isinstance(rec, dict)
                            and rec.get("exactness") == "NOT_EXACT"
                            and type(rec.get("step")) is int
                            and 0 <= rec["step"] < 50
                            and finite(rec.get("required_native_amp"))
                            and rec["required_native_amp"] > 1,
                            f"{task}: nonexact action silently certified exact")
                inexact_count[a] += len(records)
            require(finite(episode.get("max_required_native_amp"))
                    and episode["max_required_native_amp"] >= 0,
                    f"{task}: nonfinite or missing controller feasibility")

        report = {
            "n": len(episodes),
            "success_by_arm": success,
            "projected_only": projected_only,
            "stateless_only": stateless_only,
            "net_memory_gain": projected_only - stateless_only,
            "strict_memory_refused_episodes": refusals,
            "predeclared_source_competence": success["source"] >= 24,
            "predeclared_memory_gain_present":
                success["source"] >= 24 and (projected_only-stateless_only) >= 6,
        }
        require(report == payload["summary"][task],
                f"{task}: saved summary not derived from 32 original states")
        require(report["predeclared_memory_gain_present"],
                f"{task}: frozen primary memory gain gate not met")
        result[task] = dict(report, inexact_projection_steps=inexact_count,
                            source_checkpoint_sha256=spec["checkpoint_sha256"])
    require(len(all_seeds) == 64, "invalid complete cross-task denominator")
    return {"status": "passed", "original_state_episodes": 64,
            "controller_arm_episode_executions": 320,
            "tasks": result,
            "external_adoption_or_third_party_reproduction": False,
            "source_scope": "log-extracted author-operated PhysX replay; no real robot or unknown-ABI discovery"}


def self_test(original: dict) -> dict:
    audit(original)
    attacks = [
        ("missing original seed", lambda p: p["datasets"]["pull_cube"]["episodes"].pop()),
        ("reorder source states", lambda p: p["datasets"]["stack_cube"]["episodes"].reverse()),
        ("wrong checkpoint provenance", lambda p: p.update(pretrained_model_hf_revision="main")),
        ("wrong task-model SHA", lambda p: p["datasets"]["pull_cube"]
         .update(source_checkpoint_sha256="0"*64)),
        ("wrong raw archive digest", lambda p: p["datasets"]["stack_cube"]
         ["original_chunk_artifact_zip_sha256"].__setitem__(0,"0"*64)),
        ("fabricated improvement", lambda p: p["datasets"]["pull_cube"]["episodes"][0]
         ["success_once"].update(stateless=True)),
        ("nonboolean task success", lambda p: p["datasets"]["stack_cube"]["episodes"][0]
         ["success_once"].update(source=1)),
        ("missing target memory observation", lambda p: p["datasets"]["pull_cube"]["episodes"][0]
         ["initial_obs_diff"].pop("projected")),
        ("arbitrary extra task seed", lambda p: p["datasets"]["stack_cube"]["episodes"][0]
         .update(seed=51001)),
        ("misleading stored summary", lambda p: p["summary"]["pull_cube"]
         ["success_by_arm"].update(projected=32)),
        ("pretend exact approximation", _tamper_approximation),
    ]
    for description, damage in attacks:
        copied = copy.deepcopy(original)
        damage(copied)
        try:
            audit(copied)
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError(f"Audit accepted tampered evidence: {description}")
    return {"status": "passed", "tampering_mutations_rejected": len(attacks)}


def _tamper_approximation(payload: dict) -> None:
    for task in SOURCE_NAMES:
        for item in payload["datasets"][task]["episodes"]:
            for a in ("projected", "stateless"):
                if item.get("approximations", {}).get(a):
                    item["approximations"][a][0]["exactness"] = "EXACT"
                    return
    raise RuntimeError("no original nonexact projection events to corrupt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = audit(payload)
    if args.self_test:
        report["falsification_self_test"] = self_test(payload)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
