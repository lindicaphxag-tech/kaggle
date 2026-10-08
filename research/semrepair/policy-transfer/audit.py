#!/usr/bin/env python3
"""Independent stdlib audit of frozen PPO controller-ABI transfer outcomes.

Reads contributor-generated raw episode records transcribed verbatim from
public GitHub Actions logs. It independently recomputes counts, discordances,
strict-vs-approximate actions, refusals and source-seed integrity.

The author controls this mirror and this script. A green run is NOT an
independent *third-party* replication, a hardware safety certificate,
upstream maintainer adoption, or evidence for unknown-contract discovery.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECT = {
    "pickcube_blind_21001_21032": {
        "task": "PickCube-v1", "role": "precommitted Holdout",
        "start": 21001, "run": 37809892720, "job": 113423711263,
        "commit": "d95f136972c69e10c03dfaf032adfe0a958c45d9",
        "blob": "520439a5ce3031d8e06a60802c3c9dd03e66a9cb",
        "weights": "3e6c95d63a2132843323e24cf7ba962b8cf2610f04b2a5a43f3efb6fef8497a8",
        "checkpoint_file": "ppo/pick_cube_final_ckpt.pt",
    },
    "stackcube_exploratory_50001_50032": {
        "task": "StackCube-v1", "role": "exploratory task",
        "start": 50001, "run": 37747596201, "job": 113212690356,
        "commit": "9e6ebf742ff97b7b0e7055231755848146dda73b",
        "blob": "2bb77429c3974ee076c57b003c911665570c6c31",
        "weights": "e63cc8d8ffdca3d03553a21ea615c759b2b224493a7e7e12bee7efc29d5bad9c",
        "checkpoint_file": "ppo/stack_cube_final_ckpt.pt",
    },
    "pushcube_exploratory_40001_40032": {
        "task": "PushCube-v1", "role": "exploratory task",
        "start": 40001, "run": 37810291098, "job": 113425079070,
        "commit": "23e2e72b6f75ae739d43db92e3c33c35585eee83",
        "blob": "aa23f9d34609c5051a728edf6eb4c2984359171a",
        "weights": "a4a02198b309e73cb877959079023d967d5f63ec78380de9703a10c9efafc0cf",
        "checkpoint_file": "ppo/push_cube_final_ckpt.pt",
    },
}
REQUIRED_ARMS = ("source", "memory", "projected", "naive")
MAX_HORIZON = 50
OBS_INITIAL_TOL = 5e-4


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def exact_bool(value) -> bool:
    return type(value) is bool


def valid_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def audit(slug: str, doc: dict) -> dict:
    require(slug in EXPECT, "unexpected task/holdout")
    spec = EXPECT[slug]
    expected_fields = {
        "schema": "semantic-abi-frozen-policy-transfer/raw-v1",
        "task": spec["task"], "cohort_role": spec["role"],
        "source_repo": "lindicaphxag-tech/ManiSkill",
        "run_id": spec["run"], "job_id": spec["job"],
        "run_head_sha": spec["commit"], "implementation_blob_sha": spec["blob"],
        "implementation_path": "research/frozen_ppo_target_memory.py",
        "checkpoint_repo": "kattri15/actionshift-baselines",
        "checkpoint_path": spec["checkpoint_file"],
        "checkpoint_sha256": spec["weights"],
        "episode_horizon_steps": MAX_HORIZON,
        "expected_seed_start": spec["start"],
        "expected_seed_count": 32,
        "author_operated": True,
        "original_job_url": (
            f"https://github.com/lindicaphxag-tech/ManiSkill/actions/"
            f"runs/{spec['run']}/job/{spec['job']}"
        ),
    }
    for field, wanted in expected_fields.items():
        require(type(doc.get(field)) is type(wanted) and doc.get(field) == wanted,
                f"{slug}: changed provenance {field}")

    records = doc.get("raw_episode_records")
    require(type(records) is list and len(records) == 32, f"{slug}: missing original rows")
    original_summary = doc.get("original_console_summary")
    require(type(original_summary) is dict, f"{slug}: original console summary missing")
    require(original_summary.get("episodes") == 32, f"{slug}: original episode denominator wrong")

    arms = list(REQUIRED_ARMS) + (
        ["stateless"] if "stateless" in original_summary.get("success", {}) else []
    )
    counts = {a: 0 for a in arms}
    refusal_count = 0
    approximate_counts = {a: 0 for a in arms}
    any_approx_episodes = 0
    per_seed = []
    maximum_init_diff = 0.0
    seen = set()
    for offset, entry in enumerate(records):
        seed = spec["start"] + offset
        require(type(entry) is dict and type(entry.get("seed")) is int and entry["seed"] == seed,
                f"{slug}: episode source seed changed or rearranged")
        require(seed not in seen, f"{slug}: duplicated seed")
        seen.add(seed)
        if "task" in entry:
            require(entry["task"] == spec["task"], f"{slug}: incorrect episode task label")
        suc = entry.get("success_once")
        require(type(suc) is dict and not (set(suc) - set(arms)),
                f"{slug}: unrecognized outcome fields")
        for arm, value in suc.items():
            require(exact_bool(value), f"{slug}: nonboolean task success")
            counts[arm] += int(value)
        init = entry.get("initial_obs_diff")
        require(type(init) is dict and set(init) == set(arms) - {"source"},
                f"{slug}: incomplete initial task/physical state comparison")
        for arm, value in init.items():
            require(valid_number(value) and 0 <= value <= OBS_INITIAL_TOL,
                    f"{slug}: initial state not source-equivalent ({arm})")
            maximum_init_diff = max(maximum_init_diff, value)
        steps = entry.get("steps")
        require(type(steps) is dict and not (set(steps) - set(arms)),
                f"{slug}: malformed execution horizons")
        for arm, step in steps.items():
            require(type(step) is int and 1 <= step <= MAX_HORIZON,
                    f"{slug}: invalid step horizon for {arm}")
        require("source" in steps and "source" in suc,
                f"{slug}: missing reference policy rollout")
        refusal = entry.get("refusals")
        require(type(refusal) is dict and not (set(refusal) - set(arms)),
                f"{slug}: malformed refusal record")
        if "memory" in refusal:
            refusal_count += 1
        for arm, obj in refusal.items():
            require(type(obj) is dict and type(obj.get("step")) is int
                    and 0 <= obj["step"] < MAX_HORIZON and type(obj.get("reason")) is str,
                    f"{slug}: refusal is missing step/reason")
            require(arm not in suc or suc[arm] is False,
                    f"{slug}: refused episode cannot also claim success")
        approx = entry.get("approximations")
        require(type(approx) is dict and not (set(approx) - set(arms)),
                f"{slug}: malformed approximation ledger")
        any_approx_episodes += bool(approx)
        for arm, events in approx.items():
            require(arm in ("projected", "stateless") and type(events) is list,
                    f"{slug}: strict action claims approximation")
            for event in events:
                require(type(event) is dict and event.get("exactness") == "NOT_EXACT"
                        and type(event.get("step")) is int
                        and 0 <= event["step"] < MAX_HORIZON
                        and valid_number(event.get("required_native_amp"))
                        and event["required_native_amp"] > 1.0,
                        f"{slug}: missing bounded-projection loss-of-exactness evidence")
            approximate_counts[arm] += len(events)
        require(valid_number(entry.get("max_required_native_amp"))
                and entry["max_required_native_amp"] >= 0,
                f"{slug}: invalid command representability amplitude")
        per_seed.append({a: bool(suc.get(a, False)) for a in arms})

    original_counts = original_summary.get("success")
    require(type(original_counts) is dict and set(original_counts) == set(arms),
            f"{slug}: summary arms do not match the frozen raw records")
    require(counts == original_counts, f"{slug}: recomputed success disagrees with original log")
    require(original_summary.get("refused") == refusal_count,
            f"{slug}: exact-only refusal count mismatch")
    reported_approx = original_summary.get("approximate_steps")
    if type(reported_approx) is dict:
        require({arm: approximate_counts[arm] for arm in reported_approx} == reported_approx,
                f"{slug}: projection step counts mismatch")
    else:
        require(reported_approx == approximate_counts.get("projected"),
                f"{slug}: projected step count mismatch")

    def pair(a: str, b: str) -> dict:
        return {
            "a_only": sum(r[a] and not r[b] for r in per_seed),
            "b_only": sum(r[b] and not r[a] for r in per_seed),
            "both_success": sum(r[a] and r[b] for r in per_seed),
            "both_failure": sum(not r[a] and not r[b] for r in per_seed),
            "net_b_minus_a": counts[b] - counts[a],
        }

    return {
        "task": spec["task"], "cohort_role": spec["role"],
        "original_run": spec["run"], "run_head_sha": spec["commit"],
        "frozen_checkpoint_sha256": spec["weights"],
        "episodes": 32, "source_seed_start": spec["start"],
        "success": counts, "exact_only_refusal_episodes": refusal_count,
        "nonexact_projection_steps": approximate_counts,
        "episodes_with_any_approximation": any_approx_episodes,
        "maximum_initial_observation_difference": maximum_init_diff,
        "paired_projected_vs_naive": pair("naive", "projected"),
        "paired_projected_vs_exact_only": pair("memory", "projected"),
        "paired_projected_vs_source": pair("source", "projected"),
        "external_credit": {
            "independent_third_party_reproduction": False,
            "upstream_semrepair_adoption": False,
            "model_retrained_for_transfer": False,
            "automatically_discovered_unknown_contract": False,
        },
        "claim_limit": (
            "One frozen PPO and selected source/target control modes in "
            "a controlled PhysX CPU simulator. Bounded projection is NOT "
            "an exact semantic equivalence or physical safety certificate. "
            "Only PickCube seeds 21001-21032 had a predeclared confirmatory "
            "holdout; the other task cases are exploratory."
        ),
    }


def self_test(payloads: dict[str, dict]) -> None:
    for slug, original in payloads.items():
        audit(slug, original)
        attacks = (
            ("outcome", lambda x: x["raw_episode_records"][0]["success_once"].update(
                projected=not x["raw_episode_records"][0]["success_once"]["projected"]
            )),
            ("duplicate_seed", lambda x: x["raw_episode_records"][0].update(
                seed=x["raw_episode_records"][1]["seed"]
            )),
            ("tampered_checkpoint", lambda x: x.update(checkpoint_sha256="0" * 64)),
            ("omitted_episode", lambda x: x["raw_episode_records"].pop()),
            ("posthoc_summary", lambda x: x["original_console_summary"]["success"].update(
                source=-1
            )),
            ("abnormal_init", lambda x: x["raw_episode_records"][0][
                "initial_obs_diff"
            ].update(naive=10.0)),
        )
        for name, mutate in attacks:
            altered = copy.deepcopy(original)
            mutate(altered)
            try:
                audit(slug, altered)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{slug}: auditor accepted attack {name}")


def main() -> None:
    payloads = {
        slug: json.loads((HERE / "raw" / f"{slug}.json").read_text(encoding="utf-8"))
        for slug in EXPECT
    }
    self_test(payloads)
    result = {
        "status": "pass", "audited": [audit(k, v) for k, v in payloads.items()],
        "source_type": "author-operated archived GitHub job log episode extracts",
        "no_simulator_required": True,
        "tamper_rejections_per_task": 6,
        "third_party_reproduction": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
