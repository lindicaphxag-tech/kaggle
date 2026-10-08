#!/usr/bin/env python3
"""Predeclared/secondary paired inference for 96 frozen PPO task episodes.

Original per-episode data and checkpoints remain pinned in raw/*.json. This
program independently recomputes the exact discordant-pair binomial p-value
(McNemar's conditional null test) and armwise Wilson intervals. Exploratory
tasks retain their exploratory status; p-values cannot confer preregistration,
third-party adoption, or independence across repeated policies.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from audit import EXPECT, audit

HERE = Path(__file__).resolve().parent
Z95 = 1.959963984540054


def success(entry: dict, arm: str) -> bool:
    value = entry.get("success_once", {}).get(arm, False)
    if type(value) is not bool:
        raise ValueError(f"non-boolean success field: {arm}")
    return value


def wilson_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("invalid binomial count")
    p = k / n
    z2 = Z95 * Z95
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    radius = Z95 / denom * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar_two_sided(b: int, c: int) -> float:
    if type(b) is not int or type(c) is not int or min(b, c) < 0:
        raise ValueError("invalid discordance count")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def paired(rows: list[dict], treatment: str, control: str) -> dict:
    n = len(rows)
    if n <= 0:
        raise ValueError("empty paired cohort")
    both = sum(success(r, treatment) and success(r, control) for r in rows)
    gain = sum(success(r, treatment) and not success(r, control) for r in rows)
    loss = sum(not success(r, treatment) and success(r, control) for r in rows)
    neither = n - both - gain - loss
    treated = both + gain
    controls = both + loss
    assert all(x >= 0 for x in (both, gain, loss, neither))
    return {
        "treatment": treatment,
        "control": control,
        "n_paired": n,
        "both_success": both,
        "treatment_only_success": gain,
        "control_only_success": loss,
        "both_failure": neither,
        "treatment_success": treated,
        "control_success": controls,
        "absolute_task_success_difference": (gain - loss) / n,
        "exact_two_sided_mcnemar_p": exact_mcnemar_two_sided(gain, loss),
        "treatment_wilson_95": [round(x, 8) for x in wilson_95(treated, n)],
        "control_wilson_95": [round(x, 8) for x in wilson_95(controls, n)],
    }


def self_test() -> None:
    assert exact_mcnemar_two_sided(0, 0) == 1
    assert exact_mcnemar_two_sided(1, 0) == 1
    assert exact_mcnemar_two_sided(32, 0) == 2 / 2 ** 32
    assert exact_mcnemar_two_sided(0, 32) == 2 / 2 ** 32
    assert exact_mcnemar_two_sided(11, 1) == 26 / 4096
    assert exact_mcnemar_two_sided(4, 4) == 1
    assert wilson_95(0, 32)[0] == 0
    assert wilson_95(32, 32)[1] == 1
    example = [
        {"success_once": {"projected": True, "naive": False}},
        {"success_once": {"projected": True, "naive": True}},
        {"success_once": {"projected": False, "naive": True}},
        {"success_once": {"projected": False, "naive": False}},
    ]
    assert paired(example, "projected", "naive")["exact_two_sided_mcnemar_p"] == 1
    for invalid in ((-1, 0), (1, -1), (0.5, 1)):
        try:
            exact_mcnemar_two_sided(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid discordance: {invalid}")


def main() -> None:
    self_test()
    reports = []
    for slug in EXPECT:
        raw = (HERE / "raw" / f"{slug}.json").read_bytes()
        doc = json.loads(raw)
        # Evidence audit performs full source, seed, checkpoint and ledger
        # validation before any statistical report is allowed.
        existing = audit(slug, doc)
        rows = doc["raw_episode_records"]
        comparisons = {
            "projected_vs_naive": paired(rows, "projected", "naive"),
            "projected_vs_source": paired(rows, "projected", "source"),
            "projected_vs_exact_only": paired(rows, "projected", "memory"),
        }
        if doc["task"] == "PushCube-v1":
            comparisons["projected_vs_stateless"] = paired(rows, "projected", "stateless")
        if comparisons["projected_vs_naive"]["treatment_success"] != existing["success"]["projected"]:
            raise AssertionError("inconsistent archived projected count")
        if comparisons["projected_vs_naive"]["control_success"] != existing["success"]["naive"]:
            raise AssertionError("inconsistent archived control count")
        reports.append({
            "task": doc["task"],
            "cohort_role": doc["cohort_role"],
            "confirmatory_primary": slug == "pickcube_blind_21001_21032",
            "original_run": doc["run_id"],
            "original_implementation_sha": doc["implementation_blob_sha"],
            "frozen_checkpoint_sha256": doc["checkpoint_sha256"],
            "raw_episode_sha256": hashlib.sha256(raw).hexdigest(),
            "episode_count": len(rows),
            "comparisons": comparisons,
            "strict_refusal_episode_count": existing["exact_only_refusal_episodes"],
            "approximate_projection_step_count": existing["nonexact_projection_steps"]["projected"],
        })
    print(json.dumps({
        "status": "pass",
        "method": "exact conditional binomial McNemar for paired successes",
        "cohort_count": len(reports),
        "total_distinct_episodes": sum(d["episode_count"] for d in reports),
        "reports": reports,
        "inference_boundary": (
            "PickCube comparison was predeclared on frozen seeds. StackCube "
            "and PushCube are exploratory; p-values are descriptive for fixed "
            "checkpoints, deterministic simulator settings and episode-seed cohorts. "
            "Armwise Wilson intervals do not represent a CI for paired benefit. "
            "No between-robot generalization, hardware guarantee, independent "
            "external adoption, or unknown-contract identification is claimed."
        ),
        "third_party_independent_validation": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
