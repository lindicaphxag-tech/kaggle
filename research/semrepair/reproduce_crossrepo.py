#!/usr/bin/env python3
"""Independently verify the frozen public SemRepair quick reproduction.

This runner is hosted outside the Isaac-GR00T fork and uses only Python stdlib
and git. It re-checks *reported* outputs and a pinned source SHA; it is not an
independent research replication or evidence of upstream adoption.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

PUBLIC_REMOTE = "https://github.com/lindicaphxag-tech/Isaac-GR00T.git"
PUBLIC_BRANCH = "semantic-abi-v0.1"
PINNED_COMMIT = "f69254a1e3a621a10236902ac77e9c1b1c7de668"
SCHEMA = "semrepair-semantic-abi-quick-repro-v1"


def run(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args, cwd=cwd, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def verify_result(data: dict) -> dict:
    """Check concrete claims without trusting the producer's own assertions."""
    if data.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    numeric = {
        "full_abi_expected_cost": 3.5,
        "repair_authority_expected_cost": 1.0,
        "authority_saving": 2.5,
        "tap_resolved_cost": 2.0,
        "same_authority_zero_probe_cost": 0.0,
    }
    for field, expected in numeric.items():
        val = data.get(field)
        if isinstance(val, bool) or not isinstance(val, (float, int)):
            raise ValueError(f"{field}: not numeric")
        if not math.isfinite(val) or not math.isclose(val, expected, abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"{field}: expected {expected}, got {val!r}")

    for field, expected in {
        "external_alias_status": "unsafe_unidentifiable",
        "tap_resolved_status": "optimal",
        "dependency_drift_splits_authority": True,
        "independent_verifier_valid": True,
    }.items():
        if data.get(field) != expected:
            raise ValueError(f"{field}: expected {expected!r}, got {data.get(field)!r}")

    if not data["repair_authority_expected_cost"] < data["full_abi_expected_cost"]:
        raise ValueError("authority-optimal policy did not improve this fixture")
    return {
        "status": "verified",
        "artifact_sha": PINNED_COMMIT,
        "schema": SCHEMA,
        "verified_fields": len(numeric) + 4,
        "source_type": "self-authored public SemRepair artifact",
        "claim_boundary": (
            "Independent harness re-execution only; not independent third-party "
            "reproduction, prospective I2, task performance, or upstream adoption."
        ),
    }


def falsification_self_test() -> None:
    baseline = {
        "schema": SCHEMA,
        "full_abi_expected_cost": 3.5,
        "repair_authority_expected_cost": 1.0,
        "authority_saving": 2.5,
        "external_alias_status": "unsafe_unidentifiable",
        "tap_resolved_status": "optimal",
        "tap_resolved_cost": 2.0,
        "dependency_drift_splits_authority": True,
        "same_authority_zero_probe_cost": 0.0,
        "independent_verifier_valid": True,
    }
    verify_result(baseline)
    attacks = [
        ("numeric", {"authority_saving": 9.0}),
        ("unsafe_alias", {"external_alias_status": "optimal"}),
        ("tampered_dep", {"dependency_drift_splits_authority": False}),
        ("not_verified", {"independent_verifier_valid": False}),
        ("nonfinite", {"full_abi_expected_cost": float("nan")}),
    ]
    for name, delta in attacks:
        try:
            verify_result({**baseline, **delta})
        except ValueError:
            continue
        raise AssertionError(f"checker accepted falsified result: {name}")
    print("falsification self-test: 5/5 corrupted outputs rejected")


def reproduce(checkout: Path) -> dict:
    actual = run(["git", "rev-parse", "HEAD"], cwd=checkout)
    if actual != PINNED_COMMIT:
        raise ValueError(f"wrong public source SHA: {actual}")
    raw = run([sys.executable, "-m", "validation_semrepair.quick_repro"], cwd=checkout)
    result = json.loads(raw)
    return verify_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, help="Use a pre-fetched public checkout")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        falsification_self_test()
        return

    if args.checkout:
        report = reproduce(args.checkout.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="semrepair-crossrepo-") as td:
            dest = Path(td) / "artifact"
            run([
                "git", "clone", "--quiet", "--depth", "1", "--filter=blob:none",
                "--no-checkout", "--branch", PUBLIC_BRANCH, PUBLIC_REMOTE, str(dest),
            ])
            run(["git", "sparse-checkout", "set", "validation_semrepair"], cwd=dest)
            run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=dest)
            report = reproduce(dest)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
