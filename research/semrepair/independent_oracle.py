#!/usr/bin/env python3
"""Stand-alone exhaustive exact-rational oracle for SemRepair stopping.

No imports from SemRepair, PyTorch, Lean, simulator, or other third-party code.
This finite-domain oracle enumerates experiment partitions and uses Fraction
arithmetic to rule out floating-point discrepancies. It does NOT prove the
general upstream solver or any robot's task-level safety.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Hashable


@dataclass(frozen=True)
class Probe:
    name: str
    cost: Fraction
    output: tuple[tuple[str, Hashable], ...]
    risk: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("negative experiment cost")
        if len({h for h, _ in self.output}) != len(self.output):
            raise ValueError("duplicate hypothesis outcomes")

    def observation(self, world: str) -> Hashable:
        return dict(self.output)[world]


def optimum(
    hypotheses: tuple[str, ...],
    probes: tuple[Probe, ...],
    authority: dict[str, str],
    *,
    maximum_probe_risk: Fraction | None = None,
) -> Fraction | None:
    """Exact uniform-prior Bellman optimum; None means inadmissibly ambiguous.

    State is the surviving world set. Only probes that strictly partition it
    are considered. The posterior is uniform because this finite witness uses
    a uniform prior and deterministic outcomes. Every recursive child state is
    smaller, so this terminates with no iteration horizon or heuristic cap.
    """
    if not hypotheses or len(hypotheses) != len(set(hypotheses)):
        raise ValueError("empty/duplicate hypothesis set")
    if set(authority) != set(hypotheses):
        raise ValueError("incomplete authority map")
    for p in probes:
        if set(k for k, _ in p.output) != set(hypotheses):
            raise ValueError(f"probe {p.name}: incomplete outcome table")

    @lru_cache(maxsize=None)
    def solve(worlds: frozenset[str]) -> Fraction | None:
        if len({authority[w] for w in worlds}) == 1:
            return Fraction(0)

        choices: list[Fraction] = []
        for p in probes:
            if maximum_probe_risk is not None and p.risk > maximum_probe_risk:
                continue
            buckets: dict[Hashable, set[str]] = {}
            for w in worlds:
                buckets.setdefault(p.observation(w), set()).add(w)
            if len(buckets) < 2:
                continue  # Observation is identical; no information is gained.
            total = p.cost
            for child in buckets.values():
                future = solve(frozenset(child))
                if future is None:
                    break  # This probe leads to a terminal unidentifiable branch.
                total += Fraction(len(child), len(worlds)) * future
            else:
                choices.append(total)
        return min(choices) if choices else None

    return solve(frozenset(hypotheses))


def fixtures() -> dict[str, Fraction | None]:
    worlds = ("a0", "a1", "b", "c")
    library = (
        Probe(
            "repair-class",
            Fraction(1),
            (("a0", "A"), ("a1", "A"), ("b", "B"), ("c", "C")),
        ),
        Probe(
            "identity-refiner",
            Fraction(5),
            (("a0", 0), ("a1", 1), ("b", 0), ("c", 0)),
        ),
    )
    full_identity = optimum(worlds, library, {h: h for h in worlds})
    repair_authority = optimum(
        worlds, library, {"a0": "repair:A", "a1": "repair:A", "b": "repair:B", "c": "reject"}
    )

    aliases = ("correct", "masked-double-swap")
    externally_aliased = Probe(
        "external",
        Fraction(1),
        (("correct", (1, 0)), ("masked-double-swap", (1, 0))),
    )
    tapped = Probe(
        "internal-producer-tap",
        Fraction(2),
        (("correct", (1, 0)), ("masked-double-swap", (0, 1))),
        risk=Fraction(1, 4),
    )
    distinct_repairs = {"correct": "noop", "masked-double-swap": "producer+controller"}
    no_tap = optimum(aliases, (externally_aliased,), distinct_repairs)
    with_tap = optimum(aliases, (externally_aliased, tapped), distinct_repairs)
    forbidden_tap = optimum(
        aliases,
        (externally_aliased, tapped),
        distinct_repairs,
        maximum_probe_risk=Fraction(1, 8),
    )
    no_experiment_required = optimum(
        aliases, (), {"correct": "same-exact-bundle", "masked-double-swap": "same-exact-bundle"}
    )
    return {
        "full_abi_expected_cost": full_identity,
        "repair_authority_expected_cost": repair_authority,
        "external_only_cost": no_tap,
        "tap_resolved_cost": with_tap,
        "risk_blocked_cost": forbidden_tap,
        "same_authority_zero_probe_cost": no_experiment_required,
    }


def oracle_report() -> dict:
    result = fixtures()
    expected = {
        "full_abi_expected_cost": Fraction(7, 2),
        "repair_authority_expected_cost": Fraction(1),
        "external_only_cost": None,
        "tap_resolved_cost": Fraction(2),
        "risk_blocked_cost": None,
        "same_authority_zero_probe_cost": Fraction(0),
    }
    if result != expected:
        raise AssertionError(f"independent exact result does not match frozen table: {result!r}")
    return {
        "status": "independent_finite_oracle_pass",
        "solver": "uniform-prior exhaustive partition Bellman / fractions.Fraction",
        "full_cost_exact": str(result["full_abi_expected_cost"]),
        "authority_cost_exact": str(result["repair_authority_expected_cost"]),
        "external_alias_unidentifiable": result["external_only_cost"] is None,
        "tap_cost_exact": str(result["tap_resolved_cost"]),
        "risk_blocked_unidentifiable": result["risk_blocked_cost"] is None,
        "constant_authority_cost_exact": str(result["same_authority_zero_probe_cost"]),
        "scope": (
            "Exact finite deterministic uniform-prior witness only. "
            "Not a theorem for general stochastic observations, nonlinear robot dynamics, "
            "or externally maintained adoption."
        ),
    }


def self_test() -> None:
    assert oracle_report()["status"] == "independent_finite_oracle_pass"

    worlds = ("x", "y")
    test = Probe("split", Fraction(3, 2), (("x", "left"), ("y", "right")))
    same = {"x": "A", "y": "A"}
    different = {"x": "A", "y": "B"}
    assert optimum(worlds, (test,), same) == 0
    assert optimum(worlds, (test,), different) == Fraction(3, 2)
    assert optimum(worlds, (), different) is None
    with_obvious_tamper = Probe("does-not-split", Fraction(1), (("x", "same"), ("y", "same")))
    assert optimum(worlds, (with_obvious_tamper,), different) is None
    assert optimum(worlds, (test,), different, maximum_probe_risk=Fraction(0)) == Fraction(3, 2)
    try:
        optimum(worlds, (test,), {"x": "A"})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted incomplete authority map")
    print("stand-alone exact-oracle self-test: pass")


if __name__ == "__main__":
    self_test()
    print(json.dumps(oracle_report(), sort_keys=True))
