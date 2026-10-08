#!/usr/bin/env python3
"""Worst-case repair-authority diagnosis under *set-valued* noisy observations.

Finite, scalar, deterministic model predictions with bounded *adversarial*
additive observation error. A noisy observation can be compatible with several
worlds, but its compatible set need not form a disjoint partition. This script
solves the resulting exact belief-set game using fractions.Fraction.

Why: merging connected components of overlapping error balls is sound as a
conservative grouping but can falsely declare a genuinely distinguishable
active diagnosis problem unidentifiable. We keep the existing public v0.1
artifact pinned; this is an explicit falsification/correction candidate.

The minimax game and interval cell decomposition are established concepts,
NOT claims of mathematical priority or general robot-safety guarantees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

F = Fraction


@dataclass(frozen=True)
class Probe:
    name: str
    center_by_world: tuple[tuple[str, Fraction], ...]
    radius: Fraction
    cost: Fraction = F(1)
    risk: Fraction = F(0)

    def __post_init__(self):
        if self.radius < 0 or self.cost < 0 or self.risk < 0:
            raise ValueError("invalid negative bound, cost or risk")
        names = tuple(h for h, _ in self.center_by_world)
        if not names or len(set(names)) != len(names):
            raise ValueError("empty or duplicate predicted world names")

    def center(self, world: str) -> Fraction:
        return dict(self.center_by_world)[world]


def compatible_children(state: frozenset[str], probe: Probe) -> tuple[frozenset[str], ...]:
    """List *all distinct possible posterior sets* over a real-valued observation.

    For closed intervals, each elementary cell is either a boundary point or
    the open interval between adjacent unique boundary points. A midpoint
    represents the latter exactly (Fractions avoid rounding).
    """
    boundaries = sorted({
        probe.center(world) + sign * probe.radius
        for world in state for sign in (-1, 1)
    })
    if not boundaries:
        raise ValueError("empty diagnosis state")
    cells = set(boundaries)
    cells.update((l + r) / 2 for l, r in zip(boundaries, boundaries[1:]))
    children = {
        frozenset(
            h for h in state
            if abs(probe.center(h) - observed) <= probe.radius
        )
        for observed in cells
    }
    children.discard(frozenset())
    return tuple(sorted(children, key=lambda x: (len(x), tuple(sorted(x)))))


def connected_overlap_group_count(state: frozenset[str], probe: Probe) -> int:
    """Reproduce the old conservative overlap-component count for comparison."""
    remaining = set(state)
    components = 0
    while remaining:
        stack = [remaining.pop()]
        components += 1
        while stack:
            h = stack.pop()
            neighbors = {
                x for x in remaining
                if abs(probe.center(h) - probe.center(x)) <= 2 * probe.radius
            }
            remaining.difference_update(neighbors)
            stack.extend(neighbors)
    return components


def minimax_authority_cost(
    names: tuple[str, ...],
    authority: dict[str, str],
    probes: tuple[Probe, ...],
    *,
    maximum_probe_risk: Fraction | None = None,
) -> tuple[Fraction | None, dict]:
    """Exact worst-case cost, with state-indexed implementable policy witness.

    Re-use of probes is allowed but any chosen probe must strictly shrink the
    possible world set for *every* possible observation; if it leaves one
    observation compatible with the original state, an adversary could
    repeatedly return that outcome, so it cannot yield finite worst-case cost.
    """
    if not names or len(names) != len(set(names)) or set(names) != set(authority):
        raise ValueError("authority domain does not match unique nonempty worlds")
    for probe in probes:
        if {h for h, _ in probe.center_by_world} != set(names):
            raise ValueError(f"{probe.name}: incomplete prediction model")

    choices = {}

    @lru_cache(maxsize=None)
    def solve(state: frozenset[str]) -> Fraction | None:
        if len({authority[h] for h in state}) == 1:
            choices[state] = ("stop", (), F(0))
            return F(0)

        candidates = []
        for probe in probes:
            if maximum_probe_risk is not None and probe.risk > maximum_probe_risk:
                continue
            children = compatible_children(state, probe)
            if not children or state in children:
                continue
            downstream = [solve(c) for c in children]
            if any(value is None for value in downstream):
                continue
            cost = probe.cost + max(downstream)
            candidates.append((cost, probe.name, probe, children))
        if not candidates:
            choices[state] = ("reject", (), None)
            return None
        cost, _, chosen, children = min(candidates, key=lambda c: (c[0], c[1]))
        choices[state] = (chosen.name, children, cost)
        return cost

    root = frozenset(names)
    optimal = solve(root)
    # Re-verify policy correctness by independent recursion over children.
    visited = set()

    def verify(state: frozenset[str]) -> Fraction | None:
        if state in visited:
            return choices[state][2]
        visited.add(state)
        label, children, cost = choices[state]
        if label == "stop":
            assert len({authority[h] for h in state}) == 1 and cost == F(0)
            return F(0)
        if label == "reject":
            assert cost is None
            return None
        probe = next(p for p in probes if p.name == label)
        assert state not in children
        assert children == compatible_children(state, probe)
        results = [verify(c) for c in children]
        assert all(t is not None for t in results)
        recomputed = probe.cost + max(results)
        assert recomputed == cost
        return cost

    assert verify(root) == optimal
    report_policy = [
        {
            "state": sorted(state),
            "action": action,
            "children": [sorted(c) for c in children],
            "worst_case_cost_exact": str(cost) if cost is not None else None,
        }
        for state, (action, children, cost) in sorted(choices.items(), key=lambda x: (len(x[0]), sorted(x[0])))
    ]
    return optimal, {"nodes": report_policy, "verified_nodes": len(visited)}


def four_world_example(radius: Fraction = F(1), risk_cap: Fraction | None = None) -> dict:
    names = ("h0", "h1", "h2", "h3")
    p_a = Probe("A", tuple(zip(names, (F(0), F(3, 2), F(3), F(9, 2)))), radius)
    # Center-order permutation 2 -> 0 -> 3 -> 1. Both A and B have connected
    # overlap graphs, but *no pair of worlds overlaps under both probes*.
    p_b = Probe("B", tuple(zip(names, (F(3, 2), F(9, 2), F(0), F(3)))), radius, risk=F(1, 4))
    authorities = {h: "exact-repair:" + h for h in names}
    cost, certificate = minimax_authority_cost(
        names, authorities, (p_a, p_b), maximum_probe_risk=risk_cap,
    )
    naive_components = {
        p.name: connected_overlap_group_count(frozenset(names), p) for p in (p_a, p_b)
    }
    return {
        "error_radius_exact": str(radius),
        "connected_overlap_component_count": naive_components,
        "actual_possible_posterior_counts": {
            p.name: len(compatible_children(frozenset(names), p)) for p in (p_a, p_b)
        },
        "minimax_authority_cost_exact": str(cost) if cost is not None else None,
        "status": "identified" if cost is not None else "unsafe_unidentifiable",
        "certificate": certificate,
    }


def self_test() -> dict:
    success = four_world_example()
    assert success["connected_overlap_component_count"] == {"A": 1, "B": 1}
    assert success["minimax_authority_cost_exact"] == "2"
    assert success["status"] == "identified"
    assert four_world_example(risk_cap=F(0))["status"] == "unsafe_unidentifiable"
    # At radius 3/2, worlds h0 & h2 are simultaneously compatible in probe A
    # (touching at y=1.5) and in probe B (their centers 1.5 and 0 are close).
    # A worst-case observer cannot distinguish those exact different authorities.
    assert four_world_example(radius=F(3, 2))["status"] == "unsafe_unidentifiable"
    z, _ = minimax_authority_cost(
        ("h0", "h1"), {"h0": "same", "h1": "same"}, (),
    )
    assert z == 0

    # The exact touching boundary must not be dropped from the observation
    # partition by open-interval approximations.
    pair = Probe("touch", (("x", F(0)), ("y", F(2))), F(1))
    children = compatible_children(frozenset(("x", "y")), pair)
    assert frozenset(("x", "y")) in children
    assert len(children) == 3
    return {
        "status": "pass",
        "model": "bounded adversarial scalar observation, exact closed intervals",
        "claimed_math_priority": False,
        "false_unidentifiability_of_transitive_overlap_quotient": True,
        "two_adaptive_probes_worst_case_exact": "2",
        "radius_three_halves_rejected": True,
        "risk_forbidden_tap_rejected": True,
        "touching_boundary_preserved": True,
        "scope": (
            "Finite scalar deterministic predictions; worst-case adversarial noise "
            "and a declared probe library. No stochastic expectation, continuous "
            "control safety guarantee, or external robot ecosystem adoption."
        ),
    }


if __name__ == "__main__":
    print(json.dumps({"self_test": self_test(), "witness": four_world_example()}, indent=2, sort_keys=True))
