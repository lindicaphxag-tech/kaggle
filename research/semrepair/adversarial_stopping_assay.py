"""Exhaustive finite adversarial-noise replay of SemRepair authorization.

Bridge the exact minimax bounded-observation policy to the observation-only
executor, enumerating *every distinct compatible posterior* for every possible
true world. This is a deterministic finite diagnostic study, NOT physical
robot transfer or learned-policy task success.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from .blind_authority_runner import ProbeSpec, execute_blind
from .noisy_authority_game import (
    Probe, compatible_children, minimax_authority_cost,
)


def _representative(state: frozenset[str], probe: Probe,
                    child: frozenset[str]) -> Fraction:
    """Find an exact observation representative (endpoints + open cell midpoints)."""
    edges = sorted({probe.center(w) + sign * probe.radius
                    for w in state for sign in (-1, 1)})
    candidates = set(edges)
    candidates.update((a + b) / 2 for a, b in zip(edges, edges[1:]))
    for observation in sorted(candidates):
        possible = frozenset(
            w for w in state
            if abs(observation - probe.center(w)) <= probe.radius
        )
        if possible == child:
            return observation
    raise RuntimeError("planner posterior has no realizable observation")


def audit_minimax_blind_replay(
    *, worlds: Sequence[str], authority: dict[str, str],
    probes: Sequence[Probe], risk_cap: Fraction | None = None,
    maximum_traces: int = 20000,
) -> dict:
    """Validate minimax authorization on all finite observation cells.

    If no finite-cost identification policy exists, report refusal instead of
    inventing an authorized patch. Observations are produced by an external
    finite-world oracle; the policy receives only a belief set.
    """
    worlds = tuple(worlds)
    if maximum_traces <= 0 or len({p.name for p in probes}) != len(probes):
        raise ValueError("invalid finite-world replay setup")
    cost, cert = minimax_authority_cost(
        worlds, authority, tuple(probes), maximum_probe_risk=risk_cap,
    )
    if cost is None:
        return {
            "status": "UNIDENTIFIABLE_REFUSAL", "worlds": len(worlds),
            "certified_minimax_cost": None, "traces": 0,
            "mechanism_only_no_robot_task_result": True,
        }
    policy = {
        frozenset(item["state"]): item["action"]
        for item in cert["nodes"]
    }
    by_name = {p.name: p for p in probes}
    executor_probes = {
        p.name: ProbeSpec(
            p.name, dict(p.center_by_world), p.radius, p.cost, p.risk,
        ) for p in probes
    }
    total_traces = 0
    maximum_realized = Fraction(0)
    by_world = {}

    def choose(state: frozenset[str]) -> str:
        return policy.get(state, "reject")

    for true_world in worlds:
        costs: list[Fraction] = []

        def recurse(state: frozenset[str], path: list[tuple[str, Fraction]],
                    spent: Fraction) -> None:
            nonlocal total_traces, maximum_realized
            if len({authority[w] for w in state}) == 1:
                if total_traces >= maximum_traces:
                    raise RuntimeError("trace budget exceeded; incomplete coverage")
                steps = iter(path)

                def external_observe(probe_name: str) -> Fraction:
                    expected, reading = next(steps)
                    if expected != probe_name:
                        raise AssertionError("policy deviated from certified plan")
                    return reading

                result = execute_blind(
                    worlds=worlds, repair_by_world=authority,
                    probes=executor_probes, policy=choose, observe=external_observe,
                    total_cost_budget=cost,
                    max_single_probe_risk=(
                        risk_cap if risk_cap is not None else
                        max((p.risk for p in probes), default=Fraction(0))
                    ),
                )
                if result.status != "AUTHORIZED" or result.authorized_repair != authority[true_world]:
                    raise AssertionError("unsafe or wrong-authority finite replay")
                if result.probe_cost != spent or tuple(name for name, _ in path) != result.probes_used:
                    raise AssertionError("reported probe accounting disagrees with the trace")
                if spent > cost:
                    raise AssertionError("minimax cost certificate was violated")
                costs.append(spent)
                maximum_realized = max(maximum_realized, spent)
                total_traces += 1
                return

            action = policy.get(state)
            if action is None or action in {"reject", "stop"}:
                raise AssertionError("no action despite finite identified minimax cost")
            if action in [name for name, _ in path]:
                raise AssertionError("repeated probe is not supported by blind executor")
            probe = by_name[action]
            if risk_cap is not None and probe.risk > risk_cap:
                raise AssertionError("planner selected forbidden-risk probe")
            children = compatible_children(state, probe)
            if not children or state in children:
                raise AssertionError("policy did not strictly shrink every posterior")
            for child in children:
                if true_world not in child:
                    continue
                observation = _representative(state, probe, child)
                if abs(observation - probe.center(true_world)) > probe.radius:
                    raise AssertionError("adversarial observation inconsistent with true world")
                recurse(child, path + [(action, observation)], spent + probe.cost)

        recurse(frozenset(worlds), [], Fraction(0))
        if not costs:
            raise AssertionError("true world had no covered observation trace")
        by_world[true_world] = {
            "traces": len(costs),
            "worst_cost": str(max(costs)),
        }
    if maximum_realized != cost:
        raise AssertionError("claimed minimax bound not tight in exhaustive replay")
    return {
        "status": "ALL_FINITE_OBSERVATION_CELLS_AUTHORIZED",
        "worlds": len(worlds), "traces": total_traces,
        "certified_minimax_cost": str(cost),
        "observed_worst_case_cost": str(maximum_realized),
        "by_true_world": by_world,
        "mechanism_only_no_robot_task_result": True,
    }
