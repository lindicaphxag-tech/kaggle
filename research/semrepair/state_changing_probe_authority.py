"""Finite-state repair authority under diagnostics that mutate hidden program state.

This is a sound *conditional finite-model* demonstration, not a new POMDP
optimality theorem, production LeRobot execution, or robot-safety certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Optional


@dataclass(frozen=True, order=True)
class Hidden:
    implementation: str
    dirty: bool
    patch_hash: str


@dataclass(frozen=True)
class Probe:
    name: str
    cost: Fraction
    exposure: Fraction
    trusted: bool
    step: Callable[[Hidden], tuple[str, Hidden]]


@dataclass(frozen=True)
class Plan:
    action: str
    worst_cost: Fraction
    branches: tuple[tuple[str, "Plan"], ...] = ()


class StatefulAuthorityPlanner:
    def __init__(self, probes: tuple[Probe, ...], exposure_budget: Fraction,
                 horizon: int, *, complete_model: bool):
        if exposure_budget < 0 or horizon < 0:
            raise ValueError("negative exposure or planning horizon")
        if len({p.name for p in probes}) != len(probes):
            raise ValueError("duplicate probe name")
        if any(p.cost <= 0 or p.exposure < 0 for p in probes):
            raise ValueError("probes require positive cost and nonnegative exposure")
        self.probes = {p.name: p for p in probes}
        self.exposure_budget = Fraction(exposure_budget)
        self.horizon = horizon
        self.complete_model = complete_model

    @staticmethod
    def installable(belief: frozenset[Hidden]) -> bool:
        return bool(belief) and len({w.patch_hash for w in belief}) == 1 and all(
            not w.dirty and w.patch_hash != "UNKNOWN" for w in belief
        )

    def solve(self, belief: frozenset[Hidden],
              *, cost_limit: Optional[Fraction] = None) -> Optional[Plan]:
        if not belief or not self.complete_model:
            return None

        @lru_cache(maxsize=None)
        def rec(states: frozenset[Hidden], remaining: Fraction,
                steps: int) -> Optional[Plan]:
            if self.installable(states):
                return Plan("INSTALL:" + next(iter(states)).patch_hash, Fraction(0))
            if steps == 0:
                return None
            winner: Optional[Plan] = None
            for name, p in sorted(self.probes.items()):
                if not p.trusted or p.exposure > remaining:
                    continue
                children: dict[str, set[Hidden]] = {}
                for old in states:
                    obs, new_state = p.step(old)
                    if not isinstance(obs, str) or not isinstance(new_state, Hidden):
                        raise TypeError("probe outcome must be a string and Hidden")
                    # Critical: carry the POST-probe hidden state into every branch.
                    children.setdefault(obs, set()).add(new_state)
                branch_plans = []
                for obs, possible in sorted(children.items()):
                    child = rec(frozenset(possible), remaining-p.exposure, steps-1)
                    if child is None:
                        break
                    branch_plans.append((obs, child))
                if len(branch_plans) != len(children):
                    continue
                proposal = Plan(name, p.cost + max(
                    ch.worst_cost for _, ch in branch_plans), tuple(branch_plans))
                if winner is None or (proposal.worst_cost, proposal.action) < (
                        winner.worst_cost, winner.action):
                    winner = proposal
            return winner

        selected = rec(belief, self.exposure_budget, self.horizon)
        if selected is None or (cost_limit is not None and
                                selected.worst_cost > cost_limit):
            return None
        return selected

    def replay_all(self, start: frozenset[Hidden],
                   plan: Plan) -> tuple[Fraction, int]:
        """Independently replay every starting world and check final authority."""
        if not self.complete_model:
            raise ValueError("cannot certify an incomplete world catalog")
        max_spent, episodes = Fraction(0), 0
        for original in start:
            current, remaining, node, spent = (
                original, self.exposure_budget, plan, Fraction(0))
            for _ in range(self.horizon + 1):
                if node.action.startswith("INSTALL:"):
                    claimed = node.action.split(":", 1)[1]
                    assert claimed == current.patch_hash and not current.dirty
                    episodes += 1
                    max_spent = max(max_spent, spent)
                    break
                probe = self.probes[node.action]
                assert probe.trusted and probe.exposure <= remaining
                remaining -= probe.exposure
                spent += probe.cost
                observed, current = probe.step(current)
                node = dict(node.branches)[observed]
            else:
                raise AssertionError("authorization did not terminate")
        assert max_spent == plan.worst_cost
        return max_spent, episodes


def witness(*, assumed_static: bool = False, trusted_fast: bool = True,
            include_snapshot: bool = True):
    """Two implementation versions; a fast diagnostic dirties an episode cache."""
    start = frozenset((Hidden("A", False, "patch-A"),
                       Hidden("B", False, "patch-B")))

    def fast(w: Hidden):
        return w.implementation, Hidden(
            w.implementation, w.dirty if assumed_static else True, w.patch_hash)

    def reset(w: Hidden):
        return "reset-complete", Hidden(w.implementation, False, w.patch_hash)

    def snapshot(w: Hidden):
        return w.implementation, w

    probes = [Probe("fast_trace", Fraction(1), Fraction(1, 2), trusted_fast, fast),
              Probe("reset_epoch", Fraction(1), Fraction(0), True, reset)]
    if include_snapshot:
        probes.append(Probe("nonmutating_snapshot", Fraction(3),
                            Fraction(0), True, snapshot))
    return StatefulAuthorityPlanner(
        tuple(probes), Fraction(1, 2), 3, complete_model=True), start


def report():
    import json
    planner, start = witness()
    result = planner.solve(start)
    assert result is not None
    replay_cost, n = planner.replay_all(start, result)
    naive, naive_start = witness(assumed_static=True)
    naive_result = naive.solve(naive_start)
    assert naive_result is not None
    actual, _ = witness()
    wrong = sum(
        bool(dict(naive_result.branches)[y].action.startswith("INSTALL:") and
             changed.dirty)
        for original in naive_start
        for y, changed in [actual.probes["fast_trace"].step(original)]
    )
    restricted, restricted_start = witness(trusted_fast=False)
    restricted_answer = restricted.solve(restricted_start)
    return {
        "scope": "two author-constructed deterministic hidden versions",
        "state_aware_minimax_cost": str(result.worst_cost),
        "state_aware_first_probe": result.action,
        "all_world_replay_cost": str(replay_cost),
        "replayed_worlds": n,
        "state_oblivious_optimistic_cost": str(naive_result.worst_cost),
        "invalid_naive_authorizations": wrong,
        "cost_cap_1_refusal": planner.solve(start, cost_limit=Fraction(1)) is None,
        "without_trusted_fast_minimax_cost": str(restricted_answer.worst_cost),
        "assumptions": "complete finite worlds, deterministic honest observations, declared trust and transitions",
        "not_demonstrated": "vendor source integration, policy recovery gains, external adoption, general program synthesis",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(report(), sort_keys=True, indent=2))
