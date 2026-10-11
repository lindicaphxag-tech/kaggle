"""Blind finite-world semantic repair execution against an observation-only tap.

The acting policy sees chosen probe observations, not the hidden true world.
The external evaluator must isolate source/configuration/ground truth; Python
in-process functions alone are NOT an OS isolation security boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    predictions: dict[str, Fraction]
    radius: Fraction
    cost: Fraction = Fraction(1)
    risk: Fraction = Fraction(0)

    def __post_init__(self):
        if not self.name or self.radius < 0 or self.cost < 0 or self.risk < 0:
            raise ValueError("invalid probe")
        if not self.predictions:
            raise ValueError("no predicted worlds")


@dataclass(frozen=True)
class BlindResult:
    status: str
    authorized_repair: str | None
    surviving_worlds: tuple[str, ...]
    probes_used: tuple[str, ...]
    probe_cost: Fraction


def execute_blind(*, worlds: tuple[str, ...], repair_by_world: dict[str, str],
                  probes: dict[str, ProbeSpec],
                  policy: Callable[[frozenset[str]], str],
                  observe: Callable[[str], Fraction],
                  total_cost_budget: Fraction,
                  max_single_probe_risk: Fraction) -> BlindResult:
    """Allow only one authority shared by all still-plausible worlds.

    observe() is a caller-supplied I/O hook, NOT an explicit hidden-world truth
    dictionary exposed to the actor. No autonomous repair code is executed.
    """
    if not worlds or len(set(worlds)) != len(worlds) or set(repair_by_world) != set(worlds):
        raise ValueError("incomplete hypothesis family")
    if total_cost_budget < 0 or max_single_probe_risk < 0:
        raise ValueError("negative budget")
    if any(set(p.predictions) != set(worlds) for p in probes.values()):
        raise ValueError("probe models must cover every hypothesis")
    belief = frozenset(worlds)
    used = []
    cost = Fraction(0)

    def result(status, repair=None):
        return BlindResult(status, repair, tuple(sorted(belief)), tuple(used), cost)

    while True:
        authorities = {repair_by_world[w] for w in belief}
        if len(authorities) == 1:
            return result("AUTHORIZED", next(iter(authorities)))
        chosen = policy(belief)
        if chosen in ("stop", "reject", None):
            return result("REFUSED_CONFLICTING_REPAIRS")
        probe = probes.get(chosen)
        if probe is None or probe.risk > max_single_probe_risk or cost + probe.cost > total_cost_budget:
            return result("REFUSED_RISK_OR_BUDGET")
        if chosen in used:
            return result("REFUSED_REPEAT_PROBE")
        observation = observe(chosen)
        if not isinstance(observation, Fraction):
            raise ValueError("observation must be an exact Fraction in this finite toy model")
        posterior = frozenset(
            w for w in belief if abs(probe.predictions[w] - observation) <= probe.radius
        )
        used.append(chosen)
        cost += probe.cost
        if not posterior:
            return result("REFUSED_MODEL_MISSPECIFICATION")
        if posterior == belief:
            return result("REFUSED_UNINFORMATIVE_OBSERVATION")
        belief = posterior
