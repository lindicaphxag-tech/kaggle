#!/usr/bin/env python3
"""Pathwise consumable-risk budget for exact hidden repair authority.

The older noisy_authority_game.maximum_probe_risk option bounds EACH probe
independently. It must not be interpreted as an accumulated safety/exposure
budget across the adaptive sequence. This module explicitly adds that
distinct constraint, with a belief-set+remaining-budget Bellman state.

No claim that a fixed risk score equals physical collision probability;
the abstract additive nonnegative exposure must be calibrated separately.
"""
from __future__ import annotations

from fractions import Fraction as F
from functools import lru_cache
import json

from noisy_authority_game import Probe, compatible_children, minimax_authority_cost


def cumulative_risk_minimax(
    names: tuple[str, ...],
    authority: dict[str, str],
    probes: tuple[Probe, ...],
    total_risk_budget: F,
) -> tuple[F | None, dict]:
    if total_risk_budget < 0:
        raise ValueError("negative cumulative risk budget")
    if not names or len(names) != len(set(names)) or set(names) != set(authority):
        raise ValueError("invalid hypothesis/authority domains")
    if len({p.name for p in probes}) != len(probes):
        raise ValueError("duplicate probe names would make policy certificates ambiguous")
    for p in probes:
        if {h for h, _ in p.center_by_world} != set(names):
            raise ValueError("incomplete experiment table")

    choices: dict[tuple[frozenset[str], F], tuple[str, tuple[frozenset[str], ...], F | None]] = {}

    @lru_cache(None)
    def solve(state: frozenset[str], remaining: F) -> F | None:
        if len({authority[h] for h in state}) == 1:
            choices[(state, remaining)] = ("STOP", (), F(0))
            return F(0)
        candidates = []
        for p in probes:
            if p.risk > remaining:
                continue
            children = compatible_children(state, p)
            if state in children:
                continue
            downstream = [solve(child, remaining - p.risk) for child in children]
            if any(v is None for v in downstream):
                continue
            worst = p.cost + max(downstream)
            candidates.append((worst, p.name, children))
        if not candidates:
            choices[(state, remaining)] = ("REJECT", (), None)
            return None
        val, name, children = min(candidates, key=lambda v: (v[0], v[1]))
        choices[(state, remaining)] = (name, children, val)
        return val

    root=(frozenset(names),total_risk_budget)
    value=solve(*root)
    visited=set()
    byname={p.name:p for p in probes}
    def verify(key):
        if key in visited:
            return choices[key][2]
        visited.add(key)
        state, remaining=key
        name,children,val=choices[key]
        if name=="STOP":
            assert len({authority[h] for h in state})==1 and val==0
            return F(0)
        if name=="REJECT":
            assert val is None
            return None
        p=byname[name]
        assert p.risk <= remaining
        assert children == compatible_children(state,p)
        assert state not in children
        next_budget=remaining-p.risk
        results=[verify((child,next_budget)) for child in children]
        assert all(v is not None for v in results)
        assert val == p.cost+max(results)
        return val
    assert verify(root)==value
    cert=[{
        "survivors":sorted(state),
        "remaining_risk_exact":str(remaining),
        "chosen_action":name,
        "successor_sets":[sorted(s) for s in children],
        "remaining_risk_after_probe":(
            str(remaining-byname[name].risk)
            if name not in ("STOP","REJECT") else None
        ),
        "worst_case_cost_exact":str(val) if val is not None else None,
    } for (state,remaining),(name,children,val) in sorted(
        choices.items(), key=lambda kv:(len(kv[0][0]),tuple(sorted(kv[0][0])),kv[0][1])
    )]
    return value, {
        "schema":"semrepair-cumulative-risk-v1",
        "initial_total_risk_exact":str(total_risk_budget),
        "policy":cert,
        "verified_states":len(visited),
        "scope":"Finite bounded-adversarial scalar observation, deterministic additive pathwise exposure.",
    }


def self_test() -> dict:
    names=("h0","h1","h2","h3")
    probe_a=Probe("A",tuple(zip(names,(F(0),F(3,2),F(3),F(9,2)))),F(1),cost=F(1),risk=F(1,4))
    probe_b=Probe("B",tuple(zip(names,(F(3,2),F(9,2),F(0),F(3)))),F(1),cost=F(1),risk=F(1,4))
    authorities={w:f"repair:{w}" for w in names}

    # A cap for each probe, applied afresh at every node, admits two risky
    # probes even though their combined consumption exceeds the cap.
    local_cost,_=minimax_authority_cost(names,authorities,(probe_a,probe_b),maximum_probe_risk=F(3,8))
    assert local_cost == F(2)

    # Under a genuine cumulative path budget 3/8, A+B cannot both be used.
    blocked,blocked_certificate=cumulative_risk_minimax(names,authorities,(probe_a,probe_b),F(3,8))
    assert blocked is None
    assert blocked_certificate["policy"][0]["chosen_action"] in ("REJECT","STOP") or blocked_certificate["verified_states"] > 0

    feasible,cert=cumulative_risk_minimax(names,authorities,(probe_a,probe_b),F(1,2))
    assert feasible == F(2)
    assert cert["verified_states"] >= 3

    same={w:"same-exact-implementation" for w in names}
    stop,proof=cumulative_risk_minimax(names,same,(probe_a,probe_b),F(0))
    assert stop==0 and proof["verified_states"]==1

    blocked_zero,_=cumulative_risk_minimax(names,authorities,(probe_a,probe_b),F(0))
    assert blocked_zero is None
    try:
        cumulative_risk_minimax(names,authorities,(probe_a,probe_b),F(-1,4))
    except ValueError:
        pass
    else:
        raise AssertionError("negative budget accepted")

    return {
        "status":"pass",
        "local_per_probe_cap_3_8_cost_exact":str(local_cost),
        "cumulative_cap_3_8":"unsafe_unidentifiable",
        "cumulative_cap_1_2_cost_exact":str(feasible),
        "zero_risk_same_authority_cost":"0",
        "zero_risk_conflicting_authority":"unsafe_unidentifiable",
        "risk_semantics":"deterministic nonnegative additive exposure, worst-case per-path",
        "claim_boundary":"Not a calibrated probability of collision or formal real-robot safety guarantee.",
        "feasible_policy_certificate":cert,
    }


if __name__=="__main__":
    print(json.dumps(self_test(),indent=2,sort_keys=True))
