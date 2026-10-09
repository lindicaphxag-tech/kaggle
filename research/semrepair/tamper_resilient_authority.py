"""Finite repair-authority channel selection under bounded evidence substitution.

Classical Hamming-distance correctness is applied to a *concrete repair-class*
decision. This is NOT new coding theory or actual source authentication.
Every channel has one fixed categorical reading. At most f independently
delivered channels can be replaced. Correlated compromises, open-world models,
continuous observation errors and execution-time patch safety are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Hashable, Mapping, Sequence


@dataclass(frozen=True)
class Channel:
    name: str
    cost: Fraction
    signatures: Mapping[str, Hashable]
    eligible: bool = True


def _validate(authority: Mapping[str, str], channels: Sequence[Channel], f: int) -> tuple[str, ...]:
    if type(f) is not int or f < 0:
        raise ValueError("corruption budget must be a nonnegative integer")
    worlds = tuple(sorted(authority))
    if not worlds or any(not isinstance(w, str) or not w or not isinstance(a, str) or not a
                         for w, a in authority.items()):
        raise ValueError("invalid world/authority")
    if len(channels) > 18 or len({c.name for c in channels}) != len(channels):
        raise ValueError("at most 18 uniquely named channels")
    for channel in channels:
        if not isinstance(channel.name, str) or not channel.name:
            raise ValueError("empty channel name")
        if not isinstance(channel.cost, Fraction) or channel.cost < 0:
            raise ValueError("channel cost must be a nonnegative exact Fraction")
        if type(channel.eligible) is not bool or set(channel.signatures) != set(worlds):
            raise ValueError("invalid eligibility or incomplete expected signatures")
        for v in channel.signatures.values():
            try:
                hash(v)
            except TypeError as e:
                raise ValueError("channel symbols must be hashable") from e
    return worlds


def authority_distance(authority: Mapping[str, str], channels: Sequence[Channel]) -> dict[tuple[str, str], int]:
    """Different-authority Hamming distances; identical-authority aliases are OK."""
    worlds = tuple(sorted(authority))
    return {(a, b): sum(c.signatures[a] != c.signatures[b] for c in channels)
            for a, b in combinations(worlds, 2) if authority[a] != authority[b]}


def exact_robust_channel_design(
    *, authority: Mapping[str, str], channels: Sequence[Channel],
    corruption_budget: int,
) -> dict:
    """Minimum exact cost subset with cross-authority distance >= 2f+1."""
    _validate(authority, channels, corruption_budget)
    eligible = tuple(sorted((c for c in channels if c.eligible), key=lambda c: c.name))
    conflict_pairs = tuple(authority_distance(authority, eligible))
    threshold = 2 * corruption_budget + 1
    unresolvable = [list(p) for p, dist in authority_distance(authority, eligible).items()
                    if dist < threshold]
    if unresolvable:
        return {"feasible": False, "corruption_budget": corruption_budget,
                "required_cross_authority_distance": threshold,
                "unresolvable_pairs": unresolvable,
                "reason": "cross-authority signatures have overlapping f-corruption Hamming balls"}
    best = None
    for count in range(len(eligible) + 1):
        for subset in combinations(eligible, count):
            distances = authority_distance(authority, subset)
            if any(d < threshold for d in distances.values()):
                continue
            candidate = (sum((c.cost for c in subset), Fraction(0)),
                         len(subset), tuple(c.name for c in subset), subset)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    _, _, names, selected = best
    distances = authority_distance(authority, selected)
    return {"feasible": True, "selected_channels": list(names),
            "cost_exact": str(best[0]),
            "corruption_budget": corruption_budget,
            "required_cross_authority_distance": threshold,
            "minimum_cross_authority_distance": min(distances.values()) if distances else None,
            "conflicting_world_pairs": len(conflict_pairs),
            "claim_scope": "finite categorical independent channel-substitution model; no cryptographic trust guarantee"}


def selective_authorize(*, observed: Mapping[str, Hashable],
                        authority: Mapping[str, str], channels: Sequence[Channel],
                        corruption_budget: int) -> dict:
    """Authorize only when ALL models within <=f substitutions agree on repair."""
    _validate(authority, channels, corruption_budget)
    if any(not c.eligible for c in channels):
        raise ValueError("ineligible evidence channel selected")
    if set(observed) != {c.name for c in channels}:
        raise ValueError("observation and selected channel names differ")
    plausible = sorted(w for w in authority if sum(
        observed[c.name] != c.signatures[w] for c in channels
    ) <= corruption_budget)
    repairs = sorted({authority[w] for w in plausible})
    if not plausible:
        decision, repair = "REFUSE_OUTSIDE_ASSUMPTIONS", None
    elif len(repairs) != 1:
        decision, repair = "REFUSE_AMBIGUOUS_AUTHORITY", None
    else:
        decision, repair = "AUTHORIZE", repairs[0]
    return {"decision": decision, "repair": repair,
            "compatible_worlds": plausible, "corruption_budget": corruption_budget}


def exact_binary_flip_assay() -> dict:
    """Eight DEPENDENT toy observations: two worlds times <=one of three flips."""
    worlds = {"legal-build": "repair-A", "foreign-build": "repair-B"}
    channels = tuple(Channel(f"independent-issuer-{i}", Fraction(1),
                             {"legal-build": "A", "foreign-build": "B"})
                     for i in range(1, 4))
    minimum = exact_robust_channel_design(
        authority=worlds, channels=channels, corruption_budget=1)
    assert minimum["selected_channels"] == [c.name for c in channels]
    full = {"correct": 0, "wrong": 0, "refused": 0}
    naive = {"correct": 0, "wrong": 0, "refused": 0}
    two = {"correct": 0, "wrong": 0, "refused": 0}
    for world in worlds:
        initial = "A" if world == "legal-build" else "B"
        for tampered_index in (None, 0, 1, 2):
            obs = {c.name: ("B" if initial == "A" else "A")
                   if i == tampered_index else initial
                   for i, c in enumerate(channels)}
            truth = worlds[world]
            for selected, counts in ((channels, full), (channels[:2], two)):
                response = selective_authorize(
                    observed={c.name: obs[c.name] for c in selected},
                    authority=worlds, channels=selected, corruption_budget=1)
                key = ("refused" if response["decision"] != "AUTHORIZE" else
                       "correct" if response["repair"] == truth else "wrong")
                counts[key] += 1
            guess = "repair-A" if obs[channels[0].name] == "A" else "repair-B"
            naive["correct" if guess == truth else "wrong"] += 1
    assert full == {"correct": 8, "wrong": 0, "refused": 0}
    assert naive == {"correct": 6, "wrong": 2, "refused": 0}
    assert two == {"correct": 4, "wrong": 0, "refused": 4}
    return {"fixture": "two repair classes / three binary channels / at most one substituted channel",
            "dependent_constructed_cases": 8, "robust_3_channel": full,
            "unsafe_1_channel_guess": naive, "selective_2_channel": two,
            "design": minimum,
            "limitation": "finite digital fixture only; not real authenticated issuers or robot task success"}
