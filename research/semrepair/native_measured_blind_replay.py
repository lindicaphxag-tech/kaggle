"""Actual MuJoCo closed-world test with unseen development waveforms.

Model signatures are generated using the exact same four known candidate
simulator worlds. This does not establish out-of-family generalization.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from fractions import Fraction

from .mujoco_native_authority import (
    WORLDS, rollout, corrected, signature, max_error,
    MEASUREMENT_HALF_WIDTH, EFFECT_TOL,
)

HELD_EXCITATIONS = (
    (0.31,-0.83,0.47,-0.19,0.71,0.02,-0.42),
    (-0.62,0.26,0.53,-0.47,0.11,0.88,-0.08),
    (0.12,0.65,-0.39,-0.74,0.58,-0.16,0.29),
)
CHANNELS = (
    ("external",Fraction(1),Fraction(0)),
    ("producer_to_controller_ctrl",Fraction(2),Fraction(1,4)),
    ("transmission_force",Fraction(3),Fraction(3,4)),
)


@dataclass(frozen=True)
class Trial:
    world: str
    wave: int
    authorization: str
    correct_authorization: bool
    cost: str
    calls: tuple[str, ...]
    physical_closure_max_error: float | None


def _score_one(world, wave, commands, *, max_probe_risk, external_only=False):
    templates = {w.name:rollout(w,commands) for w in WORLDS}
    measurement = rollout(world,commands)
    authority = {w.name:w.authority for w in WORLDS}
    survivors = frozenset(authority)
    calls = []
    spent = Fraction(0)
    probes = CHANNELS[:1] if external_only else CHANNELS
    for channel,cost,risk in probes:
        if len({authority[w] for w in survivors}) == 1:
            break
        if risk > max_probe_risk:
            continue
        expected = {w:signature(templates[w][channel]) for w in survivors}
        exemplar = next(iter(expected.values()))
        if all(max_error(v,exemplar)<=2*MEASUREMENT_HALF_WIDTH
               for v in expected.values()):
            continue
        reading = signature(measurement[channel])
        remaining = frozenset(
            w for w in survivors
            if max_error(expected[w],reading)<=2*MEASUREMENT_HALF_WIDTH
        )
        calls.append(channel)
        spent += cost
        if not remaining:
            raise AssertionError("observation outside frozen candidate model")
        survivors = remaining
    authorized = {authority[w] for w in survivors}
    answer = next(iter(authorized)) if len(authorized)==1 else "REFUSE"
    correct = answer == authority[world.name]
    if answer != "REFUSE" and not correct:
        raise AssertionError("incorrect authority")
    closure = None
    if answer != "REFUSE":
        reference = next(w for w in WORLDS if w.name == "reference")
        a = rollout(reference,commands)["external"]
        b = rollout(corrected(world),commands)["external"]
        closure = max_error(a,b)
        if closure > EFFECT_TOL:
            raise AssertionError("physical correction failed")
    return Trial(world.name,wave,answer,correct,str(spent),tuple(calls),closure)


def study():
    variants = {"full_tap":[],"external_only":[],"risk_without_force_tap":[]}
    for i,commands in enumerate(HELD_EXCITATIONS):
        for world in WORLDS:
            variants["full_tap"].append(_score_one(
                world,i,commands,max_probe_risk=Fraction(1)))
            variants["external_only"].append(_score_one(
                world,i,commands,max_probe_risk=Fraction(0),external_only=True))
            variants["risk_without_force_tap"].append(_score_one(
                world,i,commands,max_probe_risk=Fraction(1,2)))
    def summary(trials):
        return {
            "authorized":sum(x.authorization!="REFUSE" for x in trials),
            "correct_authorizations":sum(x.correct_authorization for x in trials),
            "wrong_authorizations":sum(x.authorization!="REFUSE" and
                                       not x.correct_authorization for x in trials),
            "refusals":sum(x.authorization=="REFUSE" for x in trials),
            "max_probe_cost":str(max(Fraction(x.cost) for x in trials)),
        }
    stats = {name:summary(rows) for name,rows in variants.items()}
    if stats["full_tap"]["authorized"]!=len(HELD_EXCITATIONS)*len(WORLDS):
        raise AssertionError("candidate signatures remain ambiguous")
    return {
        "protocol":"closed_world_native_mujoco_held_excitation_v1",
        "measured_world_excitation_cells":len(variants["full_tap"]),
        "full_tap":stats["full_tap"],
        "external_only":stats["external_only"],
        "risk_without_force_tap":stats["risk_without_force_tap"],
        "per_case":[asdict(x) for x in variants["full_tap"]],
        "independent_population_samples_claimed":False,
        "limitation":"known four MuJoCo worlds; model-based candidate signatures; no out-of-family policy task validation",
    }


if __name__=="__main__":
    print(json.dumps(study(),indent=2,sort_keys=True))
