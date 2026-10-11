"""Native MuJoCo counterexample: untrusted actuator-force tap can swap repair identity.

The physical world is genuinely stepped by MuJoCo. Only its deep force readout
is substituted with another model's values; exterior and controller-interface
observations remain authentic and indistinguishable. Never mistake this
adversarial channel-integrity attack for bounded independent sensor noise.

Trust provenance here is an explicit external assumption, NOT a cryptographic
signature or evidence that an actual field sensor is authenticated.
"""
from __future__ import annotations
from dataclasses import asdict,dataclass
import json

from .mujoco_native_authority import (
    WORLDS,rollout,signature,max_error,MEASUREMENT_HALF_WIDTH
)
from .native_measured_blind_replay import HELD_EXCITATIONS

CHANNELS=("external","producer_to_controller_ctrl","transmission_force")
ALIAS_PAIRS=(("gain_fault","nuisance_alias"),
             ("nuisance_alias","gain_fault"))
MATCH_TOL=2*MEASUREMENT_HALF_WIDTH


@dataclass(frozen=True)
class ProbeOutcome:
    truth_world: str
    substitute_world: str | None
    wave: int
    assumption: str
    answer: str
    true_authority: str
    observed_force_authentic: bool
    readout_steps: tuple[str,...]


def decide_with_source_assumption(
    world, commands, *, force_source_trusted: bool,
    spoof_force_from=None
) -> ProbeOutcome:
    """Use only observed taps, never truth-world ID in the actor decision."""
    traces={w.name:rollout(w,commands) for w in WORLDS}
    readouts=traces[world.name]
    hypotheses=frozenset(w.name for w in WORLDS)
    authority={w.name:w.authority for w in WORLDS}
    queried=[]
    for channel in CHANNELS:
        if len({authority[w] for w in hypotheses})==1:
            break
        if channel=="transmission_force" and not force_source_trusted:
            return ProbeOutcome(
                world.name,None,-1,"untrusted_force_refusal","REFUSE",
                authority[world.name],False,tuple(queried))
        actual=signature(
            traces[spoof_force_from.name][channel]
            if channel=="transmission_force" and spoof_force_from is not None
            else readouts[channel])
        survivors=frozenset(
            w for w in hypotheses
            if max_error(signature(traces[w][channel]),actual)<=MATCH_TOL)
        queried.append(channel)
        if not survivors:
            return ProbeOutcome(
                world.name,spoof_force_from.name if spoof_force_from else None,
                -1,"unmatched_observation","REFUSE",
                authority[world.name],spoof_force_from is None,tuple(queried))
        hypotheses=survivors
    answers={authority[w] for w in hypotheses}
    selected=next(iter(answers)) if len(answers)==1 else "REFUSE"
    return ProbeOutcome(
        world.name,spoof_force_from.name if spoof_force_from else None,-1,
        "trusted_force" if force_source_trusted else "no_force_required",
        selected,authority[world.name],spoof_force_from is None,tuple(queried))


def experiment() -> dict:
    byname={w.name:w for w in WORLDS}
    rows=[]
    for index,commands in enumerate(HELD_EXCITATIONS):
        for truth,attacker in ALIAS_PAIRS:
            a,b=byname[truth],byname[attacker]
            traces_a=rollout(a,commands)
            traces_b=rollout(b,commands)
            for channel in CHANNELS[:2]:
                if max_error(signature(traces_a[channel]),
                             signature(traces_b[channel]))>MATCH_TOL:
                    raise AssertionError("historically aliased upstream observation changed")
            if max_error(signature(traces_a["transmission_force"]),
                         signature(traces_b["transmission_force"]))<=MATCH_TOL:
                raise AssertionError("forged force signatures not actually distinct")
            authentic=decide_with_source_assumption(
                a,commands,force_source_trusted=True)
            spoofed=decide_with_source_assumption(
                a,commands,force_source_trusted=True,spoof_force_from=b)
            fail_closed=decide_with_source_assumption(
                a,commands,force_source_trusted=False,spoof_force_from=b)
            if authentic.answer!=a.authority:
                raise AssertionError("unmodified authenticated tap should identify correct authority")
            if spoofed.answer!=b.authority or spoofed.answer==a.authority:
                raise AssertionError("force spoof did not cause incorrect authority")
            if fail_closed.answer!="REFUSE":
                raise AssertionError("untrusted critical tap was incorrectly authorized")
            rows.append({
                "wave":index,"truth_world":truth,"spoofed_world":attacker,
                "correct_authority":a.authority,
                "assumed_trust_authentic_answer":authentic.answer,
                "untrusted_tap_spoof_naive_answer":spoofed.answer,
                "declared_untrusted_source_gate":fail_closed.answer,
                "matched_external_and_controller_observations":True,
                "force_measurements_substituted_only":True,
            })
    return {
        "protocol":"semrepair-mujoco-tap-integrity-attack-v1",
        "physical_worlds":len(WORLDS),
        "held_excitation_regimes":len(HELD_EXCITATIONS),
        "attack_cells":len(rows),
        "naive_wrong_authorizations":sum(
            r["untrusted_tap_spoof_naive_answer"]!=r["correct_authority"] for r in rows),
        "source_untrusted_refusals":sum(
            r["declared_untrusted_source_gate"]=="REFUSE" for r in rows),
        "authentic_tap_correct":sum(
            r["assumed_trust_authentic_answer"]==r["correct_authority"] for r in rows),
        "cases":rows,
        "limits":"forged MuJoCo tap, NOT real network adversary; source trust stipulated not verified; refusing loses availability, not proof of robustness to arbitrary compromise",
    }

if __name__=="__main__":
    print(json.dumps(experiment(),indent=2,sort_keys=True))
