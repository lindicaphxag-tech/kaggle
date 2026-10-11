"""Open-catalog native MuJoCo selective safety vs authorized coverage.

Cohort: fixed 4 known + 4 out-of-catalog worlds, on 3 held excitations.
Rates refer to 24 *dependent world-waveform cells*. No binomial
patient/device confidence intervals or claims of out-of-vendor generalization.

The source-aware controller has an honest protected issuer as an assumption;
sensor-only cannot distinguish sensor-identical software owner aliases.
"""
from __future__ import annotations

from .mujoco_native_authority import WORLDS,rollout
from .native_measured_blind_replay import HELD_EXCITATIONS
from .open_catalog_source_gap import (
    UNSEEN,unseen_rollout,decide_from_sensor_traces,
    known_source_digest,_source_digest,
)


def audit_open_catalog_risk_coverage() -> dict:
    records=[]
    for wave,commands in enumerate(HELD_EXCITATIONS):
        templates={w.name:rollout(w,commands) for w in WORLDS}
        for known in WORLDS:
            obs=templates[known.name]
            for rule in ("sensor_only","trusted_source_required"):
                result=decide_from_sensor_traces(
                    measured=obs,templates=templates,
                    source_digest=(known_source_digest(known)
                                   if rule=="trusted_source_required" else None),
                    enforce_source_provenance=(rule=="trusted_source_required"))
                records.append({
                    "family":"catalog",
                    "true_source":known.name,"wave":wave,
                    "rule":rule,"actual_repair":known.authority,
                    "decision":result["answer"],
                })
        for unknown in UNSEEN:
            obs=unseen_rollout(unknown,commands)
            true_auth=(unknown.physical.authority if unknown.physical
                       else "unknown-servo:requires-external-source-review")
            for rule in ("sensor_only","trusted_source_required"):
                result=decide_from_sensor_traces(
                    measured=obs,templates=templates,
                    source_digest=(_source_digest(unknown.source_package)
                                   if rule=="trusted_source_required" else None),
                    enforce_source_provenance=(rule=="trusted_source_required"))
                records.append({
                    "family":"unknown",
                    "true_source":unknown.id,"wave":wave,
                    "rule":rule,"actual_repair":true_auth,
                    "decision":result["answer"],
                })
    measures={}
    for rule in ("sensor_only","trusted_source_required"):
        selected=[r for r in records if r["rule"]==rule]
        correct=sum(r["decision"]!="REFUSE" and
                    r["decision"]==r["actual_repair"] for r in selected)
        wrong=sum(r["decision"]!="REFUSE" and
                  r["decision"]!=r["actual_repair"] for r in selected)
        refused=sum(r["decision"]=="REFUSE" for r in selected)
        n=len(selected)
        measures[rule]={
            "trial_cells":n,
            "correct_authorizations":correct,
            "wrong_authorizations":wrong,
            "refusals":refused,
            "authorized_coverage":(correct+wrong)/n,
            "wrong_repair_risk_given_authorization":(
                wrong/(correct+wrong) if correct+wrong else None),
            "unconditional_wrong_repair_fraction":wrong/n,
        }
    if measures["sensor_only"]!={
        "trial_cells":24,"correct_authorizations":12,"wrong_authorizations":3,
        "refusals":9,"authorized_coverage":15/24,
        "wrong_repair_risk_given_authorization":3/15,
        "unconditional_wrong_repair_fraction":3/24
    }:
        raise AssertionError("sensor-only physical open-catalog result drift")
    if measures["trusted_source_required"]!={
        "trial_cells":24,"correct_authorizations":12,"wrong_authorizations":0,
        "refusals":12,"authorized_coverage":12/24,
        "wrong_repair_risk_given_authorization":0.,
        "unconditional_wrong_repair_fraction":0.
    }:
        raise AssertionError("trusted source physical open-catalog result drift")
    return {
        "protocol":"fixed-uniform-open-catalog-risk-coverage-24-conditions",
        "simulator":"native MuJoCo dynamics, four known plus four out-of-catalog worlds",
        "independent_hardware_count":0,
        "source_provenance":"honest issuer and nonforgeable source digest ASSUMED; not independently verified",
        "score":measures,
        "known_failure_mechanism":"sensor-identical source packages require different legal repair actions",
        "claim":"tradeoff: 3 wrong authorizations eliminated, but authorized coverage falls 15/24 to 12/24",
        "limitation":"deterministic author-selected worlds/waveforms; no population inference, no task success, not a state-of-the-art robotics comparison",
    }


if __name__=="__main__":
    import json
    print(json.dumps(audit_open_catalog_risk_coverage(),indent=2,sort_keys=True))
