"""Exact minimum-cost static trusted evidence for patch-authority separation.

The core optimization is weighted set cover over observation-aliased pairs
with DIFFERENT permitted repairs. This is a familiar decision-region
identification problem, NOT a newly invented generic algorithm: see
Javdani et al., AISTATS 2014, and Cicalese et al., ICML 2014.

The new application concern is separately *verified provenance claims*
about code and the allowed patch boundary. Values here are stipulated to
come from a trusted issuer; implementation of issuer security is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Hashable, Mapping


@dataclass(frozen=True)
class EvidenceField:
    name: str
    cost: Fraction
    values: Mapping[str, Hashable]
    trusted: bool = True

    def __post_init__(self):
        if not self.name or self.cost < 0:
            raise ValueError("invalid evidence name/cost")
        if type(self.trusted) is not bool:
            raise ValueError("trust must be an explicit boolean")


def exact_minimum_provenance(
    *, repair_by_world: Mapping[str,str],
    aliased_pairs: list[tuple[str,str]],
    fields: tuple[EvidenceField,...],
    identify_every_world: bool=False,
) -> dict:
    """Minimize cumulative field cost subject to conflict-pair separation.

    The observed alias relation is supplied by a separate, frozen simulator
    or data evidence. For safety one should include EVERY pair compatible
    within the declared noise model. This procedure is not itself a
    statistical or a real-world soundness certificate.
    """
    worlds=set(repair_by_world)
    if not worlds or any(not w or not a for w,a in repair_by_world.items()):
        raise ValueError("invalid world/repair map")
    if len({f.name for f in fields})!=len(fields):
        raise ValueError("duplicate evidence field")
    if len(fields)>18:
        raise ValueError("exponential research oracle: at most 18 fields")
    alias=set()
    for x,y in aliased_pairs:
        if x not in worlds or y not in worlds or x==y:
            raise ValueError("invalid alias pair")
        alias.add(tuple(sorted((x,y))))
    conflicts={
        (a,b) for a,b in alias
        if identify_every_world or repair_by_world[a]!=repair_by_world[b]
    }
    for field in fields:
        if set(field.values)!=worlds:
            raise ValueError("missing provenance value on some worlds")
    candidate_fields=tuple(f for f in fields if f.trusted)
    separations={
        f.name:frozenset((a,b) for a,b in conflicts
                          if f.values[a]!=f.values[b])
        for f in candidate_fields
    }
    uncovered=conflicts-set().union(*separations.values()) if separations else set(conflicts)
    if uncovered:
        return {
            "feasible":False,
            "uncovered_conflicting_pairs":[list(p) for p in sorted(uncovered)],
            "conflict_count":len(conflicts),
            "reason":"no trusted field distinguishes some observationally aliased incompatible patches",
            "identified_every_world":identify_every_world,
        }
    best=None
    for size in range(len(candidate_fields)+1):
        for subset in combinations(candidate_fields,size):
            covered=set().union(*(separations[f.name] for f in subset)) if subset else set()
            if covered != conflicts:
                continue
            names=tuple(sorted(f.name for f in subset))
            cost=sum((f.cost for f in subset),Fraction(0))
            candidate=(cost,len(subset),names)
            if best is None or candidate < best:
                best=candidate
    assert best is not None
    return {
        "feasible":True,
        "selected_fields":list(best[2]),
        "cost_exact":str(best[0]),
        "conflict_count":len(conflicts),
        "observational_alias_pairs":len(alias),
        "identified_every_world":identify_every_world,
        "proof_scope":"exact finite weighted set cover over supplied aliases, under trusted field-value assumptions",
    }


def finite_decision_region_theorem(
    repair_by_world: Mapping[str,str],
    alias_pairs: list[tuple[str,str]],
) -> dict:
    """Certificate of zero-error, full-coverage sensor-only impossibility.

    One different-repair indistinguishable pair refutes any always-authorize
    sensor-only decision rule, even under unbounded compute or perfect planning.
    """
    for a,b in sorted(tuple(sorted(p)) for p in alias_pairs):
        if repair_by_world[a]!=repair_by_world[b]:
            return {
                "sensor_only_total_correct_authority_possible":False,
                "counterexample_pair":[a,b],
                "different_required_repairs":[repair_by_world[a],repair_by_world[b]],
                "reason":"same observations, contradictory required action",
            }
    return {"sensor_only_total_correct_authority_possible":True,
            "reason":"compatible observations never require conflicting actions"}


def native_mujoco_source_certificate() -> dict:
    from .mujoco_native_authority import WORLDS, rollout, signature, max_error, MEASUREMENT_HALF_WIDTH
    from .native_measured_blind_replay import HELD_EXCITATIONS
    from .open_catalog_source_gap import UNSEEN, unseen_rollout

    known=next(w for w in WORLDS if w.name=="gain_fault")
    foreign=next(w for w in UNSEEN if w.id=="foreign-identical-gain")
    # Two catalog revisions remain the SAME mechanical model and patch
    # authority, but have different (trusted) revision values.
    labels=("catalog-A","catalog-B","foreign-package")
    repairs={"catalog-A":known.authority, "catalog-B":known.authority,
             "foreign-package":foreign.physical.authority}
    aliases=[]
    sensors=("external","producer_to_controller_ctrl","transmission_force")
    for commands in HELD_EXCITATIONS:
        k=rollout(known,commands)
        x=unseen_rollout(foreign,commands)
        if any(max_error(signature(k[ch]),signature(x[ch]))>
               2*MEASUREMENT_HALF_WIDTH for ch in sensors):
            raise AssertionError("claimed MuJoCo observational equivalence false")
    aliases=list(combinations(labels,2))
    fields=(
        EvidenceField("issuer_namespace",Fraction(1),
            {"catalog-A":"catalog","catalog-B":"catalog",
             "foreign-package":"foreign"}),
        EvidenceField("exact_build_revision",Fraction(2),
            {"catalog-A":"rev-A","catalog-B":"rev-B",
             "foreign-package":"different-origin"}),
        EvidenceField("unsigned_robot_self_report",Fraction(0),
            {"catalog-A":"catalog","catalog-B":"catalog",
             "foreign-package":"foreign"},trusted=False),
    )
    base=finite_decision_region_theorem(repairs,aliases)
    minimum=exact_minimum_provenance(
        repair_by_world=repairs,aliased_pairs=aliases,fields=fields)
    identity=exact_minimum_provenance(
        repair_by_world=repairs,aliased_pairs=aliases,fields=fields,
        identify_every_world=True)
    if not (minimum["cost_exact"]=="1"
            and identity["cost_exact"]=="2"
            and minimum["selected_fields"]==["issuer_namespace"]
            and not base["sensor_only_total_correct_authority_possible"]):
        raise AssertionError("native MuJoCo provenance separation certificate changed")
    return {
        "protocol":"native-MuJoCo-authority-minimum-trusted-provenance-v1",
        "observed_equivalence_check":"three native MuJoCo forcing traces and three sensor channels",
        "physical_sample_units":"single model with two declared catalog revisions plus one identical foreign package",
        "sensor_only":base,
        "minimum_authority_evidence":minimum,
        "full_identity_evidence":identity,
        "claimed_cost_advantage_scope":"finite hand-specified static evidence prices and known conflicting repair classes",
        "related_work":"AISTATS 2014 decision-region determination and ICML 2014 diagnosis determination already optimize such query classes",
        "not_claimed":"new combinatorial algorithm, real certified issuer, independent robot deployments or general physical safety",
    }


if __name__=="__main__":
    import json
    print(json.dumps(native_mujoco_source_certificate(),indent=2,sort_keys=True))
