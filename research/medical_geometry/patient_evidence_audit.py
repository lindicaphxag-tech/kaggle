"""Paired patient-level ClinGeom audit against canonical-first strong control.

All metrics are ingested from actual model experiments (never generated here).
This module validates provenance, pairing, evaluation scope and honest outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ArmOutcome:
    patient_id: str
    domain: str
    arm: str
    mask_sha256: str
    source_checkpoint: str
    scope: str
    dice: float
    hd95_mm: float
    wall_seconds: float

    def __post_init__(self):
        if not all((self.patient_id, self.domain, self.arm,
                    self.mask_sha256, self.source_checkpoint)):
            raise ValueError("missing identity/provenance")
        if self.scope not in ("full_volume", "cropped_roi"):
            raise ValueError("unknown evaluation scope")
        if not 0 <= self.dice <= 1 or not isfinite(self.dice):
            raise ValueError("invalid dice")
        if not isfinite(self.hd95_mm) or self.hd95_mm < 0:
            raise ValueError("invalid/undefined HD95: declare missing data before scoring")
        if not isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("invalid runtime")


def paired_patient_audit(
    rows: list[ArmOutcome], *, baseline: str = "canonical_first",
    required_arms: tuple[str, ...] = ("canonical_first", "pre_mix_repair"),
) -> dict:
    if not rows or baseline not in required_arms or len(set(required_arms)) != len(required_arms):
        raise ValueError("invalid arm inventory")
    table = {}
    for r in rows:
        key = (r.patient_id, r.arm)
        if key in table:
            raise ValueError("duplicate patient/arm likely patch pseudo-replication")
        table[key] = r
    patients = sorted({r.patient_id for r in rows})
    if {r.arm for r in rows} != set(required_arms):
        raise ValueError("arm inventory changed")
    patient_rows = {}
    for pid in patients:
        groups = [table.get((pid, arm)) for arm in required_arms]
        if any(g is None for g in groups):
            raise ValueError("unpaired patient/arm")
        signatures = {(g.domain, g.mask_sha256, g.source_checkpoint, g.scope)
                      for g in groups}
        if len(signatures) != 1:
            raise ValueError("mask/checkpoint/scope mismatch across paired arms")
        patient_rows[pid] = {g.arm: g for g in groups}
    if len({p[baseline].scope for p in patient_rows.values()}) != 1:
        raise ValueError("mixed full-volume and cropped-ROI results cannot be pooled")
    report = {}
    for arm in required_arms:
        if arm == baseline:
            continue
        delta = {
            pid: (p[arm].dice - p[baseline].dice,
                  p[arm].hd95_mm - p[baseline].hd95_mm,
                  p[arm].wall_seconds / p[baseline].wall_seconds)
            for pid, p in patient_rows.items()
        }
        bydom = {}
        for domain in sorted({p[baseline].domain for p in patient_rows.values()}):
            ids = [pid for pid, p in patient_rows.items()
                   if p[baseline].domain == domain]
            bydom[domain] = sum(delta[pid][0] for pid in ids) / len(ids)
        report[arm] = {
            "mean_patient_dice_delta": sum(x[0] for x in delta.values()) / len(delta),
            "mean_patient_hd95_delta_mm": sum(x[1] for x in delta.values()) / len(delta),
            "mean_patient_runtime_ratio": sum(x[2] for x in delta.values()) / len(delta),
            "patients_dice_improved": sum(x[0] > 0 for x in delta.values()),
            "patients_dice_worsened": sum(x[0] < 0 for x in delta.values()),
            "domain_macro_dice_delta": sum(bydom.values()) / len(bydom),
            "domain_dice_delta": bydom,
        }
    return {
        "independent_unit": "patient",
        "scope": patient_rows[patients[0]][baseline].scope,
        "patient_count": len(patients),
        "baseline": baseline,
        "comparisons": report,
        "not_claimed": "no confidence interval or patient result generated",
    }
