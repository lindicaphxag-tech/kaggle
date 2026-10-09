"""Patient-cluster exact sign summary for ClinGeom paired treatment effects.

Descriptive/exploratory audit only. A sign test makes an independence and
symmetric-null assumption, tests direction not effect magnitude, and cannot
prove general clinical utility or independence of prior model selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import comb, isfinite
from statistics import median


@dataclass(frozen=True)
class PairedPatientDelta:
    patient: str
    delta_dice: float

    def __post_init__(self):
        if not self.patient or not isfinite(self.delta_dice) or not -1 <= self.delta_dice <= 1:
            raise ValueError("invalid patient Dice effect")


def exact_exploratory_sign_summary(
    deltas: list[PairedPatientDelta], *, tie_tolerance: float = 0,
) -> dict:
    """Exact two-sided sign p-value among non-tied independent patients."""
    if not deltas or len({d.patient for d in deltas}) != len(deltas):
        raise ValueError("empty cohort or patient pseudoreplication")
    if not 0 <= tie_tolerance < 1 or not isfinite(tie_tolerance):
        raise ValueError("invalid tie tolerance")
    vals = [r.delta_dice for r in deltas]
    pos = sum(v > tie_tolerance for v in vals)
    neg = sum(v < -tie_tolerance for v in vals)
    n = pos + neg
    tail = min(pos, neg)
    two_sided = (
        min(1., 2 * sum(comb(n, k) for k in range(tail + 1)) / (2 ** n))
        if n else 1.
    )
    return {
        "independent_unit": "patient",
        "patient_count": len(deltas),
        "positive": pos, "negative": neg, "ties": len(vals)-n,
        "mean_dice_delta": sum(vals)/len(vals),
        "median_dice_delta": median(vals),
        "two_sided_exploratory_sign_p": two_sided,
        "assumption": "independent symmetric signs conditional on non-ties; no multiplicity or selection correction",
        "not_claimed": "not confirmatory evidence of efficacy or clinical utility",
    }
