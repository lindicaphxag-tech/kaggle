"""Validation-only matched-coverage abstention baseline for EEG identity certificates.

Every threshold is chosen using validation-subject margins and the validation-
subject coverage of an identity-invariance certificate. Test labels and test
certificate coverage NEVER enter calibration. Ties are deliberately exposed.
This is a standard selective-prediction baseline, not a novel method.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ThresholdFit:
    threshold: float
    requested_validation_coverage: float
    achieved_validation_coverage: float
    n_validation: int
    ties_prevent_exact_coverage: bool


def fit_validation_margin_threshold(validation_margins, validation_certified):
    scores=np.asarray(validation_margins,dtype=float)
    accepted=np.asarray(validation_certified,dtype=bool)
    if scores.ndim!=1 or scores.size==0 or accepted.shape!=scores.shape:
        raise ValueError("One finite validation margin per validation certificate required")
    if not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError("Margins must be finite and nonnegative")
    n=len(scores)
    n_target=int(accepted.sum())
    requested=n_target/n
    if n_target==0:
        threshold=float("inf")
    elif n_target==n:
        threshold=0.0
    else:
        sorted_desc=np.sort(scores)[::-1]
        upper=float(sorted_desc[n_target-1])
        lower=float(sorted_desc[n_target])
        threshold=(upper+lower)/2 if upper>lower else upper
    observed=float(np.mean(scores>=threshold))
    return ThresholdFit(threshold,requested,observed,n,abs(observed-requested)>1e-12)


def audit_test_predictions(test_margins, test_predictions, test_labels, threshold):
    scores=np.asarray(test_margins,dtype=float)
    pred=np.asarray(test_predictions)
    y=np.asarray(test_labels)
    if scores.ndim!=1 or not scores.size or pred.shape!=scores.shape or y.shape!=scores.shape:
        raise ValueError("Invalid test vector sizes")
    if not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError("Test margins must be nonnegative finite")
    if threshold<0:
        raise ValueError("Invalid threshold")
    accepted=scores>=threshold
    coverage=float(np.mean(accepted))
    acc=float(np.mean(pred[accepted]==y[accepted])) if accepted.any() else None
    return {"coverage":coverage, "accuracy_on_accepted":acc,
            "n_test":int(len(scores)), "n_accepted":int(accepted.sum()),
            "n_rejected":int((~accepted).sum()), "accepted":accepted.tolist()}
