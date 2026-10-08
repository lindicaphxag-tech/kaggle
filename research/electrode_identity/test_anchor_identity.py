import numpy as np
import pytest
from research.electrode_identity.anchor_identity import (
    anchor_fingerprint, _best_assignment, authorization,
    apply_authorized_permutation,
)


def signals(seed=11, c=8, a=12, t=1536):
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(a, t))
    weights = rng.normal(size=(c, a))
    sources = weights @ anchors + 0.02 * rng.normal(size=(c, t))
    return sources, anchors, weights


def test_two_independent_eeg_witnesses_authorize_true_swap():
    x, anchors, weights = signals()
    rng = np.random.default_rng(5)
    a2 = rng.normal(size=anchors.shape)
    x2 = weights @ a2 + 0.02 * rng.normal(size=x.shape)
    permutation = np.array([0, 4, 2, 3, 1, 7, 6, 5])
    result = authorization(x, anchors, x2, a2, x[permutation], anchors, x2[permutation], a2)
    assert result.status == "authorize_repair", result
    assert result.assignment == tuple(permutation)
    np.testing.assert_array_equal(apply_authorized_permutation(x[permutation], result), x)


def test_true_no_corruption_does_not_trigger_spurious_repair():
    x, anchors, weights = signals()
    rng = np.random.default_rng(7)
    a2 = rng.normal(size=anchors.shape)
    x2 = weights @ a2
    result = authorization(x, anchors, x2, a2, x, anchors, x2, a2)
    assert result.status == "retain_identity", result
    with pytest.raises(ValueError, match="no EEG electrode reindex"):
        apply_authorized_permutation(x, result)


def test_identical_anchors_signatures_force_nonidentifiability_refusal():
    x, anchors, _ = signals()
    x[0] = x[1]
    wrong = x[[1, 0, 2, 3, 4, 5, 6, 7]]
    result = authorization(x, anchors, x, anchors, wrong, anchors, wrong, anchors)
    assert result.status == "refuse"
    assert min(w.alternative_gap_per_channel for w in result.witnesses) < 1e-12


def test_conflicting_segments_and_bad_evidence_refuse():
    x, anchors, _ = signals()
    wrong = x[[1, 0, 2, 3, 4, 5, 6, 7]]
    result = authorization(x, anchors, x, anchors, wrong, anchors, x, anchors)
    assert result.status == "refuse"
    broken = x.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        anchor_fingerprint(broken, anchors)


def test_global_assignment_ambiguity_not_greedy_row_margin():
    witness = _best_assignment(np.ones((4, 8)), np.ones((4, 8)))
    assert witness.alternative_gap_per_channel == 0.0
