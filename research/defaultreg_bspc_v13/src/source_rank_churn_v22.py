"""V22: exact pairwise ROC-AUC rank-transition mechanism.

This is an *audit*, not a predictor of unseen-subject harm or a claimed
novel learning algorithm. All selection-time inputs must belong to held
SOURCE validation subjects. A target-subject audit is post-outcome only.

No confidence interval follows from the exact identity: the shared
source-training pools make naïve independent-subject tests inappropriate.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _validated_arrays(labels: Any, before: Any, after: Any):
    y = np.asarray(labels)
    a = np.asarray(before, dtype=np.float64)
    b = np.asarray(after, dtype=np.float64)
    if y.ndim != 1 or a.ndim != 1 or b.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if not (len(y) == len(a) == len(b)) or len(y) < 2:
        raise ValueError("labels and scores must share a nontrivial length")
    if not np.isin(y, [0, 1]).all() or not {0, 1}.issubset(set(y.tolist())):
        raise ValueError("labels must contain both classes 0 and 1")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("scores must be finite")
    return y, a, b


def pairwise_rank_transition(
    labels: Any, score_before: Any, score_after: Any, *, block_size: int = 128
) -> dict[str, Any]:
    """Return the exact 3x3 pair-credit transition and ROC-AUC decomposition.

    Each positive/negative pair has a sign state: negative (-1), tie (0),
    positive (+1), contributing AUC credit [0, .5, 1]. The 3x3 matrix
    counts pair transitions, *including ties*, with rows=before and
    columns=after. Calculations are chunked, not quadratic-memory.
    """
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    y, a, b = _validated_arrays(labels, score_before, score_after)
    plus_a, minus_a = a[y == 1], a[y == 0]
    plus_b, minus_b = b[y == 1], b[y == 0]
    pairs = int(len(plus_a) * len(minus_a))
    counts = np.zeros((3, 3), dtype=np.int64)

    for lo in range(0, len(plus_a), block_size):
        hi = lo + block_size
        # States -1/0/+1 are encoded as 0/1/2 without an arbitrary
        # floating-point tie threshold.
        old = np.sign(plus_a[lo:hi, None] - minus_a[None, :]).astype(np.int8) + 1
        new = np.sign(plus_b[lo:hi, None] - minus_b[None, :]).astype(np.int8) + 1
        counts += np.bincount(
            (3 * old.astype(np.int64) + new.astype(np.int64)).ravel(),
            minlength=9,
        ).reshape(3, 3)

    credits = np.asarray([0.0, 0.5, 1.0])
    before_auc = float(np.dot(counts.sum(axis=1), credits) / pairs)
    after_auc = float(np.dot(counts.sum(axis=0), credits) / pairs)
    delta = float(np.sum(counts * (credits[None, :] - credits[:, None])) / pairs)
    if not np.isclose(before_auc + delta, after_auc, atol=1e-13, rtol=0):
        raise AssertionError("exact rank-decomposition identity failed")

    return {
        "n_positive": int(len(plus_a)),
        "n_negative": int(len(minus_a)),
        "n_pairs": pairs,
        "matrix_before_rows_after_columns": counts.tolist(),
        "state_order": ["negative", "tie", "positive"],
        "auc_before": before_auc,
        "auc_after": after_auc,
        "auc_delta_exact": delta,
        "gross_pair_state_change_fraction": float(
            (pairs - np.trace(counts)) / pairs
        ),
        "strict_pair_reversal_fraction": float(
            (counts[0, 2] + counts[2, 0]) / pairs
        ),
        "pairwise_signed_concordance_gain": float(
            (counts[0, 2] - counts[2, 0]) / pairs
        ),
        "exact_ties_counted_at_half_credit": True,
    }


def subject_macro_rank_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    scope: str = "source_validation",
    harm_threshold: float = 0.02,
    block_size: int = 128,
) -> dict[str, Any]:
    """Aggregate one held participant per record; no pseudoreplicated trials.

    REQUIRED provenance fields for each record:
      participant_id; information_scope; source_fit_subject_ids;
      labels; score_before; score_after.
    For scope=source_validation the record must explicitly identify itself
    as source validation, not outer target data. This is a contract check,
    not independent proof of physical data provenance.
    """
    if scope not in ("source_validation", "posthoc_target"):
        raise ValueError("unknown information scope")
    if not np.isfinite(harm_threshold) or harm_threshold <= 0:
        raise ValueError("harm_threshold must be positive")
    if not records:
        raise ValueError("at least one participant is needed")

    subjects = set()
    rows = []
    for record in records:
        sid = str(record["participant_id"])
        if sid in subjects:
            raise ValueError("participant appears more than once; cannot pseudo-replicate")
        subjects.add(sid)
        if record["information_scope"] != scope:
            raise ValueError("cannot mix source and post-outcome target analyses")
        fit_ids = set(map(str, record["source_fit_subject_ids"]))
        if sid in fit_ids or not fit_ids:
            raise ValueError("held participant must not enter fitted source pool")
        transition = pairwise_rank_transition(
            record["labels"], record["score_before"], record["score_after"],
            block_size=block_size,
        )
        rows.append({"participant_id": sid, **transition})

    before = np.asarray([r["auc_before"] for r in rows])
    after = np.asarray([r["auc_after"] for r in rows])
    delta = after - before
    gross = np.asarray([r["gross_pair_state_change_fraction"] for r in rows])
    if not np.isclose(float(delta.mean()), float(np.mean(
        [r["auc_delta_exact"] for r in rows]
    )), atol=1e-13, rtol=0):
        raise AssertionError("subject-macro identity failed")
    return {
        "analysis_scope": scope,
        "usable_for_parameter_selection": scope == "source_validation",
        "no_target_generalisation_guarantee": True,
        "shared_training_pool_precludes_naive_independent_subject_inference": True,
        "n_participants": len(rows),
        "macro_auc_before": float(before.mean()),
        "macro_auc_after": float(after.mean()),
        "macro_auc_delta_exact": float(delta.mean()),
        "participant_mean_gross_pair_churn": float(gross.mean()),
        "n_held_participants_harmed": int(np.sum(delta < -harm_threshold)),
        "n_held_participants_improved": int(np.sum(delta > harm_threshold)),
        "harm_threshold_auc": float(harm_threshold),
        "participants": rows,
    }
