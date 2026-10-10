# Default Regularisation — S061-S080 target-information budget deployment block

Frozen 2026-10-10 before any S061-S080 target outcome is opened.

## Why this block exists

"No target labels" is not the same as "zero calibration". A method may still
use target feature moments or an entire target batch. This block separates
three deployment budgets rather than mixing them:

- **Z0**: no target feature/statistic/label before scoring;
- **U-k**: exactly the first k chronological target trials, labels hidden,
  may be used for a fixed unsupervised feature-translation calibration;
- **U-all**: the complete unlabeled target batch may be used and is explicitly
  labelled transductive.

The block is a deployment-sensitivity experiment, not a new algorithm search.

## Frozen dataset and source model

Dataset: official PhysioNet EEGMMIDB v1.0.0.

Source participants: S001-S008.
Untouched deployment targets: **S061-S080**.
Runs: R04/R08/R12.
Channels: the same fixed 21 FC/C/CP channels.
Band: 8-30 Hz.
Epoch: 0.5-3.5 s.
Every original T1/T2 cue must be retained.

Source tangent reference and full-domain shrinkage selection are fitted from
source participants only. The source-discriminability lambda is selected over
the already frozen full [0,1] path. Ledoit-Wolf is the analytic comparator.

No target outcome may change either source model.

## Frozen unlabeled target calibration operator

All target trial covariances are first mapped through the frozen SOURCE
tangent reference. Let X_s be source tangent features and X_t target tangent
features. Define source mean m_s.

For a target prefix C_k containing the first k chronological target trials,
labels hidden:

  delta_k = mean(X_t[C_k]) - m_s

and apply to FUTURE target trials:

  X_t_aligned = X_t - delta_k.

This is a fixed feature-translation sensitivity operator, not a learned
domain-adaptation model. It uses no target labels and does not change the
source classifier.

For U-all, the same translation uses the complete unlabeled target feature
batch. U-all is transductive by definition.

## Frozen budgets and evaluation denominators

- Z0-all: no target calibration; score all target trials.
- U5: first 5 trials are calibration only; evaluate trials 6..end.
  The paired Z0 comparator is evaluated on the SAME suffix.
- U10: first 10 calibration only; evaluate trials 11..end, with matched Z0 suffix.
- U20: first 20 calibration only; evaluate trials 21..end, with matched Z0 suffix.
- U-all: use all target features to estimate translation and score all trials;
  compare descriptively with Z0-all but label this explicitly transductive.

Chronology is fixed by run number and event sample. If a suffix lacks either
class, that participant-budget cell is marked unavailable; it is not replaced
by a different k.

## Methods

For every information budget:
1. full-domain source-discriminability-selected shrinkage LDA;
2. analytic Ledoit-Wolf LDA.

The calibration operator is identical for both models.

## Questions

D1: Does the prospective Z0 advantage survive on another untouched
20-participant block?

D2: How much, if anything, does finite unlabeled target recentering change AUC
for each model on the matched future-trial suffix?

D3: Is any apparent U-all transductive advantage already achievable with a
small prefix, or does it depend on seeing the complete target batch?

D4: Report calibration sample cost, score denominator, participant-level
benefit/harm and runtime. Do not report "calibration-free" for U-k/U-all.

## Statistics

Participant is the unit. Z0 vs U-k uses paired participant deltas on the same
suffix. Z0 model-vs-model comparisons use the same target trials.
Predeclared deployment contrasts receive Holm correction within the family.

This block is separate from S021-S040 primary mechanism confirmation and
S041-S060 representation interaction. It cannot be used to redesign either.

## Interpretation

A useful U-k curve can support practical BCI calibration-cost discussion.
U-all can only support a transductive offline statement. None of these offline
conditions proves real-time online BCI performance.
