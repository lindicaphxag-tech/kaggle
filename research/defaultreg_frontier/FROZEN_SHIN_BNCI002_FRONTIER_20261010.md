# Default Regularisation — untouched frontier confirmation freeze

Frozen: 2026-10-10, before this branch opens any target AUC from Shin2017A or BNCI2014_002.

## Evidence chronology

Already opened:
- EEGMMIDB S021-S040 prospectively confirmed the older, simpler claim that source-participant discriminability-selected scalar shrinkage can outperform both analytic Ledoit-Wolf and source covariance-NLL shrinkage under Z0.
- Cho2017 / Lee2019_MI were subsequently exposed by concurrent workflows before the later-developed Validity-Frontier and lexicographic-selector claims could remain blind.

Therefore Cho/Lee may be used only for chronology-appropriate replication of the older objective-alignment claim.

This branch is the first untouched execution eligible to confirm the later:
1. Regularisation Validity Frontier;
2. critical fidelity regret;
3. discriminability-first lexicographic selector.

FCTR is NOT a headline method and is not rehabilitated here.

## Frozen cohorts

### Shin2017A
- MOABB dataset class: Shin2017A;
- 29 participants;
- 3 sessions;
- motor imagery only;
- left_hand versus right_hand;
- use all EEG channels exposed by the frozen loader;
- no participant/session exclusion based on decoding performance.

### BNCI2014_002
- MOABB dataset class: BNCI2014_002;
- 14 subject files exposed by the loader;
- 1 session;
- right_hand versus feet;
- use all 15 EEG channels exposed by the dataset;
- all documented runs retained;
- no participant/run exclusion based on decoding performance.

The cohorts intentionally differ in montage and feature dimension. The confirmatory question is whether the regularisation-validity phenomenon survives those changes; raw NLL values are therefore interpreted within cohort first.

## Environment

Frozen primary stack:
- Python 3.11;
- MOABB 1.7.2;
- scikit-learn and pyRiemann versions resolved by that installation unless the environment receipt reports otherwise;
- single-threaded BLAS environment variables for deterministic resource use.

## Information budget

Primary track: Z0.

For every outer held participant:
- no target label is accepted by source geometry fitting;
- no target feature mean/covariance enters source selection;
- no target batch statistic enters selection;
- no target-derived quality threshold;
- no target-tuned lambda;
- no target-tuned representation.

Target labels are evaluation-only after the entire predeclared target score family has been generated and hashed.

## Signal representation

For both datasets:
- dataset-native motor-imagery interval from the frozen MOABB loader;
- 8-30 Hz;
- resample to 160 Hz;
- all dataset EEG channels;
- trial covariance with a fixed numerical SPD floor;
- source arithmetic SPD reference;
- source-fitted whitened log-tangent coordinates.

The target participant contributes nothing to the tangent reference.

## Scalar regularisation family

Historical performance-selection grid:
{0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
 .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
 .90, .95, .98, .99, 1.0}.

Validity-frontier grid:
- uniform step 0.0025 over [0,1];
- plus near-zero anchors
  {0,.0005,.001,.002,.003,.005,.0075,.01,.015,.02,.03,.05}.

No interpolation is used for primary frontier estimates.

## Source selection

For each outer target, source participants are partitioned deterministically into five grouped validation folds using sorted participant order modulo five.

Geometry is refit inside every inner fold.

Source criteria:
- participant-macro AUC, larger better;
- covariance predictive NLL, smaller better.

### Source-macro selector
Choose the FULL_GRID lambda with maximum source participant-macro AUC.

### Covariance selector
Choose the FULL_GRID lambda with minimum source covariance predictive NLL.

### Discriminability-first lexicographic selector
Primary margin epsilon = 0.01 AUC.

Let D_epsilon be all FULL_GRID lambdas whose source participant-macro AUC is at least best_source_AUC - 0.01.
Within D_epsilon choose minimum source covariance predictive NLL.

Sensitivity margins:
{0.005, 0.01, 0.02}.

The 0.01 rule is primary and cannot be replaced after target outcomes.

## Target score freeze

Before target labels are used for evaluation, freeze and hash:
- Ledoit-Wolf LDA scores;
- OAS LDA scores;
- source-macro-selected LDA scores;
- source covariance-NLL-selected LDA scores;
- lexicographic-selected LDA scores;
- tangent logistic-regression scores;
- dense-grid scalar-LDA score matrix for every feasible frontier lambda.

The score-family SHA256 is part of the result receipt.

Target labels are then applied to the immutable score family.

## Primary hypotheses

### H1 — objective alignment replication
Does source-macro selection outperform source covariance-NLL selection on held participants?

This extends the older S021 claim to different public cohorts.

### H2 — frontier consequence
As source covariance-fidelity regret increases from zero, does the held-participant AUC envelope widen materially?

Primary reported quantities:
- full regret-to-envelope path;
- maximum participant-macro AUC envelope span;
- participant bootstrap uncertainty;
- grid-resolution sensitivity.

### H3 — scientific contrast fragility
For any predeclared signed contrast available in this branch, report critical fidelity regret delta* only if the contrast actually crosses zero. No zero crossing is a valid robustness result, not a failure.

### H4 — lexicographic selector
Compare the discriminability-first lexicographic rule against:
1. pure source-macro selector;
2. source covariance-NLL selector;
3. Ledoit-Wolf.

The lexicographic rule is successful only if it preserves source-macro target performance within the predeclared tolerance while improving covariance fidelity or reducing instability. A target AUC win is not required.

## Participant-level inference

Independent unit: participant.

Per cohort:
- mean/median/Q25 paired AUC delta;
- 20,000 paired participant bootstrap CI;
- exact sign-flip if n<=20;
- deterministic 200,000-draw Monte-Carlo sign-flip otherwise;
- Holm correction over the two primary method-consequence comparisons;
- improved count;
- >0.02 AUC harm count;
- worst participant delta.

## Frontier uncertainty

Bootstrap participant indices jointly across every lambda.
The same resampled indices are used for the entire regularisation path.

Report:
- point maximum envelope span;
- bootstrap CI;
- fraction of bootstrap draws with finite delta* where applicable;
- conditional-finite delta* CI.

Participant bootstrap does not represent source-training-set uncertainty.

## Grid sensitivity

Recompute frontier summaries on strict dense-grid subsets with strides:
1, 2, 4, 8.

If qualitative conclusions change with grid resolution, the frontier is reported as discretisation-sensitive.

## Cross-cohort interpretation

Shin2017A and BNCI2014_002 are reported separately first.

Dataset is the replication stratum.
Do not pool trials across datasets for significance.

After both cohorts:
- equal-dataset mean/median/range;
- direction consistency;
- leave-one-dataset-out direction audit where defined.

Two datasets remain a small replication panel and do not justify universal language.

## Fail-closed rules

The workflow must fail if:
- participant denominator differs from frozen dataset metadata without an explicit loader-level structural reason recorded before scoring;
- target score-family hash is absent;
- target label reaches source selection;
- FULL_GRID endpoints are missing;
- dense frontier endpoints are missing;
- method set drifts across target participants;
- a target participant is absent from the final result without a visible processing failure receipt.

## Claim boundary

A positive result may support:
- replication across the studied cohorts;
- regularisation-validity / decision-fragility claims under the frozen Z0 pipeline;
- prospective evaluation of the lexicographic selector.

It does NOT establish:
- universal EEG regularisation law;
- SOTA across unrelated architectures;
- online/clinical performance;
- TNSRE-level BCI utility.

A negative result is retained and narrows the paper.

Status before execution:
UNTOUCHED_FRONTIER_CONFIRMATION_FROZEN_NO_TARGET_OUTCOME
