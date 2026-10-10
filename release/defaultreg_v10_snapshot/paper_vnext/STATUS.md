# SNAPSHOT_SOURCE: lindicaphxag-tech/cs@0b26a5b0853e3b136a92c17e614a3a1936673621 path=papers/eeg-default-regularisation/paper_vnext/STATUS.md
# Default Regularisation paper_vNext — evidence-locked status

Date: 2026-10-10.

## Current paper identity

**Regularisation validity in zero-target cross-subject EEG.**

The paper studies when covariance-fidelity-oriented regularisation and
participant-discrimination-oriented regularisation disagree, whether that
disagreement transfers to unseen participants, and whether it changes the
scientific conclusion drawn from EEG representations.

It is not:
- a universal theorem that all EEG defaults are wrong;
- a universal shrinkage multiplier;
- a new state-of-the-art covariance estimator;
- a clinical/online BCI study.

## Scientific core — completed

### 1. Full-domain source mechanism
EEGMMIDB S001--S008:
- source macro-AUC optimum lambda=0.8;
- source covariance-NLL optimum lambda=0.075;
- Ledoit--Wolf approximately 0.01567;
- OAS approximately 0.01953;
- complete [0,1] scalar domain available.

### 2. Prospective target consequence
EEGMMIDB S021--S040, N=20, Z0:
- source-macro AUC 0.72116;
- covariance-NLL AUC 0.66509;
- Ledoit--Wolf AUC 0.62919;
- +0.05607 vs covariance-NLL, Holm p=0.00597;
- +0.09197 vs Ledoit--Wolf, Holm p=0.00510;
- harmed-participant tails retained.

### 3. Source-pool robustness
Complete delete-one-source reselection:
- source-macro minus covariance-NLL remains positive in 8/8 omissions;
- source-macro minus Ledoit--Wolf remains positive in 8/8;
- selected discrimination lambda varies from 0.4 to 0.95.

Conclusion:
the stable object is objective ordering, not one universal lambda.

### 4. TRUE PRE-DATA external v5 panel
BNCI2014_002:
- N=14;
- source objective regions differ in 14/14 outer folds;
- H2 +0.01763;
- H3 +0.02287;
- frozen Holm p=0.13965.

BNCI2015_001:
- N=12;
- source objective regions differ in 12/12 outer folds;
- H2 +0.04329;
- H3 +0.04531;
- frozen Holm p=0.18555.

Two-cohort interpretation:
- source objective separation: 26/26 outer folds descriptively;
- H2/H3 target directions: positive in 2/2 datasets;
- individual external exact+Holm superiority: not established.

### 5. Corrected two-cohort Regularisation Validity Frontier
The pre-outcome material-fragility definition and paired-participant percentile bootstrap are complete; simultaneous path-level materiality inference is not.

BNCI2014_002:
- max AUC span 0.03844;
- bootstrap95 [0.02714,0.06656];
- delta_0.02 = 0.91611.

BNCI2015_001:
- max AUC span 0.05011;
- bootstrap95 [0.02238,0.10335];
- delta_0.02 = 0.17519.

Conclusion:
the observed frontier magnitude is positive in both datasets, but
whether a population-level 0.02 path span survives simultaneous-path
uncertainty remains unverified. The observed critical raw fidelity
budget is cohort-dependent. No universal delta is claimed.

Canonical receipt:
V5_MATERIAL_FRAGILITY_FRONTIER_RECEIPT_20261010.md.

### 6. Prospective representation consequence
EEGMMIDB S041--S060:
- primary FB8-vs-WIDE regularisation interaction -0.07781 AUC;
- bootstrap95 [-0.12415,-0.03451];
- Holm p=0.01328.

The historical hidden-filter-bank direction is prospectively refuted:
source-discrimination strengthens the WIDE advantage.

Matched-dimension PCA, nuisance-dimension and source-budget controls also
reject a simple monotonic p/n explanation.

### 7. Negative / demoted selector evidence
Not promoted:
- lower-tail signed selector;
- CFCTS;
- FCTR;
- v5 mean--tail--fidelity selector;
- post-S021 discriminability-first lexicographic selector.

The paper therefore has no headline "new selector" claim.

### 8. Operator mechanism
Added:
- OBJECTIVE_MISMATCH_OPERATOR_THEOREM_20261010.md;
- src/regularisation_path_geometry.py;
- derivative and finite-difference tests.

Key result:
standardized margin changes only through the evaluation-covariance-orthogonal
rotation of the discriminant path, whereas covariance-NLL derivative contains
no class contrast. Near an interior covariance optimum, fidelity regret can be
second order while discriminative margin changes to first order.

This is a mechanism derivation, not a universal EEG theorem.

### 9. Calibration identifiability
EEGMMIDB S061--S080:
- Z0 source-macro vs Ledoit--Wolf remains positive;
- U5/U10/U20/U-all ROC-AUC changes are algebraically forced to zero by the
  tested constant participant-specific score translation.

This block does not establish practical calibration utility.

## Historical claims permanently retired

Do not reintroduce:
- 4--19x universal multiplier;
- 98.3% / 82% historical summary claims without provenance;
- hidden filter-bank advantage;
- unresolved 102/4018 vs 103/4076 table lineage;
- "validation-free" language;
- universal EEG-wide default indictment.

Historical inconsistent rows are excluded from vNext evidence tables rather
than silently repaired.

## Current algorithm hierarchy

Primary operational comparator:
**participant-grouped full-domain source macro-AUC selection.**

This is classical source model selection used as an instrument for the
objective-mismatch question, not presented as a new estimator.

All newer source-only selectors remain secondary or negative.

## Submission state

### BSPC
**PRIMARY PROSPECTIVE SCIENCE CORE COMPLETE; FRONTIER-INFERENCE GATE OPEN.**

The original data-availability hold for external and representation outcomes is
cleared. However, a V10 reviewer stress test identified a remaining
simultaneous-inference issue in the post-score validity-frontier interpretation.

Current state:
**SUBMISSION-PACKAGE HOLD** with an explicit post-outcome frontier-inference gate.
The primary frozen AUC contrasts remain intact, but stronger population-level
frontier materiality wording is blocked pending simultaneous-path validation.

Remaining high-value release work:
1. retain the **completed** post-primary CSP6+LR comparator for both MI cohorts (do not relabel as pre-registered), and add FBCSP/stronger matched-budget depth if benchmark claims are intended;
2. obtain a successful executed CI/local-test receipt (recent cs runner jobs have null steps; this is not a test pass);
3. run the new post-outcome participant-harm/influence audit and generate its canonical supplemental receipt, alongside already generated manuscript metric tables;
4. finish and compile figures, references, highlights, declarations and journal-format manuscript assembly;
5. preserve damaged-participant uncertainty, all failed selectors, and external Holm non-significance.

### JNE
**Credible conditional alternative.**

The operator mechanism, two TRUE PRE-DATA cohorts, validity frontiers and
prospective representation falsification raise the paper above a simple
classification-gain study. JNE remains less conservative than BSPC and should
be chosen only if the final framing emphasizes neural-engineering validity and
the comparator/reproducibility package is strong.

### TNSRE
**Not currently justified.**

A separate threshold-sensitive, practical BCI utility study is still required,
preferably with cross-session/device or online-like validation.

## Authoritative receipts

Key current receipts include:
- PROSPECTIVE_S021_S040_FULL_DOMAIN_RECEIPT_20261010.md;
- S021_S040_SOURCE_POOL_FULL_RESELECTION_RECEIPT_20261010.md;
- S021_S040_POSTHOC_SCORE_REPRODUCTION_RECEIPT_20261010.md;
- BNCI2015_001_V5_TRUE_PREDATA_RESULT_RECEIPT_20261010.md;
- V5_MATERIAL_FRAGILITY_FRONTIER_RECEIPT_20261010.md;
- VERIFIED_PROSPECTIVE_RESULT_RECEIPT_V7_20261010.md.

**Current release decision: BSPC science-ready, package not yet submit-ready.**

## V9 release-hardening addendum (2026-10-10)

- BSPC-oriented abstract was compressed to approximately 200 words while preserving the predeclared statistics and external nulls.
- A six-contrast post-outcome participant harm/influence script and synthetic/frozen-vector contract tests were added; their presence does not certify execution.
- Mean--median disparity (especially BNCI2015_001: +0.0433 vs +0.0105 versus covariance-NLL) is explicitly disclosed.
- See BSPC_V9_ACCEPTED_PAPER_AND_RELEASE_GAP_20261010.md for verified accepted-work comparisons and a non-SOTA release checklist.
- Current decision remains **SUBMISSION-PACKAGE HOLD** until the listed receipts and actual manuscript assembly are verified.

## V10 reviewer-critical inference correction (2026-10-10)

- The primary shrinkage selector uses a finite **grid spanning** the full [0,1] interval; neither continuous optimization nor a zero-grid-error proof is asserted.
- The 405-point post-score frontier maximizes/minimizes observed AUC across lambda. The percentile bootstrap lower bound of this selected span is **not** a simultaneous lower confidence bound for the population span; the old "bootstrap lower >0.02 proves materiality" interpretation is retired.
- Added `src/frontier_simultaneous_path_sensitivity.py` with correlated whole-path resampling, a post-outcome conservative span correction and synthetic flat-null / clear-signal tests. New real v5 corrected intervals are **not yet computed** from the archived raw dense paths.
- Added a portable LaTeX manuscript shell (`main_v10.tex`) and DOI-verified starter bibliography (`references_v10.bib`); compilation and complete journal submission requirements remain unverified.
- This is a substantive downgrade of one auxiliary claim, **not** a retraction of the frozen 20-person significant primary comparisons or the two external, individually inconclusive effect directions.
