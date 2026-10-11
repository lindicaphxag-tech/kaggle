# Default Regularisation paper_vNext — evidence-locked status

Date: 2026-10-11 (V12 manuscript synchronisation).

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

## V12 current synchronisation (2026-10-11)

- Shin2017A (`N=29`, Z0) was separately frozen under PR #97 before the target outcomes were opened; mean AUC gains +0.02222 against covariance-NLL (Holm p=0.02232) and +0.04866 against Ledoit-Wolf (Holm p=0.00051). Material harm: 5/29 and 3/29, respectively.
- The participant-specific Shin2017A source-1SE fragility criterion passed (19/29 with path AUC span >0.02). This is **not** the v5 cohort-mean 405-point frontier.
- The lexicographic secondary selector failed another separately frozen non-inferiority test (Shin2017A). No new selector is promoted.
- Actual v5 original dense-path V11 analysis is complete: nominal translation-invariant simultaneous-path radii 0.04035 and 0.05923, with zero conditional conservative lower span bounds in both original external cohorts.
- Abstract, introduction, methods, results, discussion, main table, conclusion and highlights are now synchronised. Separate evidence is in `evidence/shin2017a_separate_prospective_v12.json`.
- **BSPC SUBMISSION-PACKAGE HOLD remains** pending executed tests, LaTeX build, reproducibility/journal checks. PR #50 remains Draft; nothing has been merged to `main`.

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
The frozen descriptive path results have now been reprocessed from the original artifacts under a post-outcome V11 translation-invariant simultaneous-path bootstrap. Its conditional lower span bounds are zero in both cohorts; material population path variation above 0.02 is not established.

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
the V11 post-outcome simultaneous-path sensitivity yielded zero lower span bounds in both cohorts, so a population span above 0.02 is not established. The observed critical raw fidelity
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
**PRIMARY PROSPECTIVE SCIENCE CORE COMPLETE; V11 FRONTIER INFERENCE CORRECTED.**

The original data-availability hold is cleared. The v5 raw paths have been reprocessed under V11 simultaneous-path sensitivity with a zero lower bound for each cohort. The main issue is now honest claim restraint and final package validation, not missing original frontier results.

Current state:
**SUBMISSION-PACKAGE HOLD** with corrected descriptive-only v5 Frontier claims. The primary S021 and separately frozen Shin2017A Holm-positive tests are retained. External v5 inferential families are still individually inconclusive; remaining gates are executed tests, final LaTeX build, and submission integrity.

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
- Added `src/frontier_simultaneous_path_sensitivity.py` with correlated whole-path resampling, a post-outcome conservative span correction and synthetic flat-null / clear-signal tests. The real v5 data were subsequently reprocessed in V11; radii and zero lower bounds are recorded in the authoritative v11 JSON evidence receipt.
- Added a portable LaTeX manuscript shell (`main_v10.tex`) and DOI-verified starter bibliography (`references_v10.bib`); compilation and complete journal submission requirements remain unverified.
- This is a substantive downgrade of one auxiliary claim, **not** a retraction of the frozen 20-person significant primary comparisons or the two external, individually inconclusive effect directions.

## V12 authoritative additional receipt

- `SHIN2017A_PROSPECTIVE_ROBUSTNESS_RECEIPT_20261010.md` and `evidence/shin2017a_separate_prospective_v12.json` are the separately frozen 29-person Shin2017A external replication; they are not additions to the v5 Holm family.
- `evidence/v11_actual_v5_simultaneous_frontier_reanalysis.json` is the original raw dense-path reanalysis. The prior positive percentile intervals are not population-level simultaneous inference.
- The GitHub Draft branch and `paper_vnext/main_v10.tex` (historical filename) contain V12 synced manuscript text, not an approved submission bundle.
