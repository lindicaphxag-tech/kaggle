# Default Regularisation vNext — title, claim hierarchy and venue lock

Date: 2026-10-10.

## Recommended BSPC title — current evidence supports this level

**Regularisation Validity in Cross-Subject EEG:
Prospective Objective Mismatch, Validity Frontiers, and Representation Dependence**

Why this title is now defensible:

- the complete scalar domain is evaluated rather than a truncated high-shrinkage grid;
- the operator derivation identifies the exact path-rotation term that can
  change standardized discrimination while covariance predictive loss contains
  no class-contrast term;
- EEGMMIDB S021--S040 prospectively confirms a held-participant consequence
  relative to both covariance-NLL selection and Ledoit--Wolf;
- complete delete-one-source reselection preserves the objective ordering in
  8/8 source omissions while the numerical lambda itself varies;
- two TRUE PRE-DATA external cohorts complete one unchanged v5 protocol and
  separate source objective optima in 26/26 outer folds;
- H2/H3 target effects are positive in both external datasets, while their
  frozen exact+Holm families remain individually inconclusive;
- corrected material-fragility frontiers are uncertainty-complete in both
  external cohorts;
- S041--S060 prospectively shows that regularisation objective changes the
  representation conclusion, while falsifying the historical hidden
  filter-bank direction.

The title refers to validity inside the studied covariance/tangent
linear-discriminant family. It does not imply a universal EEG theorem.

## Conservative BSPC fallback title

**When Covariance Fidelity and Cross-Subject Discrimination Disagree:
Prospective Evidence That Regularisation Alters EEG Representation Conclusions**

Use this version if an editor or co-author prefers a title with fewer newly
introduced terms. It does not change the evidence hierarchy.

## JNE-facing title — only if journal strategy changes

**Regularisation Validity in Zero-Target Cross-Subject EEG:
Decision Geometry, External Replication, and Representation Dependence**

This title is defensible only if the final manuscript retains the
neural-engineering interpretation, modern same-budget comparators and the
full provenance chain. It does not justify a clinical or online claim.

## TNSRE-facing title — future separate extension only

**Regularisation-Objective Mismatch in Zero-Calibration Cross-Subject EEG:
Reliability, Operating-Point Utility and Cross-Session Consequences**

Do not use this title for the current manuscript. The completed U-k AUC
experiment is not a calibration-utility result because its constant
participant-specific score translation makes ROC AUC algebraically invariant.
TNSRE requires a new threshold-sensitive practical endpoint and preferably
cross-session/device or online-like validation.

## Claim hierarchy

### C1 — operator mechanism

For scalar shrinkage, standardized held-participant margin changes only
through the evaluation-covariance-orthogonal rotation of the discriminant
path, while covariance predictive NLL has no class-contrast term.

Near an interior covariance optimum, covariance regret can be second order in
path displacement while a nonzero discriminative-margin change is first
order.

Status:
**DERIVED AND UNIT-TESTED IN SOURCE CODE; NOT A UNIVERSAL EEG THEOREM.**

The abstract statistical fact that covariance and classification objectives
can differ is classical. The contribution is the explicit operator mechanism
connected to the prospective EEG evidence chain.

### C2 — full-domain source objective mismatch

Within the studied covariance/tangent linear-discriminant family, source
covariance-fidelity and source participant-discrimination objectives can
select materially different regions over the complete scalar domain.

Status:
**SUPPORTED.**

Evidence includes authentic EEGMMIDB source paths, post-hoc BNCI2014_001
stress evidence and TRUE PRE-DATA external source paths.

### C3 — prospective held-participant consequence

Under EEGMMIDB S021--S040 Z0, source-macro full-domain selection improves
participant-macro AUC relative to both source covariance-NLL and
Ledoit--Wolf, with both predeclared comparisons surviving Holm correction.

Status:
**PROSPECTIVELY SUPPORTED.**

The participant harm tail remains visible and prohibits a uniform-safety claim.

### C4 — external replication

BNCI2014_002 and BNCI2015_001 completed the unchanged TRUE PRE-DATA v5
protocol.

Source objective regions differ in:
- 14/14 BNCI2014_002 outer folds;
- 12/12 BNCI2015_001 outer folds;
- 26/26 total outer folds descriptively.

Target H2 mean deltas:
- +0.01763;
- +0.04329.

Target H3 mean deltas:
- +0.02287;
- +0.04531.

Status:
**SOURCE OBJECTIVE SEPARATION REPLICATED; TARGET EFFECT DIRECTIONALLY
REPLICATED; INDIVIDUAL EXTERNAL EXACT+HOLM SUPERIORITY REMAINS INCONCLUSIVE.**

Do not pool trials or participants across datasets to manufacture a combined
p-value.

### C5 — Regularisation Validity Frontier

The corrected, pre-outcome definition measures the AUC envelope admitted by a
source covariance-regret budget and uses material span thresholds rather than
the structurally invalid generic centered-AUC zero-crossing statistic.

BNCI2014_002:
- max AUC span 0.03844;
- bootstrap95 [0.02714, 0.06656];
- delta_0.02 = 0.91611.

BNCI2015_001:
- max AUC span 0.05011;
- bootstrap95 [0.02238, 0.10335];
- delta_0.02 = 0.17519.

Status:
**TWO TRUE PRE-DATA COHORTS WITH UNCERTAINTY-COMPLETE FRONTIERS.**

Supported conclusion:
frontier magnitude reproduces, but frontier steepness / critical raw fidelity
budget is cohort-dependent.

Not supported:
one universal delta threshold.

### C6 — representation consequence

Regularisation objective can alter the scientific comparison between
wide-band and filter-bank representations.

Status:
**PROSPECTIVELY SUPPORTED; HISTORICAL DIRECTION REFUTED.**

Evidence:
- S041--S060 primary interaction -0.07781 AUC,
  Holm p=0.01328;
- source-discrimination strengthens WIDE rather than revealing a hidden
  filter-bank advantage;
- FB8-SOURCE-PCA231 remains below WIDE231;
- source-budget and nuisance-dimension controls reject a simple monotonic
  p/n explanation.

### C7 — new selector claims

Lower-tail signed selection, CFCTS and FCTR are negative/demoted controls.

The v5 mean--tail--fidelity selector fails its frozen two-cohort tail promotion
rule.

The post-S021 discriminability-first lexicographic selector is descriptively
worse than pure source-macro selection in both TRUE PRE-DATA external cohorts
(mean approximately -0.00865 and -0.00753 AUC).

Status:
**NO NEW SELECTOR IS PROMOTED AS THE PAPER'S HEADLINE ALGORITHM.**

This is intentional: the paper's innovation is the validity phenomenon and
its evidence chain.

### C8 — deployment/calibration consequence

S061--S080 confirms a positive Z0 source-macro versus Ledoit--Wolf result, but
the U5/U10/U20/U-all AUC contrasts are structural invariants of the tested
constant participant-specific score translation.

Status:
**AUC CALIBRATION-UTILITY CLAIM REJECTED AS NON-IDENTIFIABLE.**

A threshold-sensitive follow-up would be a new, post-outcome study.

## Claims prohibited from the manuscript

- no universal "EEG default is wrong" statement;
- no universal 4--19x multiplier;
- no SOTA claim from cross-paper metric comparison;
- no validation-free description of participant-grouped source selection;
- no formal-equivalence language for one-SE sets;
- no claim that every participant benefits;
- no claim that external H2/H3 are individually Holm-significant;
- no universal material-fragility budget;
- no promotion of FCTR, CFCTS, signed-tail, mean-tail or lexicographic
  selectors beyond their actual frozen evidence;
- no hidden filter-bank advantage;
- no claim that simple p/n is a validated law;
- no calibration benefit/harm claim from the constant-translation AUC test;
- no clinical, online or TNSRE-ready claim.

## Venue lock

### BSPC — primary and highest-value current submission route

The manuscript is now a coherent biomedical signal-processing /
measurement-validity study built around:
- covariance regularisation geometry;
- complete-path source objective mismatch;
- prospective Z0 target consequence;
- two TRUE PRE-DATA external cohorts;
- uncertainty-complete validity frontiers;
- prospective representation falsification;
- negative algorithm evidence;
- fail-closed participant-level provenance.

Remaining release work should be editorial / comparator / reproducibility
hardening, not invention of another selector.

### JNE — credible conditional alternative

JNE is now materially more plausible than before because the paper contains a
neural-engineering validity question rather than only a classifier gain.
Prefer JNE only if the final package includes:
- strong same-information classical baselines;
- the complete operator/validity interpretation;
- external cohort and harmed-participant distributions;
- reproducible score-level provenance;
- a clear explanation of why the result changes EEG inference rather than
  only benchmark accuracy.

### TNSRE — separate future extension

The current paper still lacks a valid direct operating-point/calibration
utility result. A TNSRE extension requires a new threshold-sensitive or
online-like BCI endpoint, preferably with cross-session/device robustness.
Do not relabel the current paper as TNSRE-ready by changing prose.
