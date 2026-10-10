# Default Regularisation — outcome interpretation and claim-release freeze

Frozen: 2026-10-10 before any Shin2017A / BNCI2014_002 target outcome is available.

This document prevents a favorable-looking subset of the new experiment from
silently redefining what counts as success.

## Independent evidence axes

The new untouched experiment contains three logically distinct questions.

A. Does the older source-discrimination versus covariance objective ordering
replicate on held participants?

B. Are held-participant decisions materially sensitive inside a low
covariance-fidelity-regret region?

C. Does the post-S021 discriminability-first lexicographic rule preserve the
target performance of pure source-macro selection while improving source
covariance fidelity?

One axis cannot rescue another. Results are reported separately.

## A — objective-alignment replication

Primary paired contrast:
source_macro_full_domain_LDA minus source_cov_nll_full_domain_LDA.

Strong replication within a cohort:
- paired mean delta > 0; and
- participant-bootstrap 95% lower bound > 0.

Directional replication:
- paired mean delta > 0;
- bootstrap interval includes zero.

Contradiction:
- paired mean delta <= 0.

Ledoit-Wolf is a secondary reference, not the only replication baseline.

Cross-cohort wording:
- "replicated in both new cohorts" requires positive mean direction in both;
- "strongly replicated" requires the strong criterion in both;
- one positive / one non-positive cohort must be described as heterogeneous;
- no pooled-trial significance can override cohort disagreement.

## B — Regularisation Validity Frontier

The frontier itself remains continuous and is not reduced to one arbitrary
binary threshold.

Primary reported participant-level quantities:
- full-path AUC span;
- source covariance 1-SE AUC span;
- first source-regret budget at which >=0.02 AUC harm relative to the
  covariance optimum becomes reachable;
- first source-regret budget at which >=0.05 AUC harm becomes reachable;
- strict grid-resolution sensitivity.

Operational practical scale:
0.02 AUC is the predeclared "material" decision change because the same
magnitude is already used in the paper's harmed-participant reporting.

Cohort-level wording:

"decision sensitivity observed":
- source-1SE span >0.02 for at least one participant.

"decision sensitivity common":
- >50% of participants have source-1SE span >0.02.

"small-regret fragility":
- the cohort has non-zero source-1SE spans;
- >50% of participants exceed 0.02;
- and the 95% bootstrap interval for the mean source-1SE span has lower bound
  >0.01.

If these conditions fail, report the measured frontier but do not use
"fragile" in the headline.

A stable frontier is a scientifically valid robustness result.

Raw covariance-NLL regret budgets are not averaged across outer participants.
Each outer participant has its own source-only regret scale.

## C — lexicographic selector

Primary rule:
epsilon = 0.01 source macro-AUC.

Target paired contrast:
lexicographic minus pure source-macro selection.

The rule is not required to improve target AUC. Its purpose is to recover
covariance fidelity without materially sacrificing discrimination.

Target non-inferiority criterion within a cohort:
- paired mean delta >= -0.01 AUC; and
- participant-bootstrap 95% lower bound > -0.01 AUC.

Source-side benefit criterion:
- selected source covariance NLL is no worse than the pure source-macro
  optimum, with a strictly positive improvement whenever the selected lambda
  differs.

Promotable lexicographic result:
- target non-inferiority in BOTH new untouched cohorts;
- source-side covariance benefit in both cohorts whenever a different lambda
  is selected;
- no cohort has a >0.02 AUC harm-rate increase of more than 10 percentage
  points versus pure source-macro selection.

If target non-inferiority fails in either cohort, lexicographic selection is
reported as a negative post-S021 algorithm result and is not retuned.

Sensitivity epsilon values {0.005, 0.01, 0.02} are descriptive only; they
cannot replace epsilon=0.01 in the primary claim.

## Multiplicity

The primary method-consequence family remains:
1. source-macro vs source covariance-NLL;
2. source-macro vs Ledoit-Wolf.

Holm correction is applied within that family.

The lexicographic non-inferiority question is a separately frozen secondary
decision problem and is reported with its confidence interval rather than
added post-hoc to the superiority family.

Frontier magnitude is a measurement-validity estimand, not another
lambda-selection significance test.

## No optional stopping

All frozen participants in a cohort are processed unless a visible technical
failure prevents processing.

A cohort is not stopped early because:
- its direction is unfavorable;
- a p-value becomes significant;
- a frontier looks stable;
- the lexicographic rule looks inferior.

Technical failures remain in the execution receipt.

## Publication consequences

### BSPC strong mechanism version
Eligible if:
- S021 evidence remains intact;
- at least one independent external cohort directionally replicates the
  source-macro vs covariance-NLL ordering;
- all external negative/neutral cohorts are retained;
- provenance and same-information baseline gates pass.

### BSPC stronger validity version
Additionally requires:
- the new untouched frontier produces a material small-regret sensitivity
  result in at least one cohort;
- the second cohort does not directly contradict the qualitative mechanism.

### JNE candidate
Requires more than a modest gain:
- multi-cohort neural-engineering relevance;
- unusually complete prediction/provenance chain;
- strong same-budget comparator coverage;
- either a replicated validity insight or a robust reliability consequence.

### TNSRE
Not unlocked by these cohorts alone.

## Prohibited outcome-dependent actions

After target outcomes:
- do not alter the 0.02 materiality threshold;
- do not alter epsilon=0.01;
- do not replace source-1SE with a different admissibility rule in the headline;
- do not drop a participant/cohort for poor performance;
- do not change task class pair;
- do not change montage based on AUC;
- do not promote FCTR;
- do not call a stable frontier a failed experiment.

Status:
OUTCOME_INTERPRETATION_FROZEN_BEFORE_TARGET_RESULTS
