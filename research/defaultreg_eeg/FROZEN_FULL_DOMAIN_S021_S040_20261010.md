# Default Regularisation — prospective full-domain S021–S040 confirmation

Frozen 2026-10-10 before any S021–S040 target outcome is opened.

## Why this block exists

The historical manuscript and the first public pilots used a truncated
discriminability search grid. A restricted lower bound can mechanically
exaggerate the apparent distance from Ledoit-Wolf. This block removes that
confound.

The scientific question is NOT whether a new bespoke selector beats nested CV.
It is:

> Does the regularisation value preferred by source-participant
> discriminability remain separated from covariance-fidelity optima when the
> scalar shrinkage domain is searched from 0 to 1, and does that source-only
> choice improve genuinely unseen-participant decoding?

## Frozen dataset split

Dataset: PhysioNet EEGMMIDB v1.0.0.

Source participants:
S001–S008 only.

Previously exposed development/elimination participants:
S009–S020. They are NOT used in source fitting, selection or confirmation.

Prospective untouched target participants:
**S021–S040 (20 participants).**

Runs: R04/R08/R12.
Task: T1 vs T2 motor imagery.
Channels: same fixed 21 FC/C/CP channels.
Epoch: 0.5–3.5 s.
Band: 8–30 Hz.
All original task cues must be retained. Any silent cue loss invalidates the run.

## Full-domain shrinkage path

Frozen lambda path:

{0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015,
 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0}.

Every source-heldout fold refits the tangent reference without that source
participant. The path is evaluated by one eigendecomposition per source fold;
no hidden ridge rescues singular lambda=0.

For each lambda record:
- source-participant macro AUC;
- signed standardized transfer margin;
- squared margin;
- lower-tail signed margin;
- held-source covariance predictive NLL;
- negative-margin fraction;
- feasibility.

## Frozen primary source-only selections

**Discriminability selector (primary):**
lambda_D = argmax source-participant macro AUC.

**Covariance-fidelity comparator:**
lambda_C = argmin held-source covariance predictive NLL.

Also report analytic Ledoit-Wolf and OAS on all source participants.

No target feature, target covariance, target label, target batch statistic or
target AUC may choose lambda_D, lambda_C or an arm.

## Objective-conflict certificate

On source folds only, form:
- C_1SE: lambdas whose mean covariance NLL is within one source-participant SE
  of the minimum;
- D_1SE: lambdas whose source macro AUC is within one source-participant SE of
  the maximum.

Report:
- the two sets;
- their intersection;
- overlap/Jaccard;
- lambda_D and lambda_C;
- source discriminability regret at lambda_C:
  AUC(lambda_D) - AUC(lambda_C);
- covariance NLL regret at lambda_D:
  NLL(lambda_D) - NLL(lambda_C);
- domain-boundary hit status.

This is a descriptive source-only certificate, not a new statistical theorem.

## Prospective target hypotheses

H1 — Full-domain mismatch consequence:
source-macro-full-domain LDA has positive participant-level mean AUC delta
versus LedoitWolf LDA on S021–S040.

H2 — Objective-specific consequence:
source-macro-full-domain LDA has positive participant-level mean AUC delta
versus source-covariance-NLL-optimal LDA.

H3 — Robustness:
report lower quartile, worst delta, improved/harmed participant counts; a mean
gain purchased by severe hard-user harm cannot be described as robust.

H4 — Search-space audit:
if lambda_D falls near the old grid lower boundary or if C_1SE and D_1SE
substantially overlap, the strong "systematic mismatch" claim must be reduced.

## Statistics

Independent outcome unit: participant (N=20).

For H1/H2:
- exact paired sign-flip test (2^20 assignments);
- paired participant bootstrap CI;
- full per-participant table;
- Holm correction across H1 and H2.

Trial-level p-values are forbidden.

The overlap of LOSO training sets is explicitly acknowledged. Participant
bootstrap does not emulate source-pool retraining uncertainty; source-jackknife
stability is a separate sensitivity analysis.

## Interpretation

This block can confirm a within-EEGMMIDB zero-target regularisation-objective
mismatch. It still cannot establish a cross-dataset law. BNCI2014_001 has
already been exposed as a development/falsification cohort; Cho2017 and
Lee2019_MI remain required untouched external confirmation cohorts.

No TNSRE-ready or SOTA claim is permitted from this block alone.
