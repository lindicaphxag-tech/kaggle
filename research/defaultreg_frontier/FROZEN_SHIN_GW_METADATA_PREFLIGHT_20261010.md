# Default Regularisation — untouched Validity Frontier cohorts: metadata-only preflight freeze

Frozen 2026-10-10 before this project opens any classification outcome from
Shin2017A or GrosseWentrup2009.

These cohorts are reserved specifically for the later-developed
**Regularisation Validity Frontier** claim. Cho2017 and Lee2019_MI are not
used as confirmatory frontier cohorts because their real-data execution had
already started before the frontier amendment.

## Cohorts

### Shin2017A
- MOABB Shin2017A motor-imagery experiment only;
- 29 released participants;
- 3 sessions;
- left-hand versus right-hand motor imagery;
- 200 Hz public data;
- primary future use: full participant-heldout Z0 decoding plus per-session
  target reliability as a TNSRE bridge analysis.

### GrosseWentrup2009
- 10 participants;
- one session;
- left-hand versus right-hand haptic motor imagery;
- 128 EEG channels, 500 Hz;
- 150 trials per class per participant in the documented release.

## Stage 0 — metadata-only channel preflight

Before any epochs, labels, covariance matrices or classifier scores are
generated, the preflight may inspect ONLY:
- dataset identifier/version;
- subject list;
- raw EEG channel names;
- sampling frequency;
- session/run keys.

It must not:
- epoch task events;
- read target class labels into an analysis table;
- compute covariance/features/AUC;
- select channels by discriminability or signal quality.

The channel candidate family is frozen now:

FC5, FC3, FC1, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6,
CP5, CP3, CP1, CPz, CP2, CP4, CP6.

For each cohort, the eventual analysis montage is the deterministic
intersection between this candidate family and the raw channel names exposed
by metadata preflight. At least 7 channels must survive; otherwise that cohort
fails before outcomes and is not replaced.

The exact resulting montage is committed as a metadata receipt BEFORE the
outcome runner is created or executed. Any later missing-channel subject causes
fail-closed execution, not post-outcome substitution.

## Future frozen analysis principles

After channel preflight is committed, a separate outcome branch will freeze:

- full outer leave-one-participant-out Z0 evaluation;
- 8–30 Hz covariance/tangent representation;
- complete scalar shrinkage domain including 0, near-zero analytic region and 1;
- Ledoit-Wolf, OAS, source covariance-NLL optimum, source participant-macro-AUC
  optimum, tangent logistic and Riemannian MDM;
- target score vectors at every feasible frozen lambda generated before target
  labels are evaluated.

The primary Frontier estimand is NOT target-best lambda. For each outer target,
source folds define covariance-fidelity regret
r_C(lambda)=C(lambda)-min C. Target AUC Q_g(lambda) is evaluated only after the
whole path has been frozen. Report:
- AUC span inside the source covariance 1-SE set;
- threshold-free envelope over all observed source fidelity-regret budgets;
- fraction of participants whose downstream AUC span exceeds 0.02 and 0.05;
- whether a source-defined near-equivalent covariance region changes the
  sign/order of predeclared scientific comparisons.

No target outcome may change lambda grid, channel set, epoch, source folds,
fidelity budget or endpoint.

## TNSRE bridge on Shin2017A

For each held participant, preserve target session identity and report
per-session AUC using the SAME source-only fitted model. No target session
calibrates another target session. Session-to-session spread and worst-session
performance are deployment-reliability outcomes, not online validation.

## Interpretation

A replicated large validity envelope would support the claim that covariance
fidelity can underdetermine downstream cross-subject decisions.

A small envelope is a valid falsification: it would show that the objective
mismatch may exist numerically without materially destabilizing deployment
conclusions in that cohort.

These two cohorts must remain visible whether positive or negative.
