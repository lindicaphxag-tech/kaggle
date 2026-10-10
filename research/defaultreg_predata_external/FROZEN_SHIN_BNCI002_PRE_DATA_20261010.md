# Default Regularisation — PRE-DATA external confirmation freeze

Frozen: 2026-10-10, before any Default Regularisation target AUC from
Shin2017A or BNCI2014_002 is opened on this branch.

## Fixed cohorts

### Shin2017A
- 29 released participants.
- 3 motor-imagery sessions.
- 30 EEG channels.
- left_hand vs right_hand.
- use all released MI participants/sessions.

### BNCI2014_002
- 14 released participant files.
- 1 session, 8 runs.
- 15 EEG channels.
- right_hand vs feet.
- use all released participants/runs.

Both use the first 4.0 s of the MOABB task epoch, 8–30 Hz and 160 Hz
resampling. Participant is the outer unit.

## Z0 information budget

For each outer participant:
- no target label before all target score vectors are produced;
- no target mean/covariance/reference;
- no target batch normalization;
- no target-derived lambda or representation;
- no target quality statistic.

## Full-domain shrinkage

Fixed grid:
{0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
 .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
 .90, .95, .98, .99, 1.0}.

Every outer fold reports source participant-macro AUC and source covariance
predictive NLL over the complete feasible domain, analytic Ledoit-Wolf/OAS
shrinkage, selected lambda, boundary hits and target AUC path frozen before
target labels.

## Frozen baseline panel

1. LedoitWolf LDA.
2. OAS LDA.
3. source covariance-NLL full-domain LDA.
4. source participant-macro-AUC full-domain LDA.
5. tangent-space LogisticRegression(C=1).
6. Riemannian MDM.
7. covariance-form CSP, six source-fitted components + LogisticRegression(C=1).

No method may be removed because it is strong. Tail-signed / CFCTS / GSDO /
TMS variants are not headline methods.

## Primary paired contrasts

H2: source-macro full-domain LDA minus source covariance-NLL full-domain LDA.
H3: source-macro full-domain LDA minus LedoitWolf LDA.

Participant paired bootstrap and sign-flip inference; Holm across H2/H3.

## Hard-user reporting

For each contrast report mean, median, Q25, Q10 where meaningful, minimum,
improved/tied counts, >0.01 and >0.02 harm counts, and lower-tail mean.

## Provenance

Archive package versions, participant/session/run/trial receipts, raw-cache
file SHA256 manifest, method set, complete per-fold lambda path, method/fold
runtime and final JSON. Any technical failure stays visible and cannot trigger
cohort replacement after result access.

Status at freeze: PRE-DATA / SCIENCE HOLD.
