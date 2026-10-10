# Default Regularisation — untouched Cho2017 + Lee2019_MI full-domain replication freeze

Frozen 2026-10-10 before this branch opens any Cho2017 or Lee2019_MI target-subject AUC.

## Purpose

The historical paper mixed different channel counts, feature dimensions and
protocols. This external replication removes that ambiguity from the primary
cross-cohort mechanism test by using the SAME 21 sensorimotor channels and
the SAME 231-dimensional wide-band covariance/tangent representation on two
independently recorded public motor-imagery cohorts.

These cohorts are independent of the PhysioNet development/prospective blocks
and of the already-exposed BNCI2014_001 targets.

## Public cohorts

### Cho2017
- all 52 available participants;
- left-hand versus right-hand motor imagery;
- one session;
- original dataset has 64 EEG channels at 512 Hz;
- main analysis selects the frozen common 21 channels below;
- MOABB epoch extraction uses 8--30 Hz, t=0--3 s, resample=160 Hz,
  reject_by_annotation=False;
- no participant is removed because of classification performance;
- the exact number of epochs returned per participant is archived rather than
  forced to equality because dataset-specific bad-trial metadata exist.

### Lee2019_MI
- all 54 available participants;
- two sessions;
- labelled TRAIN/OFFLINE run only. The dataset's MI online TEST run is
  intentionally excluded because MOABB documents that it lacks trial labels;
- left-hand versus right-hand motor imagery;
- original dataset has 62 EEG channels at 1000 Hz;
- main analysis selects the same frozen 21 channels;
- MOABB epoch extraction uses 8--30 Hz, t=0--4 s, resample=160 Hz,
  reject_by_annotation=False.

## Frozen common 21-channel montage

FC5, FC3, FC1, FCz, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6,
CP5, CP3, CP1, CPz, CP2, CP4, CP6.

If a dataset cannot provide this exact set, the workflow FAILS. It must not
silently substitute a different montage after seeing results.

## Outer evaluation

Full leave-one-participant-out (LOPO) on every available participant.

For each outer target:
- target covariances, target feature moments and target labels are excluded
  from source representation fitting, shrinkage selection and classifier fit;
- target labels are read only after all score vectors are frozen;
- the final tangent reference is fit from outer-source covariances only.

Information budget: **Z0**.

## Inner source selection

To keep the >50-participant datasets computationally auditable while retaining
strict participant grouping, outer-source participants are partitioned into
five deterministic validation folds by participant ID modulo five.

For each inner fold:
- tangent geometry is refit using only inner-training source participants;
- held-source participants contribute no labelled or unlabeled statistics to
  that fold's fit;
- each held-source participant receives its own AUC and covariance-predictive
  NLL contribution;
- criteria are macro-averaged over source participants, not over trials.

This split rule is fixed before target outcomes and is never optimized.

## Full scalar path

{0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
 .05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
 .90, .95, .98, .99, 1.0}.

Primary source-only objectives:
1. maximize participant-macro source AUC;
2. minimize held-source covariance predictive NLL.

Analytic Ledoit-Wolf and OAS are evaluated on the final outer-source feature
matrix and their actual selected shrinkage values are logged.

## Frozen held-target methods

1. LedoitWolf LDA
2. OAS LDA
3. full-domain source-participant discriminability-selected LDA
4. full-domain source covariance-NLL-selected LDA
5. tangent-space LogisticRegression (fixed C=1)
6. Riemannian MDM on the same 21-channel trial covariances

The MDM arm is a strong non-LDA covariance-geometry comparator. It is not
allowed to determine the shrinkage method or hypotheses.

## Confirmatory questions

H1 OBJECTIVE MISMATCH:
Across outer folds, do covariance fidelity and source-participant
discriminability select materially different lambda regions after the entire
[0,1] domain is available?

H2 TARGET CONSEQUENCE:
Does source-discriminability selection improve participant AUC relative to
analytic Ledoit-Wolf under Z0? This is a method consequence of the mismatch.

H3 STRONGER BASELINE:
Does source-discriminability selection improve over source covariance-NLL
selection? If not, the paper may still support objective mismatch but must not
claim that discriminability tuning is uniquely necessary.

H4 PRACTICAL ROBUSTNESS:
Report lower quartile, worst participant, harmed-by->0.02 count, train/select
time and per-trial score time. Mean improvement alone is insufficient.

## Multiplicity and inference

Participant is the inferential unit.
For each cohort:
- paired participant effects;
- participant bootstrap CI;
- deterministic Monte-Carlo sign-flip for N>20;
- Holm correction across H2/H3 and any predeclared primary method contrasts.

Across cohorts:
- report each dataset separately first;
- equal-dataset-weighted hierarchical summary is descriptive only;
- never pool trials from different participants/datasets as independent units.

## Claim boundary

A positive Cho2017/Lee2019 result can provide independent full-domain
replication for a BSPC mechanism paper. It does not by itself establish online
BCI utility or TNSRE readiness.

A negative result is retained. No subject, lambda subrange, channel subset or
session is removed because it weakens the paper.
