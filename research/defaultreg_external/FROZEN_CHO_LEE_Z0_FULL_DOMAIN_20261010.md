# Default Regularisation — prospective Cho2017 + Lee2019_MI Z0 full-domain external validation

Frozen: 2026-10-10, before any target-participant AUC from these two datasets
is opened by this project. Public execution lineage: draft PR #96.

This document supersedes no earlier provenance. BNCI2014_001 is already exposed
and remains development/post-hoc evidence. The two cohorts below are the first
independent external datasets assigned to the v4 regularisation-validity claim.

## Scientific question

Under a strict zero-target-information budget, does the separation between
source covariance-fidelity and source participant discriminability objectives
replicate outside EEGMMIDB, and does that separation have a consistent
held-participant consequence?

Secondary question: within the covariance-fidelity admissible region, can FCTR
reduce hard-participant risk without sacrificing participant-macro AUC?

The paper is not a SOTA-accuracy contest. Positive evidence must survive
correctly nested source-only scalar tuning, not only Ledoit-Wolf.

## Frozen cohorts and participant blocks

### Cho2017
- dataset: MOABB Cho2017, DOI 10.5524/100295 / original GigaScience paper;
- all 52 dataset participants;
- source block: subjects 1–26;
- untouched target block: subjects 27–52 (26 participants);
- task: left-hand versus right-hand motor imagery;
- one session;
- raw acquisition: 64 EEG channels, 512 Hz;
- common analysis montage: fixed 21 FC/C/CP channels;
- MOABB preprocessing: 8–30 Hz, 0–3 s task epoch, resample 160 Hz,
  reject_by_annotation=False, no additional post-hoc participant/trial
  selection by target outcome.

### Lee2019_MI
- dataset: MOABB Lee2019_MI, original OpenBMI/GigaScience dataset;
- all 54 participants;
- source block: subjects 1–27;
- untouched target block: subjects 28–54 (27 participants);
- task: left-hand versus right-hand motor imagery;
- two labelled training/offline sessions (train_run=True, unlabeled test run
  excluded);
- raw acquisition: 62 EEG channels, 1000 Hz;
- common analysis montage: same fixed 21 FC/C/CP channels;
- MOABB preprocessing: 8–30 Hz, 0–4 s task epoch, resample 160 Hz,
  reject_by_annotation=False, no additional target-dependent rejection.

The 21-channel montage is frozen to match the spatial support used by the
EEGMMIDB primary representation and to avoid confounding cross-dataset
replication with channel-count expansion.

## Target-information budget: Z0

Before target AUC evaluation:
- no target labels select lambda, representation or model;
- no target covariance/mean enters source tangent reference;
- no target batch normalisation;
- no target quality threshold;
- no target-derived feature scaling;
- no target participant is moved into the source block.

Target labels are used only to calculate outcomes for score vectors generated
by already frozen source models.

## Full scalar domain

Fixed grid:

0, .0005, .001, .002, .003, .005, .0075, .01, .015, .02, .03,
.05, .075, .10, .15, .20, .30, .40, .50, .60, .70, .80,
.90, .95, .98, .99, 1.0.

This explicitly includes the near-zero analytic-shrinkage regime and both
domain boundaries. Infeasible lambda=0 folds remain visible rather than being
silently replaced.

## Predeclared methods

Primary scalar references:
1. Ledoit-Wolf LDA.
2. OAS LDA.
3. source covariance-predictive-NLL optimum over the complete scalar domain.
4. source participant-macro-AUC optimum over the complete scalar domain.
5. tangent-space LogisticRegression(C=1).

Secondary conservative intervention:
6. FCTR with:
   - covariance admissibility: best held-source NLL + 1.0×SE at the best NLL;
   - lower-tail fraction alpha=0.25;
   - minimum source lower-tail gain=0.01 AUC;
   - macro-AUC non-inferiority margin=0.01;
   - fallback to covariance-NLL optimum when the intervention gate fails.

Earlier TMS / signed-tail / CFCTS methods are not headline candidates and are
not allowed to replace the above methods after target inspection.

## Primary hypotheses

H1 — source objective conflict:
the complete source path shows a reproducible separation between covariance
predictive NLL and participant discriminability. Report both optima, full
curves, 1-SE overlap/Jaccard, reciprocal regret and boundary hits.

H2 — target consequence:
source-macro full-domain shrinkage has positive paired participant AUC delta
versus source covariance-NLL full-domain shrinkage. This is the main
mechanism-to-deployment test.

H3 — analytic-default consequence:
source-macro full-domain shrinkage has positive paired participant AUC delta
versus Ledoit-Wolf. This is secondary to H2 because LW is not the strongest
fair tuned baseline.

H4 — hard-user safety:
report Q25, worst participant delta, number harmed by >0.02 AUC, and lower
quartile AUC for every method. Mean improvement cannot hide a worse hard-user
tail.

H5 — regularisation validity frontier:
for each target participant and dataset, preserve target AUC along every
feasible lambda. Use source covariance-fidelity regret as the admissibility
axis and quantify downstream AUC envelopes without selecting a target-optimal
lambda.

## Inference

- participant is the inferential unit;
- exact sign-flip when computationally feasible, otherwise deterministic
  Monte-Carlo sign-flip;
- 20,000 participant bootstrap resamples for final manuscript;
- Holm correction within the predeclared H2/H3 family;
- Cho2017 and Lee2019_MI remain separate replication strata;
- cross-dataset summary uses equal dataset weighting, not pooled trials.

Participant bootstrap does not capture source-training uncertainty. A
delete-one-source full reselection/refit sensitivity will be run after the
primary target result is frozen and will be labeled post-primary.

## Outcome-independent stopping rules

A dataset run fails closed if:
- a source/target participant block changes;
- a required fixed channel is unavailable;
- the source/target blocks overlap;
- one class is absent for a participant used in participant AUC;
- a target statistic enters selection;
- result rows are generated from unequal method trial denominators.

Download/preprocessing failures are recorded; participants are not substituted
based on outcome.

## Interpretation ladder

- H1 without H2: objective mismatch exists, but target relevance is weak.
- H1 + H2 in one external cohort only: promising replication, not a general law.
- same H1/H2 direction in both untouched cohorts plus EEGMMIDB S021–S040:
  multi-cohort evidence for the regularisation-validity thesis.
- FCTR is promoted only if its predeclared safety/mean criteria hold; otherwise
  the paper remains a measurement-validity study.
- No result from these offline public cohorts alone licenses an online,
  clinical, rehabilitation-efficacy or universal EEG claim.
