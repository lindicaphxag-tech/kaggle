# Default Regularisation — pre-run public EEGMMIDB feasibility freeze

Date: 2026-10-10. Frozen BEFORE executing this branch on any new downloaded EDF.

## Scientific question

Can a fully source-only covariance-regularised tangent-space classifier be
run end-to-end on original public EEGMMIDB files without target-batch
normalisation, target labels, dropped task cues, or post-hoc subject
selection? This first run is a **pipeline and baseline feasibility test**,
not evidence that a new regularizer beats published methods.

## Fixed data

- Official PhysioNet EEG Motor Movement/Imagery Database v1.0.0, DOI 10.13026/C28G6P.
- Source participants S001–S008.
- Untouched heldout participants S009 and S010.
- Runs R04, R08, R12 only; T1/T2 cues only.
- Exactly 21 fixed FC/C/CP channels.
- 0.5–3.5 s after cue; fixed IIR 8–30 Hz.
- Every otherwise recorded cue is retained. BAD annotations are not used as
  automatic deletion. Any lost task epoch aborts.
- Each original EDF byte file is SHA256 logged.

## Source-only representation

- Trial covariance divided by number of time samples T.
- Small deterministic SPD floor.
- Arithmetic source-training covariance reference, inverse-square-root
  whitening, SPD matrix log, 231D upper triangle.
- Final source reference is fitted on S001–S008 only. Heldout participant
  features are transformed one trial at a time with the frozen source
  reference. No target batch statistic is fitted.
- Inner source-LOPO ridge selection refits the geometric reference inside the
  source-training participants of each inner fold.

## Frozen arms for this first public-data run

1. LedoitWolf LDA.
2. OAS LDA.
3. Source-participant-LOPO tuned scalar ridge/shrinkage LDA, fixed
   lambda grid {0.2,0.4,0.6,0.8,0.95}.
4. LogisticRegression C=1, lbfgs.

These are baselines. The new signed/tail transfer selectors remain on the
private research branch until the baseline data pipeline proves executable.
They cannot use these S009/S010 labels for hyperparameter design.

## Outcomes and hard interpretation

- Primary feasibility outputs: number of original source/heldout task trials,
  per-EDF SHA256, source-selected lambda, analytic shrinkage values, and
  per-heldout-subject AUC for all arms.
- S009/S010 are TWO exposed feasibility targets. Their result must never be
  reported as multi-cohort confirmation or used to tune the new criterion.
- Any method improvement seen here is exploratory. The next signed/tail
  algorithm evaluation requires a disjoint frozen target set.
- No p-value with N=2 is used as journal evidence.
- No comparison against published papers' percentages across different
  datasets/protocols.

## Go/no-go

GO to a disjoint algorithm pilot only if authentic EDF download succeeds,
all original cue trials are retained, hashes are preserved, target subjects
are absent from every fit/tuning function, and all four baselines return
finite scores. Otherwise fix only engineering/protocol errors; do not change
the scientific split to obtain a positive result.
