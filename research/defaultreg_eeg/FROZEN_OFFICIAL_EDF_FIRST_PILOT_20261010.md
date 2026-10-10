# Default Regularisation — official EEGMMIDB strict-source feasibility freeze

Frozen before the first public real-EDF Actions run on this branch.

## Scientific question

Can a **source-only discriminant regularisation objective** select shrinkage
strengths that are more robust to unseen participants than estimators whose
objective is covariance estimation risk?  This pilot does not claim that the
answer is yes.  It is a falsification gate.

## Immutable first-run pilot design

- Dataset: PhysioNet EEG Motor Movement/Imagery Database v1.0.0,
  DOI 10.13026/C28G6P, ODC Attribution 1.0.
- Source participants: S001-S008.
- Held-out participants: S009 and S010.
- Runs: R04/R08/R12 (imagined left/right movement cues T1/T2).
- Sensors: fixed 21 FC/C/CP channels in the runner.
- Epoch: 0.5 to 3.5 s after each original T1/T2 event.
- Frequency: one fixed 8-30 Hz band only.
- All original task cues are retained; BAD annotations are not permitted to
  silently delete task trials.
- Geometry: arithmetic source-training SPD mean -> whitening -> matrix log ->
  231D tangent upper triangle. Every source-inner validation fold refits the
  geometry without the validation participant. The final geometry is fit on
  S001-S008 only. No held-out covariance enters reference estimation.
- Final classifier family is shared linear discriminant scoring.

## Same-data, same-fold arms

1. LedoitWolf covariance shrinkage LDA.
2. Oracle-free OAS covariance shrinkage LDA.
3. Ridge shrinkage lambda selected by source-participant LOPO macro AUC.
4. Same lambda grid selected by mean signed standardized cross-participant
   margin.
5. Same grid selected by squared standardized margin (polarity-blind arm).
6. Same grid selected by lower-tail signed standardized margin (lowest
   ceil(25% * source participants)).
7. LogisticRegression(C=1) on exactly the same source tangent features.

Grid is fixed at {0.2, 0.4, 0.6, 0.8, 0.95}. No target labels, features,
moments, quality statistics or AUC values may choose an arm or lambda.

## First-run outputs that must remain public

Per original EDF file: SHA256, byte count and task-cue count.  Per held-out
participant: trial count, label count and AUC for every arm. Source-only
validation criterion by lambda; analytic shrinkage values; all-trial
covariance NPZ SHA256; environment/workflow URL. Both success and failure are
retained.

## Hard interpretation rules

- Two held-out people are a **feasibility pilot**, not population evidence,
  statistical significance, independent cohort replication or a journal win.
- Do not average trial observations as if independent study units.
- A new selector is not "better" from 1/2 or 2/2 favorable targets.
- If either held-out subject loses original cues, the run fails.
- If the new signed/tail arms do not materially beat the strongest simple
  baseline, preserve the negative result and do not tune on S009/S010.
- Any algorithm change after opening first-run S009/S010 outcomes is post-hoc
  development and must use different held-out participants for evaluation.
- Historical 102-person transductive AUC is never reused as confirmation.

The next gate after this pilot is a pre-registered subject block on additional
EEGMMIDB participants and then a genuinely different public EEG cohort.
