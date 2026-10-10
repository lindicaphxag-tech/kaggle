# Default Regularisation — PUBLIC CFCTS prospective S011–S020 freeze

Frozen on 2026-10-10 before any S011–S020 execution and while the earlier
S009/S010 public pilot was still in progress.

## Fixed code identity

Prospective public strict-inner-geometry CFCTS runner commit:
9263fb7a97a47e6a2410b3f33e72910286ed03ee

The CFCTS selection logic is now evaluated inside exactly the same
leave-one-SOURCE-participant folds used for other source-only shrinkage
criteria. Each inner fold refits the SPD geometry without the held source
participant.

## Data split

Source only: S001–S008.

Earlier development/elimination targets: S009–S010. They are excluded from
this prospective target block and are not added to source fitting.

Prospective targets: S011–S020.

Runs: R04/R08/R12.
Task: T1 vs T2 motor imagery.
Band: fixed 8–30 Hz.
Sensors: fixed 21 FC/C/CP channels.
Epoch: 0.5–3.5 s after every original task cue.
Dataset: official PhysioNet EEGMMIDB v1.0.0, DOI 10.13026/C28G6P.

## Fixed methods

- LedoitWolf LDA
- OAS LDA
- source-LOPO macro-AUC shrinkage
- source signed-mean shrinkage
- source squared-margin shrinkage
- source lower-tail signed shrinkage
- CFCTS: one-standard-error covariance-NLL fidelity envelope, then
  lower-tail signed-margin maximization inside the admissible set
- LogisticRegression on identical final source tangent features

Grid: 0.2, 0.4, 0.6, 0.8, 0.95.
Tail fraction: 0.25.

No target label, target covariance, target batch mean, target quality
distribution or target performance may affect any model, geometry, lambda,
threshold or method.

## Primary falsification

The independent evaluation unit is participant.

Report all 10 per-target AUC vectors. Primary method contrast is CFCTS minus
the strongest *same-information source-only classical comparator* across the
frozen set; all pairwise method results remain visible to prevent comparator
hindsight.

Required harm outputs:
- number of participants with CFCTS delta < -0.02;
- median and lower-quartile per-participant delta;
- worst participant delta;
- comparison of covariance-NLL-best lambda, unconstrained tail lambda and
  final admissible CFCTS lambda.

No trial-level significance testing.

## Release rule

- If CFCTS fails to improve both mean participant AUC and lower-tail behavior,
  demote CFCTS from the primary algorithmic claim.
- A positive S011–S020 block is still same-dataset evidence only.
- No BSPC superiority claim before two additional public cohorts.
- No TNSRE framing unless later evidence has large participant scale and
  genuine calibration/BCI utility.
- Any algorithm changes after this first S011–S020 result require a new
  target block.

This is a prospective falsification record, not an acceptance or SOTA claim.
