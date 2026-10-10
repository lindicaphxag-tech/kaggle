# Default Regularisation — prospective untouched subject blocks (frozen before S009/S010 outcome)

Frozen: 2026-10-10 while the first official EDF workflows for S009/S010 were still IN PROGRESS. No S009/S010 AUC was available when this file was committed.

## Purpose

Prevent target-subject cherry-picking and post-hoc tuning. The first two targets S009/S010 are an exposed feasibility/development pilot only. Any method decision informed by those targets may NOT be evaluated on them as confirmation.

## Block A — prospective algorithm evaluation

- Untouched target participants: **S011–S030 (20 participants)**.
- Same PhysioNet EEGMMIDB version 1.0.0, R04/R08/R12, same fixed 21 channels, same epoch and all-cue retention contract.
- Algorithms eligible for Block A are frozen before any S011–S030 target score is opened:
  1. LedoitWolf LDA
  2. OAS LDA
  3. source-participant LOPO macro-AUC scalar shrinkage
  4. source signed-mean scalar shrinkage
  5. source squared-margin scalar shrinkage
  6. source lower-tail signed scalar shrinkage
  7. LogisticRegression
- If engineering changes are required to parse/download EDFs, they may change only data integrity / transport behavior. Any scientific criterion change invalidates this Block A algorithm confirmation and requires a later untouched block.
- Source participants for the first fixed run should be S001–S010 if all ten EDF groups are successfully source-exportable; because S009/S010 become source data only AFTER their exposed target pilot, their labels/features are allowed in training, but the selection rules themselves must remain frozen. For a stricter sensitivity analysis, rerun Block A with source S001–S008 and report both; never select the better source pool after seeing targets.

## Block B — replication and design robustness

- Untouched target participants: **S031–S060 (30 participants)**.
- Preserve untouched status until Block A method and full statistical protocol are frozen.
- Use Block B to test the same sign/direction and hard-subject claims, not to re-optimize tail fraction or lambda grid.
- If Block A fails, Block B remains a falsification/replication set; do not mine it for a new primary algorithm.

## Primary Block-A hypotheses

H1. Source-participant discriminability-selected shrinkage (any predeclared source-only criterion) differs materially from analytic covariance-risk shrinkage on the actual source folds. This is a **mechanism** claim and does not require an AUC win.

H2. The final chosen *primary new criterion* (must be selected before reading S011–S030) has positive paired participant-level AUC delta versus the strongest non-new source-only scalar baseline. If more than one new criterion remains primary, family-wise multiplicity is adjusted and no winner is chosen after target scores.

H3. The primary new criterion does not improve mean AUC by sacrificing the worst-performing participants: report lower quartile / minimum participant delta and number harmed by >0.02 AUC.

H4. Representation interaction (wide 8–30 versus frozen filter bank) is confirmatory only after the filter-bank pipeline itself is frozen without S011–S030 outcomes.

## Statistical units

- Participant is the inferential unit, not trial.
- Report all 20 targets, missing EDF/data failures and exact exclusion causes.
- Paired bootstrap CI and exact/Monte-Carlo sign-flip on participant deltas, with Holm correction over predeclared primary comparisons.
- No significance claim if the execution pipeline loses unmatched task trials or if target batch statistics enter model selection.

## Block-B and journal interpretation

Block A alone is not journal-wide external replication because source and target come from EEGMMIDB. Block B provides internal replication at larger N. At least one **different public EEG dataset** remains mandatory for the full BSPC robustness claim. TNSRE additionally requires practical BCI/cross-device value beyond this scalar-regularisation evidence.

This file is deliberately frozen before the first S009/S010 result and must remain in history even if the initial method fails.
