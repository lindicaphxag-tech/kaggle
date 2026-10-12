# V28 EEG recording-integrity corrected first-run audit evidence (2026-10-13)

## Authenticated completed V26 run

- Real, pinned MOABB 1.7.2 Liu2024; original 50 IDs, exact processed EEG identity [4,5]; 49 deduplicated representative recording groups, 1,960 total held trials.
- Each of 49 held group evaluations uses **48 representative source groups** (1,920 source trials) with new source-fitted tangent reference, new spectral LDA fit and three fresh source-only regularisation choices. All source choices SHA-locked before target AUC.
- CI [run 38285928569](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38285928569): successful; immutable real-run artifact **11765005135**.
- Source-choice SHA256: `9ff6b300a2c04b2c521c43ccbde802574f349b2d9dd10a0da65bfd00b927e719`
- Original outcome JSON SHA256: `5cfdc41ca056fa397e792eb4b5d509ced34ff9e07ad42cc2cd68bd7de4d68d8b`

| Frozen method | Original 50 ID nominal LOPO | 49 unique groups, full refit |
|---|---:|---:|
| Source-person macro-AUC shrinkage | 0.5940000000000001 | 0.5825510204081633 |
| Proper source-mean predictive Gaussian | 0.58985 | 0.5813265306122449 |
| Historical class-centred covariance | 0.59305 | 0.5811734693877550 |
| Ledoit-Wolf | 0.59085 | 0.5785204081632653 |
| OAS | 0.5898500000000001 | 0.578265306122449 |

The full-refit source-macro minus proper NLL mean was **+0.0012244897959183595** (descriptive; not new independent confirmation).

## Independent epoch trial grouping

- [V27 trial-by-trial audit](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38287715932): 2,000 epoch instances, 1,960 exact unique contents, one 40-epoch cross-identifier edge between IDs 4 and 5, no further exactly shared partial epochs.
- [V28 independent graph reconstruction](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38288719062): 49 exact-epoch-safe held groups checked against V27 immutable artifact. Output SHA256 `5b50866f464a4d8f272ed3f90e0e6faac29c479f58008fbef983448242e8d484`.

## Important prohibitions

- **NOT an 0.01145 AUC causal leakage penalty.** The 50→49 difference combines old denominator weighting, altered source folds, refitted geometry, different regularisation selection, and potential fitting interactions.
- **NOT strict physical target-isolated Z0.** Duplicate grouping required reading held target EEG signal contents; this is a retrospective post-outcome data release integrity sensitivity analysis.
- Do not call 49 unique *processed EEG contents* 49 proven distinct patients.
- Do not assert V26 fully matches all V27 epoch-safe graph splits until the separate immutable-artifact crosscheck finishes; original run has already passed the internal V26 group tests.
- Do not call LOPO samples independent replications or claim adjusted statistical significance.

An independent full three-way A(original 50 ID) → B(delete ID5 reporting only) → C(complete 49-source-refitted) audit is submitted in V28 CI. Its numeric decomposition is not included until that run has succeeded.
