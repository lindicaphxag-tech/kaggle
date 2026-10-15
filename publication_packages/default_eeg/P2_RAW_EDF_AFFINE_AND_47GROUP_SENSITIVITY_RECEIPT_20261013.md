# P2 new executed public EEG evidence — 2026-10-13

Status: REAL EXPERIMENTS EXECUTED, MANUSCRIPT SOURCE UPDATED; **NOT submitted or acceptance-ready**.

## Research question
Do newly discovered non-identical but exceptionally correlated processed Liu2024 released EEG ID pairs also show digital-level relatedness? And how does an ASSUMED more conservative similarity-based exclusion scenario affect the SAME held-out representative EEG recordings?

## Experimental receipts (do not call these independent cohorts)
- Raw EDF exact-integer streaming affine-moment code:
  `research/defaultreg_bspc_v13/src/p2_raw_edf_channel_affine_fingerprint.py`.
  Success: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38325795037
- Exact-content 49 vs assumed similarity-group 47 real same-held-target logistic/ridge refits:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38323188651
- Full-code method/scientific table:
  `research/defaultreg_bspc_v13/P2_raw_digital_affine_fingerprint_real_results.tex`
  and `P2_real_exact49_vs_assumed_near47_source_refit.tex`.
- Separate two public-cohort aligned whole-record high-similarity NEGATIVE CONTROLS:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38323467687
  Weibo2014: 0/36 shape-compatible ID pairs >=0.95 (9 pairs shape-incompatible);
  BNCI2014_002: 0/91 >=0.95.
  No extrapolated population-level false-positive estimate.

## Verified raw EDF digital-level pairs
| Released IDs | Byte-identical raw channels | Nonconstant raw channels corr >= .9999 | Median raw digital Pearson |
|---|---:|---:|---:|
| 4/5 | 32 of 33 | 31 of 32 | 1.000000 |
| 47/49 | 1 of 33 | 13 of 32 | .999884 |
| 11/15 | 1 of 33 | 13 of 32 | .999854 |
| Control 2/3 | 1 of 33 | 0 of 32 | .013885 |
| Control 6/7 | 1 of 33 | 0 of 32 | -.002453 |

The one equal channel in nonexact pairs may be a constant channel: channel similarity must not be conflated with an extra duplicate recording.
Additional 11/15 and 47/49 pairs have no nonconstant channel with normalized linear affine-fit residual under .001, so "near-perfect whole-record processed correlation" is not "channel-by-channel perfect affine identity."

## Same 47 held-target scenario outcome (already reviewed data)
| Frozen model | Exact-49 source strategy (same 47 held) | Assumed near-47 source strategy (same 47 held) | Difference |
|---|---:|---:|---:|
| Logistic C=1 | .567872 | .556436 | -.011436 |
| Ridge alpha=10 | .570585 | .560000 | -.010585 |

Logistic ID11 / ID47 changes: -.2925 / -.2550; Ridge: -.2975 / -.2525. Other 45 average changes are +.000222 (logistic) and +.001167 (ridge). Rows share overlapping source training. This is a **retrospective assumption-sensitive source refit**, not evidence that 47 patients or 47 physiological recordings exist, nor a causal leakage correction.

## Submission blockers
1. Custodian clarification for all candidate pairs, release conventions and nonconforming EDF Physical Maximum metadata; preserve patient privacy.
2. Independent input decoding/calibration and source provenance, including raw vs processed alignments and trial/window permutations.
3. Stable, immutable release of code, aggregate receipts, exact environmental versions and per-record results; do not republish EEG signal arrays.
4. Confirm latest PDF build at **exact PR HEAD commit**, check figures, bibliography, author declarations and rights.
5. Explain nonconfirmatory design, post-discovery thresholding, single affected acquisition and shared fitting cohorts prominently.

**No patient identity, causal population leakage, cohort prevalence, algorithm SOTA or publishing outcome claim is licensed by these observations.**
