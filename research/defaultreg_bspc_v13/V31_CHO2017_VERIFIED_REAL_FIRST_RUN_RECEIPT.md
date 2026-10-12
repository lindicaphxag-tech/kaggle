# DefaultReg V31 — Cho2017 first-run outcome (source-locked; independently audited)
2026-10-13 | RESEARCH DRAFT | Not a significant positive method result

## Verifiable provenance
- Original method/sensors/outcome protocol fixed by commit before target I/O:
  V31_CHO2017_SOURCE_ENVELOPE_FIRST_RUN_PROTOCOL.md.
- First-run workflow 38295151137: SUCCESS, exact 5 mathematical tests PASS.
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38295151137
- First-run immutable artifact ID 11768176859, with source lock SHA256
  677f848de0e3bc3122892e5bec55b20dc7808ab04093bdaeb33f1bd0a923f796.
- Separate no-rerun independent audit workflow 38295940609: SUCCESS;
  re-read original source and outcome JSON, recomputed 20 source/target
  IDs, all seven macro AUCs, signed deltas, harms, and ALL 2^20 exact
  sign-flip references from original participant rows.
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38295940609
- No target read before source SHA lock in runner; completed 12 fixed source
  subjects 1..12, 20 held target subjects 13..32; 200 trials each
  = 4,000 target EEG trials; MOABB 1.7.2, 3 fixed C3/Cz/C4 channels.
- This is repository-committed first-run, not formal independent preregistration.
  Physical recording identity is not inferred merely from IDs.

## Pre-declared method and source decisions
The new source-only contrast ellipsoid criterion with radius=1 selected
lambda=1.0; the EXACT original grouped source-macro 27-grid chose
lambda=0.5; correct source-mean Gaussian predictive NLL chose 0.1;
historical validation-class-centred covariance chose 0.0;
the geometric rho=0 ablation chose 0.3.
ALL source choices were frozen before any held EEG outcomes.

The source-only conditional pairwise-dominance certificate for SCE against
original source-macro was FALSE:
nominal standardized margin difference -0.01140698388368339;
worst-case difference -0.05074055361132672 at rho=1.
This means the geometry supplied NO certified superiority guarantee,
despite SCE having a higher individual worst-case margin criterion.
This conditional theorem is not a real-target EEG coverage guarantee.

## All target macro AUC results
| Source-only or analytic policy | Held-person macro AUC |
| --- | ---: |
| SCE robust source contrast envelope, rho=1 | 0.618230 |
| Original grouped source-macro 27-lambda grid | 0.614835 |
| SCE no-shift ablation, rho=0 | 0.610300 |
| Correct source-mean Gaussian predictive NLL | 0.601830 |
| Analytic OAS | 0.593580 |
| Analytic Ledoit-Wolf | 0.593535 |
| Historical validation-class-centred covariance | 0.593355 |

The single predeclared primary comparison, SCE-rho1 minus source macro,
has a participant-macro paired mean +0.003395 AUC, exact sign-flip
enumeration reference p=0.09308433532714844 (two-sided); the
20,000-draw seeded subject bootstrap descriptive 95% interval of paired
mean AUC difference is approximately [-0.000295, +0.006975] and
crosses zero. It FAILS
the predeclared p<.05 positive promotion threshold. Counts: 14/20 improved,
6/20 harmed, with 0 above +.02 benefit and 0 below -.02 harm;
max participant effect +.0155, min -.0109. All 20 are retained.

The sign-flip number enumerates all 2^20 signs for fixed paired
differences; its interpretation as a level-controlled null test
additionally requires the usual sign-symmetry/exchangeability
assumptions. This does NOT become a confirmatory claim or demonstration
of clinical utility. The one-acquisition, same-3-channel classical
shrinkage comparison cannot be marketed as broad EEG SOTA.

## Frozen individual paired AUC changes (SCE minus original source macro)
| Held Cho2017 ID | Paired AUC delta |
| --- | ---: |
| 13 | +0.0135 |
| 14 | -0.0024 |
| 15 | -0.0109 |
| 16 | +0.0103 |
| 17 | +0.0052 |
| 18 | +0.0016 |
| 19 | +0.0023 |
| 20 | +0.0043 |
| 21 | +0.0155 |
| 22 | -0.0078 |
| 23 | +0.0144 |
| 24 | -0.0090 |
| 25 | +0.0134 |
| 26 | +0.0066 |
| 27 | +0.0106 |
| 28 | +0.0035 |
| 29 | +0.0049 |
| 30 | -0.0073 |
| 31 | -0.0089 |
| 32 | +0.0081 |

## Scientific decision and paper separation
Outcome = SMALL POSITIVE / NON-SIGNIFICANT / ZERO MATERIAL HARM IN THIS BLOCK.
The new lambda=1.0 boundary saturation is a failure-mode investigation,
not evidence of a universally better shrinkage value. No hyperparameter
is permitted to be tuned on persons 13..32.
SCE remains a pre-submission methods candidate pending truly separate
acquisition and strong matched-budget classifiers. It does NOT replace
the existing DefaultReg BSPC criterion-mismatch paper's established
prospective and negative results.
