# Four Default EEG manuscripts — verifiable 2026-10-14 update

All four PRs remain **DRAFT**, not journal-submitted or accepted. All
claims below are limited to original executable receipts, exactly stated
data levels, and independently inspectable git branches.

## P1 — JNE (#143)
- Actual real AlexMI 8-point source-locked pilot:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38389770062;
  two held people (IDs 7/8), same named public acquisition.
- New original **27-point FULL_GRID** retrospective rerun:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38536408103.
  Same two *already exposed* held people.
  Macro AUC 1.000/0.995; true source-fitted-mean Gaussian predictive-NLL
  AUC 0.995/0.8075; mean paired macro minus NLL **+.09625 AUC**.
  Macro chosen λ=1, NLL chosen λ=.20. Fixed lambda=.1 and fixed
  tangent logistic included. Two source-only signed forecasts positive,
  so an always-positive direction rule also scores 2/2.
- Strengthened comparability, NOT a new blinded replication, predictive
  utility result or patient-specific clinical benefit.
- Added TeX `P1_alexmi_matched_27point_grid_retrospective_20261014.tex`.

## P2 — BSPC (#144)
- **50/50 EDF public file SHA-256 checks + all 1,225 cross-ID pairs**:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38388066957.
  Named EEG exact digital product statistics:
  4/5 = 29/29 complete streams and 9280/9280 EEG record blocks,
  47/49 = 6957/9280 exact blocks but no complete stream,
  11/15 = 171/9280 exact blocks but no complete stream,
  1,222 others zero exact aligned blocks; CPz constant excluded.
- **Original Figshare v5 author code.zip inspection**:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38535556189.
  Public article metadata enumerates 14 release files;
  code.zip file 42305226 is 20,811 bytes with SHA-256
  7c48ff5304f909927ae32e8fcbfab0816f02481aea19995d88bbe123b34e65f3;
  exactly 21 MATLAB source files, 2 electrode-location files, README.
  Static source is analysis/filter/feature extraction/plots; it does
  **not** directly supply a verifiable original raw MAT -> preprocessed
  EDF export provenance log, manual artefact intervention list or
  physiological patient identity map. This is an evidence gap, not
  misconduct nor proof that no code exists elsewhere.
- New externally reviewable **MOABB upstream issue #1248**:
  https://github.com/NeuroTechX/moabb/issues/1248.
  This is a REQUEST for upstream verification / safe split warnings;
  not a confirmed maintainer finding, merged PR or academic acceptance.
- New manuscript `P2_figshare_v5_author_code_static_provenance_gap_20261014.tex`.
- Despite code and artifact strength, original acquisition MAT->EDF
  transformation provenance not proved. Avoid clinical identity claims.

## P3 — PLOS ONE (#145)
- Original held motor-imagery DreyerB IDs69/70:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38390737635.
- FIRST attempt at six additional held IDs71..76 under original frozen
  byte-identical 8-source features failed closed:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38536634267
  (overstrict source lock keys) and
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38536886941
  (source covariance SHA drift). No new targets opened on failures.
  This is a genuine replay gap, NOT a coding-only success.
- Lower-tier exact original hyperparameter-choice lock, re-fitted
  source model, with source byte drift explicitly recorded:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38537617142.
  Six held DreyerB participants IDs71-76 x 240 trials; one dataset
  collection. Mean paired ΔAUC: SCE -0.00689815, anisotropic
  -0.00361111, group-centred -0.0009375, continuous 0.
  New source SHA differed from archived original, so this cannot be
  described as an identical fitted-model replay.
- Other prior real *mental task* target BNCI2015_004 held ID9 shows
  SCE +0.009531 AUC, but only one held person and not motor imagery:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38390289317.
- New manuscript
  `P3_dreyerb_six_new_locked_choices_source_drift_20261014.tex`.
  No new method-wide or cohort-wide superiority.

## P4 — Numerical Algorithms (#146)
- Existing conditionally sound exact-rational spectral interval and
  covariance-coordinate certificates already validated; the
  FINAL-FEATURE quantisation experiment is not a proof of the full
  MNE/EEG recording pipeline.
- New **real final tangent-feature** exact bit-precision curve:
  https://github.com/lindicaphxag-tech/kaggle/actions/runs/38537814266.
  Fixed 12 sample pairs (same prior examined BNCI2014_001 released
  participant 1); 2 and 4 fractional bits certify 0/12, 6 bits
  certify 5/12, each of 8/10/14/18 bits certify 12/12.
  Every binary64-to-rational quantisation feature error is verified
  exactly and all certificates apply to the full λ interval [0,1].
  UNKNOWN indicates loss of a sufficient certificate, **not**
  observed ranking errors or an EEG classification failure.
- New manuscript
  `P4_real_EEG_precision_budget_certification_curve_20261014.tex`.
  Need true IEEE input-to-final-feature numerical error modelling,
  dimensions + comparators, and novelty proof for Numerical Algorithms.

## Source-control and PDF gates
- Four exact-head LaTeX builds run on main .tex entry points:
  P1 `main_p1_estimand_mismatch_EVIDENCE_GATED.tex`,
  P2 `main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`,
  P3 `main_p3_source_only_optimisation_NEGATIVE_UNREADY.tex`,
  P4 `main_p4_rank_cell_certificates_NOT_READY.tex`.
- At writing these new 2026-10-14 source updates are pushed;
  successful exact-head PDF builds must be checked per latest commit
  individually, including pages, overfull boxes and actual render.
  An older valid PDF cannot be renamed 'latest' after a newer source
  commit. Author signoff, declarations and editorial submission open.

## Realistic completion gate
**Not yet achieved:** four independently confirmed doctoral-level
original mechanisms, four well-powered external replications, four
top-tail accepted-journal comparisons, journal submissions or
acceptances. CI compilation and repo-contained evidence cannot create
outside peer-review or source-custodian disclosures.
