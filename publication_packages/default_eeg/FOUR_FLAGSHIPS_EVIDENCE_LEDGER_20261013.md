# Default EEG four flagship research tracks — evidence ledger (2026-10-13)

**All papers remain DRAFT / NOT SUBMITTED.**
A manuscript standard or a locally passing program is not evidence
of independent transfer benefit, journal acceptance or novelty superiority.

## Branch heads reviewed and pushed
- P1 JNE PR #143, head d18415349974afe32c5c7a6a2c58704bd701bc6c
- P2 BSPC PR #144, head fb2c85f889ebe2f7a728fb0f5ced014b8ce8fc60
- P3 PLOS ONE PR #145, head e1279e5882b3c9712aacd8f5d43476d8370887dd
- P4 Numerical Algorithms PR #146, head 29d5f83feb7a5b2add3c637dbf52837d1bd75a6d

## What this review actually adds

### P1: a hard trivial-baseline falsification on EXPOSED REAL EEG\n- Manuscript Results now explicitly replace misleading unqualified\n  covariance-NLL references with historical class-centred CovScore;\n  this is NOT a proper source-mean predictive Gaussian likelihood.
- From frozen BNCI2014_001 9-person second-scorer JSON, all nine
  source forecasts were positive; 7/9 observed deltas positive.
- The advertised 7/9 sign accuracy **equals always-positive**.
- MAE forecast .015380524477 vs delta=0 baseline .017698688272:
  descriptive reduction .002318163795 AUC.
- Paired exact two-sided sign-flip enumeration p=.3203125,
  exploratory ONLY due outcome exposure and shared LOPO source folds.
- Script + source-frozen result path:
  research/defaultreg_bspc_v13/src/p1_exposed_forecast_trivial_baseline_audit.py
- **OPEN:** preregistered data-unseen acquisition level prediction and
  second independent model scoring; strong source-only baseline benefit.

### P2: signal-product transform stress, NOT patient identity
- 120 synthetic null calibration pairs and 120 disjoint synthetic test
  acquisitions for each of six known transformations.
- New directed channel-rowspace containment: 120/120 channel
  permutation/gain positive associations vs 9/120 flat Pearson
  after fitted synthetic null thresholds. In test nulls 2/120 false
  links for BOTH metrics; no claimed zero-FPR detector.
- Real 6x6 MAT-to-EDF source ranks are still post-discovery:
  EDFs 4/5 choose MAT4, 11/15 choose MAT11, 47/49 choose MAT47,
  but no source conversion log or independent acquisition genealogy.
- Identified NEMAR nm000158 v1.0.3 (2026-08-18) as a separately
  hosted version of the same Liu2024 source, NOT new EEG people:
  publication_packages/default_eeg/P2_NEMAR_FIGSHARE_CROSS_DISTRIBUTOR_GATE_20261013.md
- **OPEN:** MAT->EDF documented pipeline, channel/time map, independent
  held-out known-origin positive/negative cohorts, data steward clarification.

### P3: assumption-explicit acquisition-level conformal prototype
- A FIXED policy and source-only margin forecast are chosen before
  calibration. The one-sided split-conformal residual rank gives a
  marginal lower bound under exchangeability of whole acquisitions.
- Synthetic 79-calibration / 2500 fresh exchangeable trial: empirical
  90.64% bound coverage at alpha=.1. Deliberately nonexchangeable
  test-law shift: only 3.0% coverage. Neither result concerns real EEG.
- Under strict Z0, the SAME known source pool implies the SAME policy
  for unknown targets: no target-specific adaptivity.
- **OPEN:** sufficient genuinely independent meta-acquisition source
  calibration units and at least two unopened target acquisitions.
  The four original operators are not yet a common confirmatory trial.

### P4: fixed-pole rational coefficient-uncertainty certificate
- For fixed exact positive poles, bounded score coefficient error
  yields robust sign polynomials N-E and -(N+E). Strict positive
  Bernstein coefficient vectors are sound sufficient certificates;
  any undecidable sign returns UNKNOWN.
- Local test: 96 fixed-seed synthetic rational cases, 31 POS,
  39 NEG, 26 UNKNOWN, 3,120 sampled uncertainty-corner checks.
  Corner checks are regressions; mathematical soundness follows from
  convex-hull enclosure, not sampling.
- The exact finite-rational prototype is pushed with a GitHub CI
  workflow and integrated P4 manuscript section.
- **OPEN:** IEEE754 outward rounding for the full signal pipeline,
  eigenvalue and eigenvector errors, strong end-to-end comparison on
  independent high-dimensional source tasks, real decision value.

## CI evidence restrictions
- Older generic GitHub PDF workflows may complete; they do NOT by
  themselves verify each newly created method workflow or prove its
  data acquisition independence.
- New method/CI file commits are verified by PR branch heads; new
  method CI SUCCESS is NOT claimed unless its own completed run is
  separately confirmed.
- Local P2/P4 tests are executed; P1 frozen-scalar calculations
  are independently reproduced from GitHub JSON; P3 synthetic
  coverage numbers are independently checked.
- No full latest HEAD PDF layout inspection, new held-out EEG dataset,
  custodian reply, coauthor approval, or real journal submission.

## Submission decisions (2026-10-13)
- P2: the closest paper to a careful limited-scope data-integrity
  submission, but it does not yet support a general automatic
  provenance classifier or identity claim.
- P1: valid estimand-accounting manuscript direction, predictive
  algorithm headline not justified.
- P4: defensible conditional mathematical prototype; not yet strong
  Numerical Algorithms empirical/numerical assurance.
- P3: exploratory negative studies and a synthetic assumption-check
  prototype, not yet an independent full PLOS ONE study.

All four have a separate research endpoint. Outcome reuse across
papers and acquisition-level dependencies MUST be disclosed.


## 2026-10-13 subsequent joint-uncertainty and external-manifest audit
- P4 #146 head 29d5f83feb7a5b2add3c637dbf52837d1bd75a6d.
  New exact-rational *joint* positive spectral-eigenvalue BOX and
  score-coefficient BOX Bernstein certificate, not simply fixed spectral poles.
  A local standalone regression checked 126 synthetic boxes, outcomes
  52 POS / 50 NEG / 24 UNKNOWN, with 208,560 exact rational
  uncertainty-corner/parameter point checks. The mathematics is a
  sufficient interval enclosure; corner samples are not proof.
  New script: research/defaultreg_bspc_v13/src/
  p4_joint_spectral_coefficient_box_certificate.py
  and new TeX:
  P4_joint_spectral_coefficient_box_scope_20261013.tex.
  Real EEG input-error bounds and external numerical superiority
  are NOT verified. The exact new CI job is NOT individually
  confirmed just because generic repository workflows succeeded.
- P2 #144 head fb2c85f889ebe2f7a728fb0f5ced014b8ce8fc60.
  Audited version-pinned NEMAR nm000158 JSON manifest at external
  commit 11e3eeb4b439d5f8ee1b5e3b8e6032824bc2de83.
  It states 50 whole EDF files imported via MOABB 1.5.0 on
  2026-08-11 and 50 distinct recorded SHA-256 whole-file digests.
  Published pairs 04/05, 11/15, 47/49 have distinct
  full-file digests. This does not contradict their numeric
  within-file channel similarity, does not independently
  hash raw downloaded file bytes, and does not establish
  raw MAT ancestry or clinical identity.
  Metadata-only audit:
  research/defaultreg_bspc_v13/src/p2_nemar_release_manifest_audit.py.
  Embedded P2 manuscript paragraph:
  P2_nemar_whole_EDF_manifest_scope_20261013.tex.
- P1 #143 and P3 #145: previous limitations unchanged,
  and no new prospective independent-acquisition data.
- All four remain Draft; author declarations, PDF HEAD/layout
  verification and release decisions are not closed.

