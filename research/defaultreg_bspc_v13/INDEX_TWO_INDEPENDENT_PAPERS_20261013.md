# Default Regularisation: two independent publication tracks
2026-10-13 | Evidence register | neither manuscript is submitted.

## Paper A — primary BSPC research article
Provisional title: When Predictive Fit and Discriminative Transfer
Disagree: The Geometry of EEG Regularisation Selection.
Canonical editable LaTeX:
research/defaultreg_bspc_v13/main_bspc_v31_EVIDENCE_GATED_CANDIDATE.tex
Aims to establish validation estimand mismatch and how a proper
source-predictive log score can be dominated by AUC-invariant
individual location/orthogonal class-contrast shifts. Source-only
three-acquisition postoutcome mechanism evidence:
Cho2017 / Dreyer2023A / native BNCI2014_002, 38 total source
validation groups, SHA-audited. Distinct historical frozen source
selection target gains and negatives (EEGMMIDB/Shin/BNCI/Weibo,
Cho40, Dreyer12) retained with information budget markings.
GitHub draft PR #140. Requires author approval and journal
submission package review, not an already accepted paper.

## Paper B — SEPARATE recording-content reliability study
Provisional title: EEG Recording Identity Beyond Subject Identifiers:
Exact Epoch Overlap and Matched Model--Selection Audits.
Canonical editable LaTeX:
research/defaultreg_bspc_v13/main_bspc_v30_RECORD_IDENTITY_EVIDENCE.tex
It checks exact EEG RECORDING CONTENTS, not physical-patient identity.
Liu2024 released 50 IDs / 2000 processed trial instances but
1960 unique exact processed contents; ID4 and ID5 all 40 epoch
contents match, 49 recording-content groups. Raw EDF identity
partly corroborated on 32 of 33 channels but cannot determine
patient identity. Controls Weibo2014 and BNCI2014_002 have no
cross-ID identical processed epoch contents.
Matched 49-record 2x2 original/purged fitting x frozen lambda:
  0.588673 / 0.587347
  0.580765 / 0.582551
Prior nominal 50-ID mean 0.594000.
Denominator shift, refit and lambda substitution are
distinct descriptive quantities, NOT a causal leakage effect.
Existing reproducible technical note already has these
real measurements and source files; it should not be merged
into Paper A's EEG participant count or Holm families.
Retrospective and not automatically independent validation.
Verify journal fit (data integrity / biomedical informatics)
independently of Paper A before selecting publication venue.

## Paper C — Lee2019 GID falsification SIDE EXPERIMENT, no paper yet
Fixed source and held split and source SHAs:
research/defaultreg_bspc_v13/LEE2019_GID_FROZEN_FIRST_EXPERIMENT_PROTOCOL.md
GitHub Draft PR #141. No independent original-novelty claim.
The source-only signed forecast was NEGATIVE for new GID
(12 source people paired AUC -0.0008); its classifier
direction differs from original (cosine 0.98839).
No held-target success claim can be made until A/B real
results and their immutable audit. This study may only
enter Paper A as a transparent sensitivity/falsification
unless it passes all precommitted performance gates.

## Non-negotiable cross-paper safeguards
1. No physical-patient deduplication conclusion without
   clinical data custodian provenance.
2. No retuning Lee models on held A or B, even after a
   surprising result.
3. No treating 49 recording groups as 49 new independent
   unseen persons in Paper A's confidence intervals.
4. No repeat of universal 4–19x lambda, 98.3% transferable
   class energy or hidden spectral-filter gain previously
   retired for provenance/replication deficiencies.
5. All figures, sources, authorship and ethics/declarations
   must be verified and approved before publisher submission.
