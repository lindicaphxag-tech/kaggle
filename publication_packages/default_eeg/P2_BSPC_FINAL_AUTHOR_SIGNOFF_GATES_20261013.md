# BSPC P2 Final Scientific Evidence and Author Submission Gate — 2026-10-13

**Journal submission: NO.** The scientifically expanded source draft is available at
`research/defaultreg_bspc_v13/main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`.
Published paper, acceptance, and formal Elsevier submission are NOT claimed.

## Independently checkable executed evidence

| Finding | Actual source | Status |
|---|---|---|
| Liu2024 public EEG 50 IDs, 2000 trials, 1960 unique processed labelled trial contents; ID4/5 full 40 trial alias | Original 8–30 Hz/160 Hz MOABB1.7.2 archive | historical executed |
| Same 49 held recording-content groups crossed full model refit by source selection: AA .588673, AB .587347, BA .580765, BB .582551 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38315269388 | PASS |
| Alias-affected content group's matched LDA change −.2625 AUC; other 48 average −.00078125 | same immutable 49-record audit | PASS |
| New real public EEG tangent LogisticRegression(C=1) old/purged mean 0.580867→0.576071; ID4 −0.2125; other48 −0.00046875 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319072438 | PASS |
| New real public EEG tangent RidgeClassifier(alpha=10) old/purged mean 0.582296→0.578827; ID4 −0.2225; other48 +0.00109375 | same run, separate fixed algorithm | PASS |
| Alternative **same dataset** processing 6–35 Hz/128 Hz: complete ID4/5 alias and 49 groups remain, EEG shape 40×29×512 per released ID | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319497449 | PASS |
| Independent raw EDF **digital integer sample payload** reader: IDs4/5 32/33 channels identical byte-for-byte; control IDs47/49 1/33 | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319951161 | PASS |
| Strict independent pyEDFlib physical-calibration reader | https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319368441 | FAIL — original EDF header Physical Maximum noncompliance; do not misstate a completed physical-calibration reproduction |
| Source version including vector workflow diagram and all new real robustness numbers | this PR #144, latest HEAD | PDF latest-head check needed |

**Interpretation**: All new experiments reuse the same Liu2024 public acquisition. The cross-ID signal-content alias and the localized refit effect are confirmed with stronger technical evidence; neither proves duplicate human patient identities, clinical acquisition causes, causal population leakage, or universal AUC correction. The 49 held folds share overlapping source training. Other two EEG collections are negative controls for exact overlap under specified processing, not a prevalence estimate.

## Manuscript files and submission peripherals

- Journal-directed manuscript: `research/defaultreg_bspc_v13/main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`.
- Frozen real additional experiment subsection: `research/defaultreg_bspc_v13/P2_additional_real_multimodel_and_preprocess_results.tex`.
- Raw signal reader explanation: `research/defaultreg_bspc_v13/P2_methods_signal_identity_benchmark_upgrade.tex`.
- Grayscale vector audit flow: `research/defaultreg_bspc_v13/P2_vector_identity_audit_workflow.tex`.
- Edited cover letter: `research/defaultreg_bspc_v13/P2_EDITOR_COVER_LETTER_DRAFT.md`.
- Elsevier-style Highlights: `publication_packages/default_eeg/P2_BSPC_HIGHLIGHTS_AUTHOR_REVIEW.txt`.
- Existing authoritative pre-figure PDF: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38318902315 . **This PDF predates new results**; do not present as the latest final PDF.

## Submission-blocking author decisions and release steps

- [ ] The new latest-head journal PDF finishes with SUCCESS, and full-page figures/table overflow, bibliography, caption, exact numerical provenance, author names and final submission attachments are inspected.
- [ ] Confirm and document contact with responsible Liu2024 public-data custodians (clinical origin of released ID4/5 is **not established**); disclose independently observed noncompliant EDF Physical Maximum header rather than infer source data handling.
- [ ] Stable release archive with all source code, version-locked environment, non-identifying 49-group numeric evidence and SHA-256 receipts; original raw public EEG should remain with its legitimate dataset publisher, not redistributed here.
- [ ] Human all-author approval: authors/order, emails/ORCID/affiliations, CRediT, funding, ethics/secondary-use obligations, licensing, conflict/AI use, related-paper overlap, exclusive submission declaration and corresponding author's portal decision.
- [ ] The corresponding author performs formal online submission to BSPC. GitHub PR / Actions PDF cannot submit to Elsevier.

## Overlap restrictions

P2 alone claims exact ID4/5 content alias and matched 49-group full refit-by-selector analysis. P1 alone claims Gaussian validation-estimand anatomy; P3 and P4 cannot reuse these exact results as separate original discoveries. The 37/67-paper comparison is a mixed accepted-paper sample and is not a calibrated acceptance probability.
