# P2 BSPC — author-facing final-submission gate (2026-10-13)

**Status: BSPC research manuscript candidate / NOT submitted.** Main PDF successful after LaTeX repair; publication-stage experimental addenda remain pending.

## Manuscript and independent primary finding
- Journal: *Biomedical Signal Processing and Control* (Elsevier), signal-data quality/reliability rather than SOTA EEG classification.
- Main: `research/defaultreg_bspc_v13/main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`.
- Cover letter: `research/defaultreg_bspc_v13/P2_EDITOR_COVER_LETTER_DRAFT.md`.
- Highlights: `publication_packages/default_eeg/P2_BSPC_HIGHLIGHTS_AUTHOR_REVIEW.txt`.
- Prior validated full PDF: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38318902315 ; new vector figure/latest-head PDF must be checked independently.
- Frozen public pipeline: MOABB 1.7.2, Liu2024, 8–30 Hz, 160 Hz, 29 selected EEG channels. 50 released IDs / 2,000 trial instances / 1,960 unique exact processed epoch contents; released IDs 4 and 5 duplicate all 40 processed labelled trial contents. **This does not prove any physical patient's identity was duplicated**.
- Full source refit 49-group crossed four-cell outcome, all on the same held groups: AA .588673, AB .587347, BA .580765, BB .582551. The change from nominal 50-ID .594000 to 49-group AA .588673 is a separate report-weight change.
- Crucial heterogeneity: one content-overlap held group: -0.2625 AUC; other 48: -0.00078125 average; all 49: -0.00612245. Do not claim universal or causal leakage effect.
- Prior immutable outcomes independently SHA-audited: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38315269388

## Newly commissioned validity checks — DO NOT advertise as completed
1. Two fixed, non-tuned tangent classifier objectives (logistic C=1 and ridge alpha=10), old/purged source fits and EXACT same 49 held contents. Real EEG run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319072438
2. Independent **pyEDFlib** EDF numerical read path for ID4/5 versus negative control ID47/49: workflow `.github/workflows/p2-independent-edf-reader.yml`. Prior run failed because MNE_DATA directory was not created; code/workflow corrected. No independent result is confirmed until successful artifact inspection.
3. Alternate publicly released EEG preprocessing 6–35 Hz / 128 Hz versus old 8–30 Hz / 160 Hz. Same acquisition; NOT another cohort and NOT independent patients. Workflow `.github/workflows/p2-alternative-processing-identity.yml`.

## Mandatory final release gates
- [x] Original exact trial/label signature + 49 full refit comparisons preserved and isolated from P1.
- [x] Synthetic canonical byte-content and source-choice arithmetic checks succeeded.
- [x] Main PDF after original invalid \n LaTeX fixes successfully rebuilt.
- [ ] Latest **vector figure** PDF compile success and inspect full page/figure/citation extraction.
- [ ] Independently inspect raw EDF reader and alternate processing results, report any contradictions.
- [ ] Inspect actual real-EEG two-classifier additional results; only integrate them if successful and reproducible, whether positive or negative.
- [ ] Contact public dataset's responsible custodians for a good-faith technical clarification before publication; **no email has been sent by this assistant**. Share nonidentifying code/aggregate evidence, not public accusations about patients.
- [ ] Obtain permanent code/data archive with exact versions and source integrity receipts; no proprietary/raw patient signal redistribution.
- [ ] Verify human authorship/CRediT/ORCID/affiliation/ethics and licensing/funding/AI use/conflicts and originality/exclusivity.
- [ ] Ensure finished cover letter, Highlights, manuscript source, figures and reproducibility statement conform to Elsevier submission guide.

## Critical interpretation boundaries
One complete cross-ID processed record alias was detected in one affected acquisition; negative checks in two other acquisitions do not estimate an overall EEG dataset prevalence. The 49 held content groups do NOT equal 49 independently recruited people or independent model fits, since source training sets overlap. Artifact-level exact match does not prove acquisition-side cause. Mean AUC contrast is a conditional reanalysis of known data.

The author-approved article can be submitted using the publisher portal only when these gates pass; neither this GitHub PR nor a PDF build is journal submission.
