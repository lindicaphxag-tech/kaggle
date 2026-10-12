# PAPER 1 — Estimand mismatch and exact score geometry
Status: Independent research article draft. **NOT submitted / NOT accepted**.
Date: 2026-10-13.

## Primary claim
Historical validation-class-centred covariance log score (CovScore)
and a proper predictive log score using source-trained class means
are different. A correct predictive Gaussian likelihood can heavily
penalize source participant location/score-orthogonal contrast
variation invisible to a FIXED classifier's per-participant AUC.
The paper tests the consequence for EEG shrinkage selection while
preserving all positive, null and harmed-person results.

## Manuscript source
`research/defaultreg_bspc_v13/main_p1_estimand_mismatch_EVIDENCE_GATED.tex`
and `00_abstract_p1_estimand_jne_style.tex` (same source directory).
Bibliography: `references_v10.bib`.
Fitted analysis components are shared within this project:
`131_bspc_twofactor_auc_invariant_transfer_mechanism.tex`,
`132_bspc_3acquisition_source_only_anatomy_results.tex`.

## Exclusivity vs papers 2 and 3
- P1 owns the Gaussian validation estimand distinction,
  exact within-person AUC-invariant three-factor score
  decomposition, 38 source group real-EEG anatomy and
  the historical frozen EEGMMIDB +0.0561 macro AUC contrast.
- P2 owns DATA PROVENANCE and exact duplicate recording
  identification, Liu2024 recording-group purging and matched
  refit-selector 2x2; it does not claim P1's algorithm result.
- P3 reserves nonpositive continuous/SCE/GID source-only
  transfer policy outcomes. Some P1 negative source-selection
  controls overlap P3's datasets/rows: P3 must not be
  simultaneously submitted with redundant outcome claims.
  P1 references those failures only as validity boundaries,
  not a separate algorithm innovation.

## Evidence verified from existing Actions
- 3 EEG acquisition source-only score anatomy independent
  audit: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38304579126
- Existing paper 18-page compile before new journal-specific
  abstract: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38304779662
- Comparison to recent published BSPC estimand/method papers
  contained in `134_bspc_accepted_head_to_head_evidence_and_budgets.tex`.
- Original published output from prior experiments is NOT the
  same as a preregistered new acquisition.

## Journal positioning: choose ONE at actual submission
- Journal of Neural Engineering: scope fits neural signal processing
  and has 'Paper' vs shorter 'Note', Objective/Approach/
  Main results/Significance abstract and subscription no APC.
  Does NOT accept unvalidated SOTA claim. Publisher official:
  https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
- Brain Topography: EEG functional/neurophysiological features,
  but the manuscript needs more neural interpretation. No
  acceptance guarantee. Check original 2025 CAS partition through
  institution, not unaffiliated 2026 'new partition' vendors.

## Required submission gates
[ ] Paper-specific research question and hypotheses are clear.
[ ] Each abstract/table result traceable to source artifact and code SHA.
[ ] Original 21-channel EEGMMIDB comparisons not conflated with
    Cho/Dreyer 3-channel or BNCI 15-channel source anatomy.
[ ] Zero held-target moment access distinction explicit.
[ ] Strong related-work peer-specific novelty boundary.
[ ] No causal/general predictive benefit claim from deterministic
    score geometry.
[ ] Journal styling, tables, figures, source archive and code/data DOI.
[ ] All authors verify title, authorship, disclosures, licenses,
    funding, ethics, data availability, and AI-use declaration.
**DO NOT upload without author approval.**


## CURRENT PRIMARY TARGET (2026-10-13): Journal of Neural Engineering — JNE
Publisher IOP; 2025 legacy CAS large category Medicine 3; neither JCR
nor unofficial 2026 New Journal Partition is substituted for 2025 CAS.
Subscription-access publication **no publication charge** per IOP.
Official journal statement:
https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
Manuscript content must stay estimand/EEG decoding focused, present
traceable full source+target effects and the source-only negative
controls. JNE structured Objective / Approach / Main results /
Significance abstract is already drafted in this P1 branch.
Alternative only after editor/author reassessment: Brain Topography
(Springer, legacy CAS 3, subscription route normally no APC), but it
needs deeper sensorimotor physiological interpretations.
**JNE is the one selected PREPARATION target, NOT a submission.**
