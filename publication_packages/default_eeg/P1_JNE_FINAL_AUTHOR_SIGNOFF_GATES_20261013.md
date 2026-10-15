# P1 JNE — author-facing final-submission gate (2026-10-13)

**Status: Journal-specific review PDF built; journal NOT submitted.** This is an evidence-gated scientific manuscript, not a promise of acceptance.

## Actual paper
- Target: *Journal of Neural Engineering* (IOP); type: Paper, not clinical algorithm SOTA.
- Main LaTeX: `research/defaultreg_bspc_v13/main_p1_estimand_mismatch_EVIDENCE_GATED.tex`.
- Structured abstract: `00_abstract_p1_estimand_jne_style.tex` (Objective / Approach / Main results / Significance; corrected comparator front-and-center).
- Full latest revised main PDF: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38319187419
- Central contribution: distinguish validation-class-centred CovScore from proper source-trained predictive NLL; exact three-factor penalty and fixed-model within-person AUC-invariance.
- Confirmatory historical EEGMMIDB +0.0561 AUC is versus **historical CovScore**, NOT versus proper predictive NLL. Five persons harmed >0.02.
- Corrected proper predictive comparisons are retrospective: Shin2017A -0.000307 AUC; BNCI2014_002 +0.006819 (interval crosses zero).
- False original filter-bank interaction hypothesis is disclosed: -0.07781 AUC.
- No universal new decoder, externally validated decision trigger, unseen-person calibration saving, or superior proper-NLL algorithm is claimed.

## Evidence release requirements — mandatory human verification
- [x] 2026-10-13 main LaTeX JNE abstract revision committed, PDF CI success.
- [x] Exact-rational score algebra test and source/target chronology represented.
- [ ] Independent prospective EEG data supporting any *operational* policy must exist before claiming one. Present scope is a score-auditing measurement paper.
- [ ] Freeze a release tag or DOI with all code, preprocessing environment, SHAs, person-level tables, and nonredistributed public dataset retrieval.
- [ ] Authors validate provenance of all results; inspect PDF figures, caption legibility, page count and citation metadata.
- [ ] Approve author order, affiliations, ORCIDs, corresponding author, contributions.
- [ ] Approve funding, conflict-of-interest disclosure (including cover letter), data licensing, secondary-use/ethics, AI-assistance disclosure, prior/preprint/related-manuscript statements.
- [ ] Confirm no concurrent duplicate journal submission or double counting of P3 exploratory Cho/Weibo/Dreyer rows.

## Strict publication firewall
P1 owns Gaussian/CovScore estimand and three-factor real source results. P2 owns Liu2024 ID4/ID5 digital duplication and matched 49 recording-group fit-by-selector results. P3 must not recycle P1 negative-control rows as an independent empirical discovery; P4 shares exposed Weibo *source* groups but distinct numerical endpoint. Related manuscripts must be cited/disclosed appropriately.

## Required editorial action
Once all human authors confirm and all hard gates pass: assemble author-approved JNE cover letter, PDF, editable TeX/BibTeX, figure source, reproducibility archive, supplementary negative results, publication statements; corresponding author submits directly through official portal. This project PR, its PDF Action or this checklist is **NOT** a journal submission.
