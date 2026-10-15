# P2 BSPC — submission preflight decision, 2026-10-14

**Decision: AUTHOR SIGNOFF HOLD. No journal submission has been made.**

## Verified work this round

- [x] Original P2 evidence-preserving manuscript retained unchanged: `research/defaultreg_bspc_v13/main_p2_eeg_record_provenance_EVIDENCE_GATED.tex`.
- [x] Reduced to a 12-page, source-locked focused candidate: `research/defaultreg_bspc_v13/main_p2_BSPC_FOCUSED_AUTHOR_REVIEW_20261014.tex`. Exact-commit standalone PDF CI [success](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38591854107) at SHA `7328618042453982029589798152ceefa7aee22b`, **zero overfull hboxes**, all citations/ref labels resolved.
- [x] Source-native vectorized 50-by-50 EDF block identity map and 49-target 2-by-2 refit matrix: `P2_BSPC_native_matrix_and_attribution_20261014.tex`. Counts based strictly on prior real public digital EEG receipts.
- [x] Eighteen technical provenance/sensitivity subsections preserved separately under `P2_BSPC_SUPPLEMENTARY_EVIDENCE_AUTHOR_REVIEW_20261014.tex`, pending **successful independent PDF build**.
- [x] Elsevier `elsarticle` preprint-style and numeric-citation manuscript source `main_p2_BSPC_ELSEVIER_AUTHOR_REVIEW_20261014.tex` prepared; native CI still to be verified as successful.
- [x] Five short highlights (all <=85 characters) and corrected evidence-limited cover letter drafted on this branch.
- [x] Numerical/claim check source `src/p2_bspc_submission_numerical_gate_20261014.py` created, with `submission_authorized=false`. Its own CI result must be checked before calling it passed.
- [x] Official corresponding author Junwei Hao (original Scientific Data 2024 paper, DOI 10.1038/s41597-023-02787-8) and email verified from open public sources; a nonidentifying technical clarification email was saved in the account's Gmail Drafts. **It was NOT sent.**
- [x] Existing actual public EDF waveform panel `P2_BSPC_Fig4_true_native_EDF_waveform_panels.pdf` preserved in author-review package for optional separately uploaded publication supplementary figure.

## Numerical integrity: no general leakage correction claim

50 released IDs / 2,000 processed trial instances = 49 full-epoch exact-content groups / 1,960 distinct processed contents in Liu2024. Exhaustive native EDF digital pair census: `C(50,2)=1,225`; 4/5 9,280/9,280 named time-varying EEG storage blocks, 47/49 6,957/9,280, 11/15 171/9,280 and 1,222 other pairs zero under exactly the specified stored channel/record alignment. The 47/49 and 11/15 pairs are **not** complete digital recording aliases.

Descriptive original nominal 50-ID AUC `0.594000`; matched 49-group unchanged-fit reporting AUC `0.588673`; group-purged refit/choice `0.582551`. Weighting effect `-0.005327`, matched full change `-0.006122`, total arithmetic `-0.011449`. Within matched change, overlap-affected group `-0.262500`, other 48 mean `-0.00078125`. None provides patient identity, independently sampled clinical units or a causal population-average estimator.

## Honest open gates

- [ ] Main and SI **and** native `elsarticle` compile successfully on same latest commit, independently visually check finished PDFs and references; verify uploaded figure file matching.
- [ ] Receive/send the data custodian's technical clarification after author review; if unanswered, explicitly say original MAT-to-EDF release mapping is unknown and use only digital-product claims.
- [ ] Freeze clean, durable scripts, lockfile, data distributor checksums, non-identifying scalar results and a source archive or DOI. Do not rehost raw clinical EEG.
- [ ] Confirm all authors' complete identities, consent/order, institutions, emails and contribution CRediT entries, originality and other related submissions.
- [ ] Actual human approval of secondary public-data terms, ethics, funding, conflicts, acknowledgments, AI-assisted writing/research declarations, cover and highlights.
- [ ] Corresponding author approves and performs actual submission through Elsevier; a GitHub PR is not a journal submission.

Publishing guide: https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions and figure standards https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions/artwork-types

**Do not equate a successful PDF build with scientific validation, submission, acceptance, or guaranteed review.**
