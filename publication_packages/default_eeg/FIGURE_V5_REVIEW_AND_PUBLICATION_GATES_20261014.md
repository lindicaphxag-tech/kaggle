# V5: evidence-first four-paper figure and submission gates (2026-10-14)

**Status: editorial AUTHOR REVIEW; not submitted, not publication-ready.** A separately downloadable V5 local package was created in the current ChatGPT conversation containing seven main candidate figure PDFs, editable SVG/Python sources, four figure-inserted research-review PDF proofs, captions, LaTeX staging snippets, and two PLOS-specific TIFF images. The binary package was **not** checked into GitHub, so no native `\includegraphics` reference to these files can work on a fresh clone yet.

## Publication-native figure placements

| Paper | Core field-native multi-panel figures | Placement | Limit of inference |
| --- | --- | --- | --- |
| P1 / JNE / PR #143 | Full-size six-panel actual MNE public scalp bandpower (μ/β, left/right/difference); separate 4-held-person ΔAUC heatmap and strong-baseline comparison | Results, after existing Fig 1 | Topomap is *not* learned feature attribution, cortical source location or independently verified selector benefit |
| P2 / BSPC / PR #144 | Public EDF native-sample six-panel waveforms, distinct 50×50 released-content identity matrix and 2×2 retrospective refit/selection AUC matrix | After existing Figs 1–3 at exact-content result and matched model outcome | Exact public digital sample overlaps **do not** confirm patient identity or the complete original MAT-to-EDF transformation chain |
| P3 / PLOS ONE / PR #145 | Three-protocol-held-person-policy ΔAUC matrix, plus explicitly analytic (NOT measured EEG) two-dimensional source vs target loss contours | Results + identifiability explanation; each `Fig1.tif`, `Fig2.tif` separate from manuscript caption | 13 people from DreyerB are not multiple independent acquisitions; no proven new transferable policy |
| P4 / Numerical Algorithms / PR #146 | Measured final-feature fractional-bit certification matrix, strict centred-Horner comparator matrix, status of conditional vs unproved error guarantees | Numerical experiment | Final-feature conditional rational guarantee is not an end-to-end IEEE-754 EEG processing proof |

The V5 visual update improves actual observed panel readability over V4, separates EEG and EDF witnesses from small schematic cards, distinguishes missing/n.a. measurements from tested failures, and labels illustrative loss surfaces. Full-figure proofreading is still required at final journal reduction width. Published panels were used as **layout guidance only**, not copied images.

## SHA audit
- P1 source branch `653680b98badef9e3cd7c83a9780fdd9db8165a0`; local available research PDF currently built at **older** `a414b84f06fdf85c19300c83d727ab8512b2754a`. This is **not an exact-HEAD final PDF**.
- P2 `182935889edfbd3c034c9f8f2eaffac2687a4ea1`, P3 `3dee0880e7683d5c55937843867756f9e2459ad8`, P4 `e91433afcb3a11fef0abb3f7dd32890949583d61`: their baseline text PDFs match exact branch HEAD; **V5 figure insertions have not been incorporated into a fresh native LaTeX build of those HEADs**.
- No journal submission, author contribution/funding/ethics signoff, or independent academic validation is inferred.

## Mandatory remaining manuscript/scientific gates
1. **P1**: independent acquisition action-value validation over simple constant selectors + rebuild latest source.
2. **P2**: full acquisition source-to-release audit, no clinical identity extrapolation.
3. **P3**: independently acquired held replication with fixed fitted source state and a falsifiable negative-transfer mechanism.
4. **P4**: a credible end-to-end preprocessing roundoff input budget, strong incumbent methods and novel numerical algorithm advantage.
5. **All four**: authentic authors, affiliations, contribution statement, conflicts, funding, ethics/data permissions, AI disclosure, cover-letter approval, bibliography consistency, true native compilation and editor file checks.

PLOS ONE figure submission requirements: https://journals.plos.org/plosone/s/figures
Springer figure formatting: https://authorservices.springernature.com/figure-and-table-formatting/
Elsevier artwork instructions: https://www.elsevier.com/researcher/author/policies-and-guidelines/artwork-and-media-instructions

**Claim restraint:** neither a high-quality figure nor successful CI is evidence of acceptance or a scientific breakthrough.
