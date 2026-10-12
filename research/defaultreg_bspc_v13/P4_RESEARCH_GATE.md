# Paper 4: Polynomial Rank-Cell Certificates (technical note candidate)
Status: DRAFT / DO NOT SUBMIT.
Manuscript: `research/defaultreg_bspc_v13/main_p4_rank_cell_certificates_NOT_READY.tex`
Methods & real data result:
`P4_rank_cells_method.tex`, `P4_rank_cells_real_source_result.tex`.

## Distinct scientific endpoint
P4 is about certification COVERAGE of stable source
positive-negative rankings over a continuous scalar
regularisation path. Its primary endpoint is the
percentage of source label pairs whose ranking is
provably fixed by a polynomial bound on an interval.
It does NOT use P1's predictive-Gaussian density
difference; P2's record-level duplicate or matched
refit; or P3's held-target AUC comparison as its
primary result. Reuses Weibo2014 source validation
folds: overlap must be explicitly disclosed.

## Verified real numerical endpoint
Original GitHub workflow run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38283087191
Artifact ID 11762626131; paper receipt `117_v25_real_source_bernstein_result.tex`.
- Full [0,1] source rank-stability coverage: old 94.121% vs
  new Bernstein 94.267%, +0.146 percentage points.
- Between source-selected endpoints: 96.274% vs
  96.338%, +0.064 percentage points.
- 9 source-validation groups, SIX tangent coordinates, one
  previously exposed EEG acquisition: not prospective target
  validation. There is no improved held AUC.
- Rank-cell numerator has degree <=p-1 on SPD-feasible scalar
  intervals, a standard rational-polynomial fact.
- Implementation uses heuristic numerical cancellation margins,
  not formally rounded machine-proof certification.

## Evidentiary promotion gates
[ ] New independent source-domain acquisitions and p>6 dimensions.
[ ] Bench accuracy/complexity vs root isolation, Taylor source
    envelope, dense grid and floating-point stress tests.
[ ] Outward-directed rounding if using formal certificate language.
[ ] Meaningful downstream scientific use of rank coverage
    (e.g. identical vs changed selection regimes in new sources).
[ ] Clear P1/P3 nonredundant claim and overlap disclosure.
[ ] Actual authors' declarations and journal release materials.

A niche short methods note may eventually fit a computational
signal-processing or numerical methods outlet, but a source
certificate increase of 0.146 pp is too small to assume
journal-level superiority or rapid acceptance.


## PROVISIONAL OUTLET ONLY: Digital Signal Processing
Elsevier; 2025 legacy CAS large category Engineering & Technology 3;
hybrid subscription mode can avoid open access APC (verify final
author-guide for ancillary fees). Publisher scope explicitly covers
statistical signal processing and mathematical tools with
signal-processing application:
https://shop.elsevier.com/journals/digital-signal-processing/1051-2004
However the current P4 is only ONE SOURCE EEG collection, a small
+0.146 percentage-point interval-certificate coverage improvement
and unverified rounding. It is **NOT a submission-ready DSP method
paper**. Substantive numerical method novelty, runtime savings,
multiple signal families and reproducible independent comparisons
are required. Outlet not locked until promotion gates pass.
