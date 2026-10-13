# P3 / PLOS ONE — conditional source-only transfer decisions, doctoral study hypothesis (NOT executed)
Date: 2026-10-13 | Preserve negative results honestly; do not portray this design as an accepted new algorithm.

## Current empirical state
- Old four proposals on heterogeneous cohorts are NOT repeated tests of the same hypothesis. Weibo continuous vs 27-grid equal; Cho SCE ~-.00005264 over 40 target people (second block sequential); Dreyer selected alpha0 (~-.001088); Lee GID -0.003933 n12 p=.1289 and reserved block aborted before target on source SHA mismatch.
- Same-six-selector exploratory BNCI2014_004 n9 and Weibo2014 n10 (source-macro .778002/.731405). Original four operators were NOT in that identical six-selector matrix.
- More recent single BNCI2014_004 source IDs1-8 -> target ID9 harmonises original four operators. Held target previously exposed; largest positive gain anisotropic +.000401 AUC on ONE subject, not confirmation.
- Fixed-source no-free-lunch sign reversal and total-variation bound are elementary conditional facts; neither is an algorithm nor a distribution-free transfer guarantee.

## Doctoral question with a testable positive mechanism
Can an EEG model-selection SYSTEM trained solely on historical labelled source ACQUISITIONS produce valid abstain/retain/switch decisions for a NEW acquisition *under a prespecified assumption on meta-acquisition exchangeability or bounded shift*, while reducing worst-person harm relative to blindly choosing maximal source macro AUC?

## Proposed mechanism: conditional conformal lower margin & abstention
Meta-training unit = entire independent EEG acquisition, NEVER a single trial or overlapping LOPO fold.
For each historical leave-one-acquisition-out meta-unit k compute only pre-target features z_k (e.g. source-group selector disagreement, nuisance-score discordance, source-LOPO risk dispersion); after each historical held-out target is revealed, record prospective policy-margin y_k = AUC_k(candidate)-AUC_k(default) at the acquisition level.
Fit a simple source-only predictor m(z), with a split-conformal lower residual quantile q calibrated at the **acquisition** level. For a fresh acquisition observable z*, form L(z*)=m(z*)-q; switch only when L(z*)>delta_min, otherwise deploy default or abstain. Freeze z, data split, model, q and selection budget before fresh target labels.
A conformal *marginal* coverage statement requires exchangeability of meta-acquisitions AND the chosen calibration scheme; it does not follow from unrelated EEG subjects and does NOT ensure worst-case clinical safety. Without eligible independent calibration acquisitions, NO formal coverage is claimed; report only descriptive selection.
The old TV inequality can be listed separately as a conditional analysis with EXTERNALLY known eps. Never estimate target TV using unseen target data under Z0.

## Required experiment / failure gates
1. Same four original algorithms + six classical selectors on same target people, same source-fitted geometry, matched compute/time, same feature data. Full source-lock SHA required.
2. Pre-register >=2 genuinely unexposed target acquisitions AND enough independent source meta-acquisitions for nontrivial leave-acquisition-out conformal calibration. If insufficient, choose transparent descriptive negative-result paper rather than advertise conformal confidence.
3. Hierarchical primary endpoint: acquisition macro margin vs default and per-person P(delta AUC<-0.02); calibration validity and abstention coverage are separate endpoints.
4. Compare naive source argmax, proper NLL, historical CovScore correctly labeled, fixed lambda, simple CVaR, confidence-based rejectors, and conformal residual calibration with/without nuisance disagreement. Ablate every added component; measure fraction of abstentions and utility conditional on coverage.
5. Include explicit violating-shift counterexamples and show coverage breakdown when acquisition exchangeability fails, not only successful conditions. Correct multiplicity and selection histories.
6. HOLD if no positive selective benefit or robust limit result on unseen data. Never combine this paper's result rows with P1 for two apparently independent confirmations.

## Journal positioning
PLOS ONE official criteria explicitly allow negative/null results but demand scientific validity, complete comparisons and sufficient justification for derivative work:
https://journals.plos.org/plosone/s/criteria-for-publication
A PLOS ONE 'strong accepted paper' is one with a finished, valid and reproducible QUESTION; it need not show SOTA, and an arbitrary novelty percentile cannot be claimed. Do not turn failed exploratory candidate algorithms into a universal theorem.