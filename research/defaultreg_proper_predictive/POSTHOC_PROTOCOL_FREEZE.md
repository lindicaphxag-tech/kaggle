# V16 proper predictive covariance comparator: post-outcome secondary EEG re-analysis

Frozen **procedural specification**, dated 2026-10-11. This is **NOT** a pre-data experiment: all Shin2017A and BNCI2014_002 main target results have already been seen. It must never be inserted into the original confirmatory Holm family, regardless of direction or significance.

## Core scientific question
Can the original covariance-objective-vs-source-macro-AUC contrast survive replacing a validation-subject *class-centred covariance score* with a genuinely source-mean predictive class-conditional Gaussian log score?

## Source-only design
For each held OUTER target subject, discard its own labels/features/stats from every selection calculation. On the remaining source subjects, use the original five-fold participant-grouped source split, refitting each tangent reference, source discriminant covariance, and class means only on the inner source TRAIN participants.

For every feasible scalar lambda in the original frozen 27-point FULL_GRID:
1. Score the held SOURCE validation participant's class-conditioned feature vectors with `m_train[y]` obtained ONLY from its corresponding inner source training fold.
2. Use the original Gaussian regularized covariance `A_lambda=(1-lambda)S_train+lambda nu I`, with the original `predictive_nll(state, residual, lambda)` numerical solver but pass `residual=X_val-m_train[y_val]`. Unlike the archived class-centred score, DO NOT estimate `m_val[y]` or any held-validation class mean.
3. Average held-source participant scores equally; choose minimum log score, tie break to the smaller lambda (same as historical code).
4. Restrict candidate values to the ORIGINAL source-only feasible FULL_GRID for each outer fold. Never inspect the outer target AUC to select lambda.

Write the full selected-lambda vector and a SHA256 receipt **before** loading the already archived target AUC paths. Subsequently read the immutable original frozen target score-family AUC at the selected lambda, and compute post-outcome paired summary against original source-macro, original class-centred covariance selector and analytic Ledoit--Wolf. Do NOT retune any hyperparameter based on result direction.

## Correct chronology
- Existing target outcomes and method successes are known. This run is retrospective, regardless of the within-run ordering or artifact hashes.
- This is not a new target-free confirmation cohort and not a chance to repair the old family p-values.
- Do NOT add another falsely independent sign-flip p-value to the main abstract, even if a posthoc result appears strong.
- If proper prediction equals or improves the source-macro baseline, report that too and narrow the earlier claim.
- If the direction is favourable, it is descriptive and requires a genuinely untouched pre-data cohort for confirmatory promotion.

## Reproducibility
Original EEG loader and fold routines are imported verbatim from Kaggle `research/defaultreg-frontier-shin-bnci002-20261010`, file `research/defaultreg_frontier/run_untouched_frontier.py` (original blob SHA `d551633f3934506ac7e0c536a08de4da4ebc3137`), frozen stack `moabb==1.7.2`, one BLAS thread, original source-group and train-only tangent reference rules. Original target AUCs are joined from Kaggle run 38065081504 artifact 11674238395 with original ZIP SHA256 `cda3210d24a8e9c7f3bf2d48490e51ad51f023beb652816a8679088cc3fb1842` and only after lambda vector lock.

**Status of actual EEG execution is separate from this frozen procedural document.**
