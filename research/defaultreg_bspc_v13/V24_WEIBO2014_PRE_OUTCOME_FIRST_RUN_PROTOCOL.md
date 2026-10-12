# V24 FIRST-RUN EXTERNAL EE​G test — Weibo2014 source-only MI

Freeze date: 2026-10-13. This text and experiment code are committed before the new cohort outcomes are computed. In-repository protocol-first, NOT registered independently; previously published dataset is not itself novel. This cohort is not known from this repository's DefaultReg experiments, but global author exposure cannot be excluded.

## Scientific hypothesis and one allowed comparison

Primary EXTERNAL estimand: participant-macro held-out AUC difference between source-only V24 **grid-anchored continuous shrinkage** and the original **27-point source-macro shrinkage**. Null/negative permitted. Secondary: correctly predictive source Gaussian NLL selector, historical class-centred covariance criterion, Ledoit-Wolf and OAS on the SAME model and source-only folds. No improvement sign presumed.

## Fixed acquisition and split (not adapted after targets)

- Dataset `moabb.datasets.Weibo2014` under MOABB==1.7.2. All subjects IDs 1–10 must exist, or FAIL.
- Exactly the binary event pair `left_hand` and `right_hand`; drop all compound imagery and rest classes by initial fixed MOABB paradigm.
- EEG sensors exactly `['C3','Cz','C4']`, excluding non-EEG; no channel fallback or target-optimised electrode selection. Original MOABB preprocessing: `MotorImagery(n_classes=2, events=['left_hand','right_hand'], channels=['C3','Cz','C4'],fmin=8,fmax=30,resample=160,baseline=None,reject_by_annotation=False)`.
- Original covariance / Riemannian source-fitted tangent reference / scalar spectral LDA, original 27 FULL_GRID and original fivefold held-source-subject CV. Fixed continuum domain [0,1], inner tolerance 0.02 source AUC, max 5 interval bisections, preseed with all feasible ORIGINAL FULL_GRID lambda candidates; no tuning on target outcomes.
- Each of 10 subjects held out exactly once. Source predictors select all parameters using other 9 source people. Full 10-person vector of choices must be saved and SHA256-hashed BEFORE any target score generation or ROC-AUC.
- The source-only reference model and correction comparator must use source-trained means; historical class-centred score must be labeled as such. Same target trial set, same Riemannian source reference and same held-person evaluation for all selectors.
- Fail on missing EEG channel, one-class participant, fewer than 20 trials/class, mismatched subjects, infeasible candidate grid, unexpected target access, or absent downloadable data. Do not silently exclude persons or move the dataset.
- Source-only selection does not train geometry on held outer target. A shared initial loader still reads raw targets and is not a hardened physical data-isolation boundary.
- This dataset is an acquisition and task-variant external test, but due to THREE selected electrodes it tests low-dimensional transfer only; not a general across-architectures comparison.

## Statistics, decision and claims

Report individual target-person AUC, paired deltas to macro-grid and proper NLL, participant mean/median, harms >0.02, total trials. All 10 held people share overlapping source pools, so any simple participant bootstrap is descriptive/conditional. No fold-level independent p-values, no SOTA or zero-calibration deployment claim. Compare algorithm computation cost (lambda evaluations, intervals) transparently, alongside whether continuous source optimality certificate actually closed (may fail).

GO only if (a) first-run data/provenance controls pass, (b) new selector is source-only and gives at least a consistent practical advantage without disproportionate harm, and (c) result survives matched strong baselines and independent follow-up acquisitions. One dataset with 10 people does not establish top-tier transfer efficacy.

Evidence chronology: 2026-10-13, downstream first run must happen only after this protocol and executable are committed. Outcomes will be appended in a separate commit; no manuscript performance numbers are invented.
