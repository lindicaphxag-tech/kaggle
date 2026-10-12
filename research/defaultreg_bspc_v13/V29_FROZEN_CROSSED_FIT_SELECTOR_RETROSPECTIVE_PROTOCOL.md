# V29 crossed-fit × shrinkage-selector counterfactual: frozen, RETROSPECTIVE

2026-10-13. This is an exploratory *after-exposure* mechanism experiment, not independently preregistered, and not proof of a causal leakage effect. All source choices and both target AUC families are already exposed from original V25 and purged V26. The method and denominators are fixed before the V29 calculation.

## Fixed source and outcome artifacts

- Original 50-ID Z0-style experimental artifact (before detecting record overlap): GitHub Actions artifact `11760774972` with `FIFTY_SOURCE_ONLY_CHOICES_LOCKED_BEFORE_ANY_TARGET_AUC.json` and `FROZEN_AFTER_SOURCE_LOCK_50PATIENT_LOPO_RESULTS.json`.
- Globally record-deduplicated 49-group retrospective full refit: artifact `11765005135` containing `PURGED_RECORD_IDENTITY_SOURCE_POLICIES_BEFORE_TARGET_AUC.json`, `FULL_COHORT_EXACT_SIGNAL_GROUPS_AFTER_GLOBAL_AUDIT.json` and `PURGED_LIU2024_RECORD_INDEPENDENT_RETROSPECTIVE_RESULTS.json`.
- Exact-epoch overlap graph and group equivalence independently checked in V28 run `38289471605`.

## Mechanism experiment

Use **exactly** Liu2024 public MOABB==1.7.2, all 50 original identifiers/2,000 nominal trials; same 29-channel, 8–30 Hz, 160 Hz bilateral motor imagery, same original frozen tangent-LDA pipeline. Use the 49 recording-group representatives as shared target set; each target has 40 trials.

For each group and each of exactly three original source-only selection policies (source-macro AUC, correct predictive source-mean Gaussian, historical validation-class-centred covariance), read lambda_A from the originally frozen 50-ID source-selection manifest and lambda_B from the separately frozen purged 49-record manifest. Neither is recomputed using held target data in V29.

Fit **two** models:
- A: original source training pool with the nominal held ID omitted; for target representative 4, identical copy 5 is still in the source (an invalid record-independent design).
- B: only 48 unique source representative recordings, with full record-content group omitted and all geometry and spectral LDA refitted.

Evaluate all four frozen model × lambda combinations on the SAME 49 held representative EEG target recordings:
`Y_AA = A-fit + lambda_A`; `Y_AB = A-fit + lambda_B`; `Y_BA = B-fit + lambda_A`; `Y_BB = B-fit + lambda_B`.

**Hard integrity gate:** all 49 Y_AA values must reproduce each matching original archived target AUC, and all 49 Y_BB values must reproduce each corresponding V26 fully purged real target AUC, within 1e-10. If not, FAIL; never repair via outcome tuning.

## Descriptive estimands

Mean over 49 unique held recording groups. Conditional contrasts:
- parameter substitution within original model: AB-AA;
- model-fit substitution at original parameter: BA-AA;
- parameter substitution within purged model: BB-BA;
- model-fit substitution at purged parameter: BB-AB;
- factorial interaction: BB-BA-AB+AA.

Exact identity `BB-AA = (AB-AA)+(BB-AB) = (BA-AA)+(BB-BA)` must hold for every target ID and aggregate.
Report method-specific 2×2 means, all 49 individual deltas, changed lambda count, target group harm counts (>0.02), and non-significance-sensitive descriptive summaries.
This is a **counterfactual software intervention over fixed, already-exposed EEG arrays**, not a causal intervention on patients or a claim that every observed AUC change is leakage inflation.

## STOP

No source-selection threshold may be changed; no new target cohort is admitted; no crossfold independent p-values; no new Z0 or clinical safety claim. Raw EEG arrays must not enter public artifacts. Only numerical AUCs and previously public subject identifiers and signed paired changes may be emitted.
