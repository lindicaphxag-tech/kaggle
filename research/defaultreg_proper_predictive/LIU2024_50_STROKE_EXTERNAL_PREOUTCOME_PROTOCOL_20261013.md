# V24 locked-before-outcome external acute-stroke EEG protocol

**Protocol authored:** 2026-10-13. This is an internal repository-first specification, **not external preregistration**. Do not edit endpoint definitions or selection rules after any target result is visible; if the first run fails, preserve the preflight failure and annotate fixes. The repository's preceding EEG studies involve healthy/other MI subjects; this new source acquisition specifically stresses generalisation in a 50-person acute-stroke population.

## Dataset and fixed universe

Public [Liu2024 EEG motor imagery acute stroke data](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Liu2024.html), original [Scientific Data 2024 article](https://doi.org/10.1038/s41597-023-02787-8), all **50** originally released participants, 29 EEG channels (never include 2 EOG), 2 classes left-hand vs right-hand; 40 nominal trials/person, one session/person. The code must verify actual trials, labels and channels; no participant may be excluded after observing outcome. If public retrieval, event quality, or dataset package incompatibility prevents all-subject execution, declare incomplete and retain receipts rather than substituting subjects.

## Protocol and information budget

- Same predeclared 8–30 Hz, 160 Hz, no baseline, source-fitted tangent reference as the original V21 pipeline where compatible; this is **not** a clinically optimised pipeline.
- Leave-one-patient-out (LOPO): held outer patient never contributes features, labels, normalization or geometry to source selection. Source labels from the other 49 patients are allowed. Five grouped SOURCE-validation folds, no trial-level leakage.
- Class encoding: right_hand is positive class, left_hand negative.
- Original historical 27-point scalar regularisation grid and original original_frozen_runner spectral class means and trial covariance mechanics remain intact.
- Baselines: historical held-source-class-centred score selector, held-source patient macro ROC-AUC selector, genuine source-training-mean Gaussian predictive score selector, fixed Ledoit-Wolf and OAS on source-fitted tangent features.
- **Experimental algorithm (fixed in advance):** run V23 conditional source-only branch-and-bound continuous search over lambda in [0.05,0.95], fixed tolerance 0.02, max_splits=5, one spectral subinterval per node. Use its best source-AUC lambda **only if** the source-AUC optimality certificate is satisfied; else fall back to original 27-grid source-patient macro AUC choice. Any result that uses the fallback MUST disclose how often this happened. This rule is not claimed to guarantee unseen-person improvement.
- Fixed target metrics: participant macro ROC-AUC, change relative to macro-source and genuine source-mean predictive comparator, mean signed change, >0.02 person-level harm/improvement count, patient-level paired bootstrap interval (descriptive with overlapping source pools), and selection-fallback frequency. No trial-as-subject pseudoreplication.
- **Temporal gate:** compute all 50 outer source-only lambda choices, list model/data protocols and SHA-256 lock to disk **before** evaluating any held outer labels or predictions; only then fit outer source models and evaluate target.
- No use of target-person unlabeled distribution either (strict Z0). Loading raw records in a common process does not constitute physical isolation; source masks must be passed to EVERY fitting/selecting subroutine.

## Failure/promote rule

If source-only continuous selector never certifies, report this as a negative numerical optimisation result; do not silently substitute parameter choice or claim V23 superiority. A clinical dataset is **not** validation of deployment efficacy. No top-paper claim follows without independent replications, stronger baselines and BCI task utility. Outcome evaluation requires the complete protocol and all 50 patient records to be checked, with any missing persons openly reported.

**Pilot/preflight:** code may download and inspect the first participant's structural metadata to confirm 29 EEG channels, 2 classes and available trial length, without scoring target labels or selecting parameters. This preflight alone gives no EEG effectiveness evidence.
