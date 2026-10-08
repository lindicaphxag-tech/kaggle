# Prospective six-patient label-blind freeze (before first six-patient model outcome)

**Audit provenance:** [Public GitHub Actions #37827321553](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827321553) downloaded, hash-audited and physically coregistered the following six source images **before any six-patient segmentation metric existed**. That initial run was blocked *before model scoring* by a hardcoded `!=3` uniqueness assertion in a generic patient-count validator. The next version corrects only that assertion.

All six are in the public manually refined mask release `MedOtter/SynthStrip`, exact Hub revision `d5a92c65c549b2a099bdb5a6e61ebeed6cb6eca3`. They are exactly the **third through eighth eligible unique-subject T1 images** under the previously committed metadata-only selection, not patients cherry-picked by predictions or annotations.

| Reserved subject alias | Raw public MRI file SHA-256 (content) |
|---|---|
| `synthstrip_heldout_001` | `13f39ae65edc62dc8e898fc6c816560ae897bf4c9a01a51246d879608e0194dd` |
| `synthstrip_heldout_002` | `d2192b83f889d383b4ba7cce42bea37e0136e5383ae802cfe66c10d2a245dec6` |
| `synthstrip_heldout_003` | `7af40aba1dfe13bfb64aa76df8c58ff9a9e7e0d256d8bae632a63926ecdbd4e4` |
| `synthstrip_heldout_004` | `1d24f27185863a694d4ca4d7e335aa29b51f5030dd95b0415ae46db876c84dc7` |
| `synthstrip_heldout_005` | `53657f8a8c394698ef04a20f697e027e76efac2a5aa7fc8bbdc429a3c09563d5` |
| `synthstrip_heldout_006` | `20bc8856934df20744f212d87a7b6b7cb9e53fcb351ca9b706f35f8a1e6bef23` |

- **Source selection:** six next unique T1 metadata subjects, list ordered by fixed source path and `selection-offset=2`. File size constraint `<=65MB` is based on available repository metadata, not imaging/label content.
- **Frozen original pretrained model:** exact source Git ref `ca789fb26b70e359b61a8d52968813dd772fb4b1`, HDF5 original checkpoint Git blob `6ac54600739585f54d9f612d13264b74965ffd33`.
- **Cropping:** three predeclared 64³ regions per patient (center, axis0 -24, +24). Patient macro average across three ROIs; six patients, **not 18 independent patients**.
- **Seven fixed arms:** canonical unpermuted first-view output, physical-space probability and logit 3-view means, nonlinear untransported and spatially conjugated learned-block averages, raw-index negative control, and **new source-label-free disagreement-gated fusion**.
- **Gate written before six-patient results:** `0.5 * exp[-min(50, max_crossview_prediction_disagreement/0.10 + max_pretrained_operator_chart_noncommutation/0.015)]`, multiplying the departure from the original first-view prediction toward the physically aligned three-view probability mean; no access to any mask or six-subject outcome. No calibration/coverage/safety guarantees.
- **Endpoints:** Dice, physical symmetric HD95 with fixed common-ROI empty-mask treatment disclosed in the discovery pilot, all-case table, gate blend coverage, original source model same-weight reassembly/replay residuals. Neither method selection nor admission depends on improved performance.

**Timestamp/claim caution:** These file hashes were recorded from a run that failed its *patient-count validator*, not from a successfully completed six-case experiment. This page freezes the file identity and experimental questions **before** the corrected run's first measured outcomes. Its presence does not prove any improvement, independent reproduction or external clinical validation.

Raw image/mask volumes stay in ephemeral CI and are not published in repo or artifact.
