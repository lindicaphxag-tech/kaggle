# ClinGeom — patient-space repair journal gate (2026-10-10)

**Status:** protocol added to an existing public engineering artifact, not independent patient-level validation or proven clinical value.

## The novel object needing independent proof
A *typed channel-dependent patient-space fiber and structural support certificate* that determines a repair insertion **before an irreversible learned feature-mixing operation**; supports either an executable exact/approximately bounded transport or an explicit refusal.

Do **not** claim novelty for NIfTI affine tracking, NiBabel orientation, standard MONAI MetaTensor inverses, generic min-cut, numerical interpolation, or post-hoc segmentation confidence.

## What current actual evidence says (retain negative outcomes)
The authentic pretrained 3-D MRI 6-patient held-out pilot measured approximately:
- source/canonical-first Dice **0.186392** and HD95 **41.5667 mm**;
- gated output Dice **0.186315** and HD95 **41.5681 mm**;
- equal probability fusion Dice **0.088262** and HD95 **49.5186 mm**.
The gate did **not** beat the canonical-first strong control. These are cropped ROI summaries, not whole-volume clinical efficacy. See CLINGEOM_PUBLIC_ARTIFACT.md and the frozen six-patient result.

## Next genuinely independent test
1. **Before** selecting patients, commit repository SHA, source checkpoint/hash, patient identifier inventory, annotated mask provenance, coordinate convention, voxel spacing and NIfTI affine validation, and a disjoint patient split. No duplicates across patch/ROI folds. Preserve full failed volumes.
2. Pin five matched variants: canonical-first view; conventional resample+logit fusion; original learned nonlinear feature fusion; ClinGeom typed **pre-mix** execution; refusal/fallback. Fix compute/latency/IO budgets.
3. Separate mechanism endpoints (physical-mm sampling error, violation coverage, refusing an invalid merge, genuine pre-mix numerical execution) from task endpoints (patient-macro Dice, physical HD95, failure rate).
4. Run all variants on **independent held-out full patient volumes** with published annotations; report patient-cluster paired differences, domain/modality strata, wall-clock and GPU memory. No patient mask can select runtime repair.
5. Include negative controls: already-aligned channels (unnecessary repair must be inert); nonlocal normalization (structural support must be global or refuse); aliasing/saturation; genuinely out-of-bound resampling; invalid affine or left/right flips.
6. Compare against modern MONAI trace/inverse and simple common-lattice resampling; source-fixed baselines must not be purposely broken to yield an advantage.

## Claim decisions
- If the advantage exists **only** against naive coordinate-breaking fusion, keep this an invariant/correctness paper, not a general segmentation-improvement paper.
- If strongest canonical first-view wins, report a **negative** efficacy result as already seen.
- If the learned nonlinear-feature repair produces reliable cross-patient gains at matched compute and no ID harm, progress to imaging-method submission (IEEE TMI / Medical Image Analysis, conditional).
- Distinguish controlled analytic counterexamples from true pretrained network measurements and from actual clinical/patient accuracy. No "clinical safety certified" claim.

## Relationship to other lines
GeoTrust predicts when geometry harms **inference**. ClinGeom verifies whether channels refer to a physically consistent **patient coordinate object**. C2P diagnoses **which representation/readout stage fails**. Sharing medical datasets, model outputs or source code is allowed only with a clear cross-paper provenance ledger and no duplicated independent evidence.

**Current status:** NOT TOP-JOURNAL SUBMISSION READY. It has substantive executable systems evidence, but lacks independent full-volume intervention success versus the strongest baseline.
