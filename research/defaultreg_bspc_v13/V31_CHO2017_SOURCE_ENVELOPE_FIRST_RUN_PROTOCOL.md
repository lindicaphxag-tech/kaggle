# V31 research protocol — source-only contrast envelope, Cho2017 external acquisition
Date: 2026-10-13. Repository-committed protocol, not independent registration or journal acceptance.

## Claim under test
Does a source-only group-contrast ellipsoid alter regularisation decisions and
improve target-person macro ROC-AUC versus the original source-macro-grid
selector, when both choose among the SAME 27-point regularised LDA path?
The robust criterion is:
  max_lambda (w_lambda^T dbar - rho sqrt(w_lambda^T B w_lambda))
                / sqrt(2 w_lambda^T Sigma_within w_lambda).
Here w_lambda uses the original globally class-centred pooled LDA covariance
for fitting. dbar is the equally weighted mean of source participant class
contrasts. B is unbiased source between-contrast scatter + 0.01 times
within-class average feature variance times identity. Sigma_within is
group-by-class-centred within-source covariance + 1e-8 isotropic relative ridge.
rho=1 fixed before any external outcome. rho=0 is a mechanism ablation.

This is a Gaussian contrast-uncertainty sensitivity score. Neither
Gaussianity, equal source/target covariance, nor unknown-target ellipsoid
membership is established by the observed source groups. Existing V15
conditional geometry is prior internal work; do not claim ellipsoidal robust
optimization, conditional Cauchy-Schwarz bound, or shrinkage LDA as novel.
Innovation under investigation is measured predictive discrimination and
conditional routing utility, NOT the elementary analytic formula.

## FROZEN external data scope, before inspecting target outcome
- Dataset: MOABB Cho2017, source subjects 1..12, held target subjects 13..32
  (20 distinct released IDs; no 33..52 in primary results).
- Sensors exactly C3, Cz, C4; classes left_hand/right_hand,
  positive right_hand; band 8-30 Hz, resample 160 Hz,
  MotorImagery baseline=None, reject_by_annotation=False.
- Full source-only geometry and 27 candidate lambdas inherited from original
  runner and exact source_path feasible grid.
- No held target EEG, labels or feature moments read before SHA256 freezing
  all source decisions on disk. Source-fit tangent reference only.
- Each participant requires BOTH events, three EEG channels, at least
  20 trials per class, fully finite numerical arrays; failure aborts,
  and no replacement IDs/changed sensors/alternate bands allowed.
- No target-specific model, lambda, reference, threshold or normalization.
- If dataset package downloads extra disk files, no extra target data opened
  by this runner prior to source-choice lock.
- Metadata alone is not proof of physical recording identity; this is a
  released-ID split unless independently audited.

## Compared on SAME target trials
1. Primary method: ellipsoid margin selector rho=1.
2. Primary reference: grouped full-grid source macro ROC-AUC choice.
3. Correct held-source-trained-mean Gaussian predictive-NLL selector.
4. Historical held-source-validation-class-centred covariance criterion.
5. Classical analytic Ledoit-Wolf and OAS LDA.
6. Ablation: source geometric criterion rho=0.
Original classification score function and feature pipeline shared by
(1)-(4) and (6).

## Frozen evaluation and rejection gate
- Primary estimand: participant-macro paired target AUC difference
  (rho=1 minus source-macro 27 grid), N=20.
- Report each signed participant delta, >0.02 harms and benefits,
  paired participant-bootstrap 95% descriptively, all method means.
- Publication-level positive outcome requires positive mean improvement,
  two-sided paired sign-flip p<0.05 for this single predeclared comparison
  and no increased >.02 harm count.
- A 0.01+ mean AUC gain would be practically interesting but is not guaranteed.
  If primary is equal, negative, or non-significant, report failure plainly.
- Conditional dominance certificate is source-only and tested against
  observed aggregate target ordering only after held participant evaluation.
  It is NOT an unconditional prediction or calibrated target coverage.
- No parameter/radius/sensor tuning on these target subjects after results.
- This is first-run by code commit, not external formal preregistration.
- Do not claim SOTA over deep neural architectures based on this classical
  shrinkage-family comparison.

## Editorial separation
Primary BSPC candidate retains its V24 source-versus-predictive mechanism and
prospective/negative results. V26-V30 recording provenance stays a separate
paper. V31 does not enter headline abstract until real external test and
strong classifier comparator checks pass. All outputs remain research draft.
