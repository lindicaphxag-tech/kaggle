# BSPC primary paper: source-only two-factor NLL anatomy protocol
Date 2026-10-13 | Research method audit | NOT pre-target external registration.

## Hypothesis under investigation
Historical held-source validation-class-centred covariance NLL (CovScore)
does not equal true source-trained Gaussian predictive NLL.
Their EXACT difference is a nonnegative quadratic mean-shift term;
the new diagnostic factorisation separates:
(1) weighted global common-location shift, which is INVARIANT for each
held person's within-person ROC-AUC for any fixed linear scorer, and
(2) class-contrast shift, which can influence ROC-AUC but need not do so.
The second is density discrepancy, NOT a proven AUC loss surrogate.
Both terms are evaluated on the SAME source-trained covariance and
held-SOURCE-VALIDATION labels. No held target labels or EEG are used.

## Frozen source groups and data scope
Two existing, completely EXPOSED acquisition collections:
Dreyer2023A and Cho2017, SOURCE IDs 1..12 each; these are not new
untouched target studies and contain NO new confirmatory performance.
Use original MOABB 1.7.2, exactly C3,Cz,C4, left/right motor imagery,
8--30 Hz, 160Hz, baseline None, reject_by_annotation=False.
The same original 5-fold round-robin SOURCE group split, per-fold
source-trained Riemannian tangent reference, original full 27-lambda
pooled-LDA path and held-source class label convention are preserved.

## Precise algebraic score definitions
For any held source group g:
 C_cent=(logdet(A)+mean_i ||x_i - val_class_mean[y_i]||²_A^-1)/p
 C_pred=(logdet(A)+mean_i ||x_i - train_class_mean[y_i]||²_A^-1)/p
 pi_c=n_gc/n_g, delta_c=mu_g,c - mu_train,c,
 u=sum_c pi_c delta_c, v=delta_1-delta_0.
 P_common=||u||²_A^-1/p
 P_contrast=pi0*pi1*||v||²_A^-1/p.
 EXACT C_pred - C_cent = P_common + P_contrast.
 - Shift a validation participant's BOTH class samples by arbitrary
   same vector: C_cent and its within-person ROC-AUC unchanged, while
   C_pred and P_common generally change.
 - Class-contrast shifts orthogonal to the fitted discriminant can
   change P_contrast while leaving ROC-AUC unchanged.
Neither P term estimates causal generalization improvement.

## Precommitted source-only descriptives, not significance tests
Run both acquisitions without reading ANY IDs >12.
For every feasible lambda report mean of the exact three scores,
source-group macro ROC-AUC, P_common, P_contrast and per-group rows.
At the original source-macro, historically-class-centred and proper
predictive selected lambdas, report contributions and fraction
P_common/(P_common+P_contrast).
Include an exploratory "CovScore+P_contrast" source diagnostic,
not as a valid predictive likelihood or a performance-proven selector.
A positive common-location share would empirically support the
existence of AUC-invariant error terms in real EEG source validation;
no posthoc threshold defines confirmation. If the share is low,
preserve that negative observation.
Do NOT use the already exposed 20/40 Cho targets or Dreyer targets
for new policy tuning, any "fresh test", or new inferential p values.

## Paper positioning
The actual flagship contribution, if supported, is identifying
non-equivalent EEG regularization VALIDATION ESTIMANDS, testing their
practical effects and showing honest failures of source-only
optimality; not claiming that a two-term algebraic identity or
a contrast-only source diagnostic is an algorithmic breakthrough.
An independent previously unseen collection and precommitted
comparison with strong methods remains required for L8/L9.
Keep Liu V26-V30 recording identity separate.
