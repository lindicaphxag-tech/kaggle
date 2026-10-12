# V23: spectral rank-stability certificates on an EEG regularisation continuum

**Status:** 2026-10-13 mathematical/methodological development; candidate for mechanistic research. This is not independently preregistered, not a novel target-generalisation bound, and not a demonstrated improved regularisation-selection algorithm.

## Motivation and prior-work boundary

V21 observed source selector disagreement without systematic corresponding target improvement. V22 decomposed *observed* source-person AUC changes into pairwise sign transitions at two selected lambda values. V23 goes beyond endpoint bookkeeping: derive an analytic **interval enclosure for all regularisation values between endpoints**, conditional on the trained spectral model and fixed held-source observations.

Do not claim the mathematical idea of interval arithmetic or LDA shrinkage-path eigen-decomposition is novel. The contribution is their specific combined construction for zero-calibration EEG regularisation decision sensitivity, with matched source-only experiments and a falsifiable aim to predict when an objective discrepancy is genuinely decision-relevant.

## Continuous-path spectral operator

Source-trained LDA:
\[
 A_\lambda=(1-\lambda)S+\lambda\nu I,\quad
 S=U\,{\rm diag}(s_1,\ldots,s_p)U^\top,\quad
 w_\lambda=A_\lambda^{-1}d_s.
\]
For a fixed held **source** positive-negative feature pair
\(v_{pn}=x_p^+-x_n^-\), define \(a_{i,pn}=(u_i^\top d_s)(u_i^\top v_{pn})\).
Then score contrast is
\[
 z_{pn}(\lambda)=\sum_{i=1}^p\frac{a_{i,pn}}
 {(1-\lambda)s_i+\lambda\nu}.
\]
For feasible subinterval [l,r], every denominator is strictly positive and affine, so EACH summand is monotone (increasing or decreasing). Therefore define
\[
 L_{pn}[l,r]=\sum_i \min\{a_i/q_i(l),a_i/q_i(r)\},\qquad
 U_{pn}[l,r]=\sum_i \max\{a_i/q_i(l),a_i/q_i(r)\}.
\]
It follows for every lambda in [l,r] that L <= z(lambda) <= U.
If L>0 (respectively U<0), the pair is guaranteed positive (negative) over this entire subinterval. Across ALL subintervals, require each to prove the SAME strict sign to declare interval-wide stability. Failure to certify is uncertainty, NOT evidence of an actual sign crossing.

For the fixed source participant g, let U_g be fraction of uncertified positive-negative pairs. Then for all lambda in [a,b],
\[
 |\mathrm{AUC}_g(\lambda)-\mathrm{AUC}_g(a)|\le U_g.
\]
This is a deterministic finite-sample bound under the stated fixed model and observations, not a distribution-free confidence interval. Floating point implementation adds conservative slack to reduce fragile near-zero certificates; numerical certification is conditional on the computed eigenbasis and is not a formally verified directed-rounding computation.

## Frozen numerical verification

- Exact 1-D no-crossing and 2-D known-root cases; ties and singular paths fail closed.
- Random SPD eigenbases and source pair trials; evaluate dense intermediate lambdas and confirm every empirical AUC stays within the computed bound.
- Evaluate subdivision refinements 1/8/32; certificate should not weaken with additional subintervals (modulo numerical tolerance).
- Run ORIGINAL frozen source-only BNCI2014_002 tangent-LDA pipeline (no new preprocessing), first one, then all 14 outer source pools; 13 source people available for each outer pool, inner held-source people scored under source-trained geometry.
- Compare with V22 EXACT source AUC delta for identical selectors; any deviation > 1e-10 must fail CI.
- Report certified stable fraction and bound width, while preserving false uncertainty (loose bounds) and any negative results.
- Do NOT use outer target features or labels to tune a path, choose subintervals, or decide a lambda.
- No new target unseen-person AUC evidence; after an independent new-data protocol is frozen, measure how often source certificate meaningfully predicts target harm. Do not relabel post-outcome selection as a test.

## New method promotion gate

A genuinely useful zero-calibration regularisation policy needs a FROZEN mapping from source statistics/certificates to a selected lambda, and validation on a cohort not previously examined while designing it. A bound on SOURCE-trial rankings cannot by itself license predictions about unseen-person risk. The V23 chapter is a mechanism-level contribution; the top-tier algorithm/clinical impact gap remains open until those experiments exist.

## External literature scope

- Shrinkage LDA and spectral regularisation paths are classical (Friedman; Ledoit-Wolf; standard LDA tools).
- Stability selection is established methodology, including Meinshausen/Buehlmann.
- Source selection, confidence gating and complex EEG domain generalisation were actively published in 2025/2026; do not claim novelty for a generic gating or composite-loss recipe.
