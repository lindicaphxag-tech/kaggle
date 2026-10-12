# V32 actual Dreyer FIRST source-only decision: structural innovation NOT selected
Verified 2026-10-13. Independent read of immutable original first-run artifact,
not a rerun and NOT a future-target performance result.

## Source-only evidence BEFORE held target EEG access
- Original first-run 38299747910 produced SOURCE choices SHA256
  9bfc60e1da10575315917aa6662532198a892dfa99f045341bb71adde58a04b9.
- Separate independent GitHub Actions 38300740581 fetched the failed run's
  original artifact, rederived the SHA, verified the source/held ID sets,
  and parsed ORIGINAL selector and geometry fields. SUCCESS.
- 12 original Dreyer2023A source persons (IDs 1..12), 240 trials each,
  exactly fixed 3 channels and the original group folds.
- Source-macro original full 27-lambda scalar choice: lambda=0.3, alpha=0.
  Source CV macro AUC 0.6254224537037038.
- New mean-AUC matrix operator selector: lambda=0.3, alpha=0.
- New risk objective (macro group AUC −0.25*group SD) selected:
  lambda=0.1, **alpha=0.0**, source grouped macro AUC 0.6253009259259259,
  risk source score 0.5947729253248183.
- Scalar-only risk ablation chose the IDENTICAL lambda=0.1, alpha=0.
- Correct predictive source-mean Gaussian NLL also chose lambda=0.1.
- Pooled-trace raw across-source contrast dispersion 0.02362513341928126;
  estimated within-person finite-trial sampling-noise trace
  0.008354332162259883;
  PSD-debiased heterogeneity trace 0.01591878139103952.
  These positive quantities DO NOT prove transferable heterogeneity.
- Direction cosine of V32 risk selected direction vs *same-lambda scalar*
  is 0.9999999999999999. No matrix effect for this source fit.

## HARD STOP
The new nonisotropic operator's **source-selection structural novelty
gate failed**, independently of any held-target AUC:
  selected alpha > 0 AND source PSD B nonzero AND nonparallel
  new direction did not all hold.
If subsequent Dreyer target AUC for risk lambda0.1 beats source macro
lambda0.3, that would be a scalar selection-criterion effect,
NOT evidence of a novel nonisotropic algorithm. If it loses, it is
an honest negative. Neither case justifies promotion to flagship
new-algorithm results from this data.

## Independent target acquisition status
Original 38299747910 locked source first and completed target IDs 13..17,
then failed on held ID 18 OSF ReadTimeout, so it has NO full-result JSON.
The transparent continuation 38300422475 changes only file transport,
enforces bitwise original SOURCE SHA before opening targets, and is
NOT a newly preregistered replication. See the OSF provenance memo.
Do not report any 12-person held AUC until all IDs 13..24 have been
evaluated and the original source SHA is rechecked independently.

## Implication for two manuscript strategy
Paper A (BSPC) remains the corrected validation-estimand study with
prospective and falsification cohorts including Cho40. V32 may be
included as additional sensitivity only if explicitly non-novel.
Paper B remains the separate Liu2024 V26-V30 recording identity study.
The useful deeper problem is WHY between-person mean components enter
pooled covariance although each held person's rank AUC is
invariant to shared within-person translations (exact ANOVA identity),
not another unconstrained alpha grid.
