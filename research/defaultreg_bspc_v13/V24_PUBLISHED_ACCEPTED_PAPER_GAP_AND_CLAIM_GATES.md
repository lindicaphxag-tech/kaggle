# V24 external referee gap analysis: Default EEG vs accepted research (2026-10-13)

## Evidence-backed scientific positioning

**BSPC first**, as a validity-of-model-selection and zero-target-information methodological investigation rather than a claim of a new SOTA classifier. Avoid portraying a conditional source-dataset numerical certificate as an unseen-person guarantee.

### Closest published problem-oriented paper

Gil & Hernandez-Sabate, BSPC 120 (2026) 110020, doi:10.1016/j.bspc.2026.110020, "What is the across-subject performance of your Artificial Intelligence method for EEG analysis?" This paper builds hierarchical mixed-effects validation and demonstrates sizeable window-to-subject leakage/optimism in seizure EEG. Its methodology and participant-unit focus create a genuine **novelty overlap** with our leakage audit. Differentiate DefaultReg specifically through (i) historical class-centred vs proper predictive Gaussian score estimands, (ii) frozen source-only shrinkage decisions, (iii) continuous LDA spectral rank envelopes, and (iv) source-optimisation versus target-generalisation gap decomposition. Do NOT claim novelty for participant-level testing alone.

### Contemporary algorithm-heavy EEG comparison

He et al., BSPC 118 (2026) 109762, doi:10.1016/j.bspc.2026.109762, "Cyclic deep representation-based domain adaptation for cross-subject motor imagery classification". Algorithm/architecture + tested adaptation, rather than our strict inductive source-only classical mechanism. Direct performance rank not defensible across different preprocessing, datasets, and target access.

Zhang et al., Cortex 201 (2026) 171-189, doi:10.1016/j.cortex.2026.05.005, "Evaluating the effects of regularization and cross-validation parameters on the performance of SVM-based decoding of EEG data". Regularisation sensitivity and validation choices are **not new topics**. DefaultReg must focus on the class-centred covariance estimand error, source-target regret gap, and transfer harm, not generic tuning.

A 2026 source-alignment paper (doi:10.3389/fnsys.2026.1840121) explicitly reported that tuning with held target labels yields oracle rather than leakage-free LOSO. This reinforces the importance of distinguishing true Z0 from oracle tuning but does not establish our algorithm's superiority.

## Actual gaps to a publishable, strong methods article

| Gate | Real state | Minimal remedy |
|---|---|---|
| Objective correctness | V18+ proper predictive NLL correction exists | Keep true predictive comparator as principal method baseline; historical class-centred criterion only as an estimand-error example |
| Pre-outcome chronology | Original S021-S040 and frozen BNCI panels plus later exposed cohorts; V24 new Weibo cohort protocol committed | Do not call GitHub freeze a registered trial; preserve failed acquisition / adverse results |
| Paired participant statistics | Earlier prospective tests; overlapping later LOSO source pools | No independent-fold p-values; per person harm and mean differences, posthoc sensitivity properly labeled |
| Novel method | V24 continuous source AUC optimisation with numerical bounds | Only conditionally source-optimal; test whether any unseen-target gains really exist; zero gain is legitimate |
| Dataset novelty | V24 Weibo2014 10 participants new to current repository DefaultReg code search | Run all 10 under one fixed source-only protocol, no target tuning |
| Modern algorithm breadth | Tangent-LDA, LW/OAS, logistic & Riemannian MDM already present in original experiments | New dataset must compare old and proper objectives on identical features, time, target budget, plus compute cost |
| Main paper scope | V21 explicit negative CVaR and V23 source-only negative optimisation | Favor one falsifiable hypothesis: selection-rule correctness vs actual transfer significance |

## Promotion criteria

- Promote V24 continuous optimisation to method-level contribution ONLY IF source-only frozen protocol yields authentic improvement or useful selective safety behaviour over correctly predictive NLL, 27-grid macro, LW and OAS in a genuinely new acquisition, and harms/statistical uncertainty support it. Otherwise discuss conditional source optimality as a secondary measurement instrument.
- Reject fictional "4--19x universally superior" and "top-tier achieved" claims. The previous V24 source experiment CLOSED an ε=0.02 numerical source optimality gap at λ=0.8 with **zero SOURCE AUC gain**. This is reliability progress, not automatically a better decoder.
- Existing 4-dataset prior outcomes, V22/V23 source results, and new V24 Weibo results must remain separate chronology tiers. No pooling to manufacture significance.

## Current real evidence

Matched anchor baseline BNCI2014_002 first outer source pool: source AUC 0.7939663461538462, new anchored source AUC 0.7939663461538462, source upper 0.8101923076933077, conditional residual gap 0.016225961539461542. Run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38278924695

Weibo2014 first-run is pending run receipt; publication outcomes intentionally absent from this document until verified.
