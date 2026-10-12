# V22 mechanism protocol: pairwise rank-transition decomposition

**Status:** 2026-10-12 post-V21 research extension; NOT a registered confirmatory experiment, NOT a claim of unseen-dataset benefit, and NOT a new target-free predictor already validated on real EEG.

## Research question

Why can source-only regularisation choices disagree strongly, even on a three-channel EEG pipeline, while target AUC and downside-risk counts scarcely change? Scalar lambda disagreement, discriminant-vector rotation, pairwise rank change and person-level ROC-AUC are distinct estimands. The existing V21 CVaR negative result must be preserved.

## Frozen algebra (identity, not a novel theorem)

For within-person positive-negative pairs p,n and scores f_lambda(x), define z as positive score minus negative score. The exact person-level ROC-AUC is
A_g(lambda)=sum_pairs h(z)/Npairs, where h(z)=1[z>0]+0.5*1[z=0].
For fixed source-only choices a,b, let M_ij count pairs whose sign class changes from i in {-1,0,+1} at a to j at b. With credits c=(0,0.5,1), exact
A_g(b)-A_g(a)=sum_ij M_ij(c_j-c_i)/Npairs.
Gross crossing G_g=sum_{i!=j}M_ij/Npairs may be large even when signed AUC effect cancels. Report participant-macro averages of G and delta AUC, NEVER a pooled-trial effect. Floating-point ties are exact unless a separate numerical robustness analysis is declared.

## Mechanism tests and falsifications

1. Exactness and null behavior: positive rescaling/translation of scores yield zero churn. Opposing flips can yield zero net AUC. Tests reproduce direct ROC-AUC including ties.
2. Retrospective source-fold stress: for BNCI2014_002 and Shin2017A, recreate the original source_path and original grouped source-validation training/geometry; compare lambda_C (historical validation-class-centred score) to lambda_D (source macro AUC). Before computing outer target outcomes, save source-only reports and SHA-256. Do not substitute the properly predictive score for the historical comparator.
3. Prospective source diagnostic hypothesis, NOT fact: test whether source gross crossings, cancellation and source-person dispersion predict absolute held-person AUC consequences. Freeze candidate features and thresholds using development persons; then evaluate a genuinely untouched external dataset. Existing four datasets were used/exposed and cannot confirm a newly invented mechanism as untouched tests.
4. Mandatory negative cases: source churn but negligible target effect; little source churn but meaningful target effect; harmed participants. Null findings invalidate an early-warning claim.
5. Baselines: absolute lambda distance, macro-source AUC gap, covariance-fit regret, normalized source discriminant rotation, random source-only selection and worst-quarter source CVaR, on matched folds and representations. CVaR is prior art and negative in BNCI2014_004.
6. Statistics: people not trials are units. Inner folds share training people, outer folds overlap source pools; observations are dependent, and standard independent-fold CIs are invalid. Source algebra cannot certify target regret.

## Target-information contract

- source_validation records: SOURCE training people and disjoint held-SOURCE validation people. Their labels are allowed for source AUC and source-only policy construction.
- posthoc_target records: only after choices and frozen scores SHA. This is for explanations; never use to tune selectors, thresholds or hypotheses.
- Archive source-fold receipts, source-only choice vectors, software versions, sample counts, SHA hashes and per-person summaries.
- Original runner initially loads all data in memory but passes only sliced source arrays to fit and selection. This is no-use-of-target-data-in-selection, NOT physical data-access isolation. Stronger isolation would require a separate process/data-access boundary.

## Stop/go manuscript gate

Promote to a new main experimental result only if an independently frozen source diagnostic is useful on a genuinely untouched external acquisition with participant-level uncertainty and strong matched baselines. Otherwise retain this exact identity as an explanatory supplement, not a novel algorithm, SOTA classifier, or claimed target-risk certificate.

Current new evidence: auditable software and deterministic unit tests only. V19/V20/V21 published internal receipts remain unchanged. No improvement in unseen target performance is claimed.
