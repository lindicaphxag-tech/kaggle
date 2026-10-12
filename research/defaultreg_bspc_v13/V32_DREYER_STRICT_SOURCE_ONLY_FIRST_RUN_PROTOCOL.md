# V32 pre-outcome research protocol — Dreyer2023A spectral transfer operator
2026-10-13; GitHub protocol-first only, NOT independent preregistration.

## Scientific hypothesis / prior outcome handling
Cho2017 V31 (40 target subjects) showed near-zero average SCE improvement and
an exact lambda=1 endpoint degeneracy. All Cho 1..52 outcomes are
EXPOSED and PERMANENTLY EXCLUDED from V32 new confirmatory inference.
This attempt tests a different MECHANISM, not a retuned SCE radius.

Source-group contrast heterogeneity mixes true between-person direction
drift and finite-trial class-mean estimation noise. The original scalar
pooled-LDA covariance path may overweight directions unreliable
across people. We estimate the source group-contrast covariance,
subtract within-source finite-trial sampling-variance expectation, project
to PSD, and add alpha*B_debiased to the original pooled LDA matrix.
Alpha=0 is proven identical to original source LDA for every lambda.
This is NOT a globally new LDA or distribution-free target theorem;
the original research claim is whether the unbiased-noise correction
produces a measurable, decision-predictive transfer benefit.

## Frozen new acquisition / strict boundaries
Dataset MOABB 1.7.2 Dreyer2023A; 60 released participant IDs:
SOURCE IDs 1..12, HELD IDs 13..24 (12 never opened for V32 method dev).
Nothing from any Dreyer target ID may be read or used before the source
parameter manifest is saved to disk with SHA256.
Channel names exactly C3,Cz,C4; left_hand/right_hand, right_hand positive;
8-30 Hz, resample160, baseline None, reject_by_annotation False.
No target label/statistic/geometry/normalization for any selection or
training. Class and channel schema fail closed; min 20 trials/class per ID.
No missing-subject replacement, no different task or window after results.
If download fails/time budget exhausts, mark experiment NOT COMPLETED,
do not substitute another acquisition and claim an untouched success.
A released-ID split is not a physical-patient identity guarantee.
Precommitted intergroup source fold protocol: original modulo-5
round-robin source-group folds. Inner tangent reference, source
contrast B, source LDA and all alternatives must be re-fitted using
training GROUPS ONLY.

## Locked optimization / baselines
Original finite 27 shrinkage lambdas from original_frozen_runner FULL_GRID.
Alpha coefficient {0,0.5,2.0}, no target tuning.
Source macro benchmark: optimal alpha0 lambda by mean held-source
group AUC. New risk-aware selector: maximize
source-group macro AUC - 0.25*(SD of source-group held AUCs).
Ablations include alpha0 risk-aware (scalar path only), unrestricted
alpha>0 mean-AUC selector, alpha0 source macro.
In all cases train the final source-fitted classifier on exactly the
same 12 source IDs. New method only counts as structurally novel for
this dataset if risk selector returns positive alpha AND changes
the direction relative to scalar path. If alpha=0 or B=0, the method
is identified as BASELINE, not a novel successful algorithm.
Additional source-only controls: correct predictive source-mean Gaussian
NLL selector, analytical LedoitWolf/OAS, fixed lambda1, pooled
tangent-space LogisticRegression(C=1) and Riemannian MDM.
CSP+LogReg may be included only with exactly same channels, epochs,
source persons and no target-tuned hyperparameters; no claim of
SOTA even if AUC surpasses subset of these methods.

## Success, metrics and causal restraint
Primary paired macro AUC: risk-selected matrix-operator minus ORIGINAL
source macro-AUC scalar path on exactly 12 held persons (not 12
independent collections). Record signed per-person AUC changes and
per-person losses beyond .02; report bootstrap descriptive 95% CI
and exact two-sided sign-flip p (2^12) with assumptions.
Method promotion requires mean gain at least .01, exact p<.05,
non-increasing >.02 harm rate vs the SAME baseline, and
nontrivial alpha>0 + nonparallel change; the four conjunctive
conditions are criteria for this exploratory first-run, not an
independent regulatory/statistical guarantee.
Report all seven scalar/matrix or matched-budget competitors and
the alpha0-only ablation; do not conceal failures.
Results do NOT become independent preregistered evidence simply
because choice manifest preceded access to target trials.

## Mathematical limitations
d_group sample covariance - E[class-mean sampling covariance]
is unbiased only under independent group/trial sampling and
finite second moments; PSD projection is positively biased and may
overstate heterogeneity in few-group, high-dimensional settings.
No assumption-free target membership, no future-target guarantees,
no patient-level causal claims. Operator can help or hurt.

## Publication gate
The V24 BSPC criterion-mismatch manuscript remains the primary target,
with all historic true positive and negative results unchanged.
V26-V30 recording-identity story is a DISTINCT research note.
V32 is a sidecar METHOD RESEARCH experiment and must NOT replace
the BSPC headline unless held new acquisition supports all predeclared
gates plus subsequent independent replication and matched classifiers.
