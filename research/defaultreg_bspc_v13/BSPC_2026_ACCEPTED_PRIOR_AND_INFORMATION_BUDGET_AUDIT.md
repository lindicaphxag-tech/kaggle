# BSPC 2026 accepted-paper comparator audit for DefaultReg
2026-10-13 | editorial positioning | not a journal reviewer decision

The flagship question is NOT generic "a new domain-invariant EEG
network". It is WHY source-train Gaussian likelihood and a
participant's pairwise ranking can optimize different shrinkage.
The historical executed covariance comparator must be named
CovScore, not proper Gaussian NLL.

## High-relevance accepted comparator set
- Gil and Hernandez-Sabate, BSPC 120 (2026), 110020,
  DOI 10.1016/j.bspc.2026.110020. Subject-level evaluation,
  hierarchical statistical dependence; prior art prevents us
  from claiming participant-aware evaluation as new.
- Qin et al., BSPC 116 (2026), 109535,
  DOI 10.1016/j.bspc.2026.109535. Spatial normalization,
  within- vs between-person SVM across 9 paradigms.
  Breadth exceeds DefaultReg. Do not claim 3 acquisitions
  equal a 9-paradigm domain-generalization benchmark.
- Morvay et al., BSPC 113 (2026), 108892,
  DOI 10.1016/j.bspc.2025.108892. Wasserstein barycenter
  transport, multi-paradigm cross-subject classifiers.
  Explicit target-information budget needs inspection
  before direct numerical accuracy benchmarking.
- He, Cao and Luo, BSPC 118 (2026), 109762,
  DOI 10.1016/j.bspc.2026.109762. Cyclic deep
  domain-adaptation MI classification. Input channels,
  target unlabelled access and compute budget may differ;
  do not make unfair numerical superiority claims.
- Sosulski et al., Neuroinformatics 2021,
  DOI 10.1007/s12021-020-09501-8. Tiny EEG covariance
  matrix shrinkage has direct precedents.
- Zhang et al., Cortex 201 (2026), 171-189,
  DOI 10.1016/j.cortex.2026.05.005. EEG decoding
  regularisation/cross-validation effects, relevant
  prior art for "regularisation matters" motivation.
- Hu et al., Domain Generalization via Multidomain
  Discriminant Analysis, UAI/PMLR 115 (2020), 292-302.
  Source-domain class-centering is established before GID.

## What an elite BSPC methods reviewer should demand
1. Precisely fix each validation estimand's class means:
   held source own labelled class means vs training-source
   class means; these are NOT test-target label leakage.
2. State all compared AUCs as released-ID person-macro,
   not pseudo-independent individual EEG trial tests.
3. Show original frozen prospective harms, BNCI nonsignificance,
   failed V31 envelope, V32 alpha0 negative, and no universal
   new selector claim.
4. Identify SOURCE-only, target-X transductive and target-label
   supervised budgets for any deep or OT accuracy numbers;
   absent equal budgets, use conceptual comparison only.
5. Show the exact three-part quadratic identity and its
   counterexample: correct likelihood may penalize
   subject translations that do not change within-person AUC.
   Contrast shifts orthogonal to current score also have
   no effect on fixed classifier rankings.
6. Report all 3 source EEG acquisitions and 38 group
   receipts with independent artifact SHA recomputation,
   LOO group reaggregation ranges, a native 15-ch case
   beyond two low-dimensional cohorts, and legacy
   27-point CV parity.
7. The source-only Lee2019 prospective new-acquisition
   GID study is a separate falsification sidecar.
   Its method claim depends on untouched target A and B
   and a pre-data source signed forecast; fail if
   source-criterion gains are absent (as already indicated
   by frozen source delta -0.0008).
8. Provide executable public methods and platform costs.
   Do not assert SOTA, clinical performance or TMI-level
   algorithm contribution on three-channel, 12-person tests.

## Editorial decision
Retain the focused single primary BSPC article;
the mathematical score-decomposition is part of a
scientific measurement study, not new Fisher algebra.
V26-V30 recording-content identity: second technical paper.
The new GID operator, while invariant in a fixed feature
space, is prior-art-adjacent, exploratory and NO transfer
benefit is established at the time of writing.
