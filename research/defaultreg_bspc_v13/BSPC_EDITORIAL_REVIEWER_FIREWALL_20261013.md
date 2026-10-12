# Default Regularisation: BSPC submission firewall / strict reviewer checks
Research status: 2026-10-13 | DO NOT submit before author and integrity gates.

## One flagship argument, not an invention-per-version narrative
Journal target: Biomedical Signal Processing and Control (BSPC).
Working title: *When Predictive Fit and Discriminative Transfer Disagree:
The Geometry of EEG Regularisation Selection*.

A. Historical program used held-SOURCE participant class means to
calculate a class-centred covariance log score and called it
"Cov-NLL". This is NOT source-trained-mean predictive likelihood;
it never used outer held-TARGET labels for selection.
B. A correct predictive score adds a nonnegative between-source-person
mean-shift term. An exact weighted 3-way decomposition distinguishes
(i) common location drift invisible to fixed within-person AUC,
(ii) class contrast drift orthogonal to the actual trained LDA score,
also invisible to its fixed pair ranks, and (iii) discriminant-score
aligned contrast drift which can change score margins but does NOT
guarantee any positive or negative AUC effect.
C. Source-only post-outcome diagnostics in three public MI-EEG
acquisitions show unusually high fractions of the predictive-versus-
class-centred mean-shift *penalty* residing in AUC-invariant terms
(Cho, Dreyer three-channel 6D; BNCI native 15-channel 120D).
These fractions are NOT fractions of useful EEG information,
nor do they predict an unseen target improvement by themselves.
D. Frozen prospective EEGMMIDB and independently protocol-frozen
Shin show significant group-macro source-discrimination superiority
over HISTORICAL class-centred CovScore, with harmed persons.
External BNCI families individually did not meet adjusted significance;
Weibo continuous-vs-grid selection was a transfer tie; Cho40
robust ellipsoid policy was no improvement; Dreyer12 novel matrix
penalty was alpha0 and a nonsignificant small negative.
E. No claimed novel universal better EEG decoder. Keep failed
V31/V32 policies as falsification evidence, NOT positive methods.

## Novelty review: credit classical methods accurately
- The three-factor partition is weighted quadratic/parallel-axis
  decomposition plus projection in an SPD metric. It is mathematically
  exact but standard algebra; never advertise as an unprecedented theorem.
- LDA, scalar covariance shrinkage, Riemannian tangent decoding,
  source-grouped validation, random-effects covariance and ellipsoidal
  robustness are established prior art.
- Publication candidate novelty is the *measurement discovery and its
  empirically demonstrated consequence for interpreting cross-subject
  EEG validation*, with strict temporal evidence provenance.
- A paper claiming superior algorithm performance would need a
  genuinely nondegenerate operator, pre-outcome frozen strong
  comparators, new acquisitions, paired inference, and repeated gains;
  the current evidence explicitly does not support that claim.

## What a hostile reviewer should be able to verify
1. Eqs for C_cent and C_pred use the SAME source-trained A_lambda and
   identical logdet and (1/p) normalisation; validation-class means are
   computed only from held SOURCE groups. No target leakage.
2. The contrast decomposition is exact even with class imbalance;
   fixed-score ROC-AUC is invariant to subject common translation.
   Even score-orthogonal contrast changes can be large in predictive NLL.
3. The original 27-point participant-grouped source macro-AUC and
   historical class-centred lambda are exactly matched by new code.
   New 3-acquisition source group diagnostics are labelled post-outcome
   despite all data being SOURCE for those new grouped folds.
4. The subject, not trial, is the transfer statistical unit; paired
   bootstrap intervals are descriptive and Holm-adjusted p-values
   belong ONLY to their original separately predeclared families.
   Grouped SOURCE validation folds share training people, so the
   12/14 group records are not independent new replication studies.
5. Every materially harmed person remains visible. Injection, removed
   participant and duplicate content issues belong only to the
   separate Liu2024 recording-identity paper, not main causal claims.
6. Comparators keep equal channels, signal preprocessing, training
   information budgets and target-calibration rules; MDM and
   tangent LR are nonidentical classifier controls, not proof that
   the new selection policy beats deep EEG DG architectures.
7. Repo CI source files, SHA256 results, per-group measurements and
   independently audited artifact records remain accessible.

## Target-journal editorial fit and likely referee objections
BSPC accepts methodological biomedical signal analysis and
experimental signal-processing studies, including EEG and BCI.
But empirical methodology papers must provide more than a basic
algebraic identity; the three-source-acquisition result and frozen
transfer falsification are essential for subject-specific relevance.
A top-quartile accepted comparator may include higher-channel and
multi-paradigm studies (Qin et al., BSPC 2026) and multi-source domain
generalisation models. We cannot equate our 3-acquisition source
anatomy with an accepted deep DG algorithm's SOTA accuracy claim.

Likely MAJOR-REVISION risks:
- Quantified source-score anatomy is retrospective on fully exposed
  data; high percentages do not imply generalisable target-AUC gains.
- Ternary decomposition can be regarded as algebraically elementary;
  the signal-processing interpretation and actionable evaluation
  policy must be discussed without novelty inflation.
- Strong neural or classical nonidentical comparators in each
  original prospective cohort are incomplete/mixed information
  budgets; algorithm leaderboard claims must not be made.
- Selected source λ values are correlated with same-fold
  diagnostics; fraction sensitivity across λ/group must be included
  descriptively, not assigned overconfident independent p-values.
- Some original participants can experience large harms despite
  positive group macro-AUC. Scope excludes deployment utility,
  calibration savings, and clinical benefit.

## Mandatory publication release gates
- Only verified, immutable metrics printed in main text and tables.
  Re-run focused LaTeX compile, inspect warnings/overfull figures,
  validate bibliography and page layout in final PDF.
- Complete author roster/affiliations, individual author
  contributions, ethics/data-use statement, funding, interests,
  data/code availability and journal AI-use disclosures after
  author approval. No auto-signing.
- Verify dataset exact licensing and download links, raw signal
  provenance, MOABB and Python dependency pinning. Physical identity
  cannot be inferred from released subject IDs alone.
- Finish official publisher style and independent full manuscript
  scientific review. Draft GitHub PR is NOT journal submission.
- V26--V30 Liu processed-record identity + 2x2 rerun must remain
  SEPARATE paper; do not add its 49 content groups to a 40+12
  EEG target sample-size summary.

## Current decision
An evidence-gated BSPC *research manuscript candidate* with a
clearer core mechanism. Not a demonstrated L8/L9 novel algorithm,
not a journal acceptance, and not cleared for publisher submission
until the above release gates pass.
