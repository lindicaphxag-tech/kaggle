# P1 / JNE — doctoral-level upgrade, hypothesis & preregistration draft (NOT executed)
Date: 2026-10-13 | Status: RESEARCH DESIGN ONLY; original JNE article is NOT ready to claim breakthrough.

## Existing evidence (do not rewrite)
- Frozen EEGMMIDB n=20 source-macro vs **historical class-centred CovScore** +0.0561 AUC, Holm p=.00597, with five >.02 harms. This is NOT a benefit versus proper predictive likelihood.
- Retrospective corrected source-mean predictive NLL contrasts: Shin2017A n29 −.000307; BNCI2014_002 n14 +.006819; BNCI2015_001 n12 +.035989 (paired p=.1572); BNCI2014_001 n9 +.009265. No consistent positive sign across independently prespecified data.
- Fixed-coordinate exact decomposition: NLL additional mean-shift penalty = common-location + score-orthogonal class-contrast + score-aligned class-contrast. The first two are AUC-invariant for fixed w / coordinates, NOT under target-refitted tangent reference.
- This is a score/estimand correctness contribution, not a newly invented shrinkage method or a proven domain-generalisation algorithm.

## Doctoral-level unanswered question
When, using SOURCE PARTICIPANTS ONLY, can we forecast that a proper predictive-density selector is overly sensitive to group-specific AUC-invariant shifts, and will a reliability-aware selector avoid large worst-person harms on fresh unseen EEG?

## Proposed mechanism: nuisance-discordance-triggered SOURCE choice (working hypothesis)
For each inner-source held participant g and fixed fitted score w and covariance A(lambda), compute the established exact nonnegative nuisance components:
 D_common(g,lambda), D_orth(g,lambda), D_align(g,lambda);
 D_total = D_common + D_orth + D_align.
Define source-only nuisance fraction q_g(lambda)=(D_common+D_orth)/(D_total+eps) with an explicit zero-denominator convention.
A candidate policy family:
  S_eta(lambda)=group_macro_auc(lambda)-eta * group_CVaR_alpha[(q_g(lambda)-q0)_+ * |AUC_g(lambda)-macro_auc(lambda)|]
with eta>=0 and thresholds eta,alpha,q0 tuned entirely in NESTED SOURCE PARTICIPANT folds.
IMPORTANT: this is a new *proposal*, NOT a theorem, performance result, or proven genuinely new functional in the literature. The nuisance quantity is established algebra; CVaR / robustness are established tools. Verify originality before claiming it.
Predefine a simpler, falsifiable alternative: use q to predict sign and magnitude of proper-NLL-versus-macro-AUC discrepancy, without selecting a new decoder.
Prove: (i) the original exact decomposition under explicit fixed-coordinate hypotheses; (ii) the trigger and estimator are measurably invariant to constructed common shifts; (iii) provide counterexamples under re-fitted references and arbitrary target laws; (iv) characterize finite-source sensitivity (no target-free universal domain guarantee).

## Required experiment — not completed by this document
1. Choose >=3 acquisition families NOT used to invent or revise the decision rule (genuine unopened outcome status logged); prerecord per-dataset version, participant IDs, sample/channel geometry, seed, labels, exclusion of content aliases. If no genuinely unexposed acquisition is available, downgrade claim to retrospective and acquire different data.
2. Compare SOURCE macro 27-grid, correct predictive NLL (source-trained means), historical CovScore correctly labelled, LW, OAS, fixed lambda, logistic / tangent MDM and nuisance-trigger policies at MATCHED preprocessing, target access and cost. Include a naive CVaR selector to isolate the novelty of the nuisance mechanism.
3. Entire grid/hyperparameters/source-fitted geometry selected with nested group CV. Freeze SHA of design, features, fitted states and estimand prior to any test outcome read.
4. Primary end points: subject-macro target ROC-AUC and fraction with delta AUC<-0.02 versus best predefined classical comparator; secondary: CVaR harm tail, calibration of nuisance-risk prediction, runtime and sensitivity to channel count.
5. Bootstrap at acquisition/person hierarchy, disclose overlapping LOPO source fits, effect uncertainty and multiplicity. Always report negative cases and unintended harms.
6. At least one multi-model decoder test and one nonmotor EEG paradigm to demonstrate whether it extends past tangent LDA. NOT all data can be called new if previously explored.

## Go / no-go gate for exceptional JNE claims (standards proposed, not facts)
A: no estimand leakage or historical CovScore mislabel;
B: a stable, preregistered nuisance-risk prediction beyond ordinary CVaR and proper NLL on multiple genuinely unseen datasets, with narrow enough uncertainty to exclude a trivial null;
C: measurable per-person harm improvement without hiding AUC utility or violating Z0; report tradeoff Pareto frontier;
D: independent reimplementation, open protocol and sensitivity counterexamples;
E: distinct from P3 transfer-policy generalities and P2 data-provenance findings.
If B/C fail, publish an estimand audit with narrower claims; do not retrofit the primary hypothesis.

## Accepted-peer anchors and new-frontier context (not claims of parity)
- Eder et al. JNE 2024 DOI 10.1088/1741-2552/ad6793 (multi-paradigm Riemannian / CNN decision implications).
- Seibel et al. JNE 2026 DOI 10.1088/1741-2552/ae453d (data marginal-value predictive study).
- Xiong et al. ICML 2026 EEG-FM-Bench: https://proceedings.mlr.press/v306/xiong26f.html (14 datasets, 10 paradigms; outside-journal high bar).
A paper cannot claim top 10% journal placement without a predeclared same-topic accepted-paper sample and external validation.