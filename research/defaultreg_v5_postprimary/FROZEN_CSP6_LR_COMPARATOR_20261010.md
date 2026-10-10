# Default Regularisation — post-primary CSP6+LR comparator extension

Date: 2026-10-10.

Evidence role:
POST-PRIMARY CLASSICAL COMPARATOR ONLY.

This extension is added after the v5 BNCI2014_002 and BNCI2015_001 target
outcomes are known. It must not be used to upgrade the prospective status of
H1/H2/H3 or the Validity Frontier.

The comparator design itself reuses the fixed covariance-form CSP6 +
LogisticRegression(C=1) specification that appeared in the closed, unexecuted
PRE-DATA PR #98 before the v5 outcomes. No outcome from PR #98 existed.

## Cohorts

Use exactly the authoritative v5 cohort loaders and preprocessing:
- BNCI2014_002, all 14 participants, native 15-channel montage;
- BNCI2015_001, all 12 participants, native 13-channel montage and every
  returned session;
- right-hand vs feet;
- 8-30 Hz;
- official 0-5 s task interval;
- resample 160 Hz;
- reject_by_annotation=False;
- same numerical SPD floor;
- full outer leave-one-participant-out.

## CSP definition

For each outer source pool:
- compute source class-mean trial covariances C0 and C1;
- solve the generalized eigenproblem
  C1 v = gamma (C0 + C1 + eps I) v;
- rank filters by |gamma - 0.5|;
- retain the six most discriminative filters, or all filters if p<6;
- project each trial covariance into the CSP basis;
- take log normalized diagonal variances;
- fit LogisticRegression(C=1, lbfgs) on source participants only.

No target statistics or target labels enter CSP fitting or LR fitting.

## Score chronology

For every outer target:
- generate CSP score vector first;
- archive the concatenated target score bundle and SHA256;
- only then calculate participant/session AUC.

The raw EEG file manifest is regenerated using the authoritative v5 loader.

## Reporting

Report:
- participant-macro CSP6+LR AUC;
- per-participant AUC;
- session AUC for BNCI2015_001 descriptively;
- target score bundle SHA256;
- raw-file identities and participant/trial denominators;
- runtime.

Pairing to v5 source-macro / covariance / LW is performed later from
machine-readable absolute metrics. No hand-entered gain column is generated
here.

Status:
POST_PRIMARY_COMPARATOR_FREEZE.
