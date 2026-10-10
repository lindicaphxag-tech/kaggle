# CSP6 comparator implementation correction

Date: 2026-10-10.

Evidence role:
POST-PRIMARY comparator implementation correction.

The first BNCI2014_002 CSP6+LR execution in workflow run 38063284077 failed
before any target AUC was produced.

Failure:
the CSP feature projection intended to compute

    W^T C W

used the NumPy einsum string

    "pi,nij,jq->npq"

for W with shape [p,k]. This incorrectly interpreted the first W dimension as
the output component dimension and produced a broadcast mismatch whenever
p != k.

Correction:
replace only the tensor index string with

    "ip,nij,jq->npq"

which implements W^T C W for W:[p,k], C:[n,p,p].

Frozen scientific choices are unchanged:
- CSP6 component count;
- generalized eigenproblem;
- |gamma-0.5| filter ranking;
- log normalized variance features;
- LogisticRegression(C=1, lbfgs);
- v5 data loader and preprocessing;
- outer LOPO;
- target-information budget Z0;
- score freeze before target AUC;
- all cohort participants/sessions;
- all reporting endpoints.

No BNCI2014_002 CSP target AUC existed before this correction. The failing
artifact contained the protocol file only.

This correction does not alter the already completed prospective v5 H1/H2/H3
or Validity-Frontier evidence. The CSP extension remains POST-PRIMARY.

Status:
IMPLEMENTATION_ONLY_INDEX_CORRECTION_BEFORE_COMPARATOR_OUTCOME.
