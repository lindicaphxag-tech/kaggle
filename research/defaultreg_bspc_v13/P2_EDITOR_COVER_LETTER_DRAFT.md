# P2 — Editor cover letter (DRAFT; NOT TO SEND WITHOUT AUTHOR REVIEW)
Dear Editor,

We submit for consideration our original research manuscript,
"When EEG Participant IDs Share Recording Contents: A Matched
Audit of Validation Denominators, Model Refitting and Regularisation."

The work addresses a distinct signal-data reliability issue:
released EEG participant identifiers need not guarantee independence
of processed numerical recording contents. In the publicly
distributed Liu2024 motor-imagery acquisition, an exactly
predefined EEG preprocessing pipeline reveals complete
cross-identifier overlap for one 40-trial record pair. We
evaluate the consequences with fixed 49-content-group held
targets and an explicit 2-by-2 analysis separating reporting
denominator, model refitting and frozen shrinkage choice.

Additional real public-data checks have now reproduced the exact
40-trial content alias under a second preprocessing configuration
(6--35 Hz, 128 Hz). Two fixed classifier objectives, tangent logistic
and ridge classification, each retained a pronounced refit-related AUC
change for the aliased recording group while showing near-zero mean
change over the other 48. These are retrospective analyses on the
same acquisition, not new patients, independent cohorts or a causal
population leakage estimate.

An exploratory screening of all 1,225 trial-aligned released-ID recording pairs also detected two further, previously unrecognized, very high full-record similarities: 47/49 (0.99999999166) and 11/15 (0.99999998340). The latter two are candidate near-overlaps, NOT exact byte-matches or independently confirmed duplicate physical acquisitions. We do not extend the primary 49 exact-content-group AUC estimand to an asserted count of independent patients.

We report the full numerical observations without equating
recording-content identity to clinical patient identity, nor
claiming that arithmetic decomposition identifies a causal
leakage effect. Two other public acquisitions were examined
as negative controls under their pinned pipelines. The
code and independently checked numerical receipts support
reproduction of the exact estimands.

This study is separate from our work on EEG covariance
prediction-score versus discrimination-validation estimands;
the primary measurements, scientific question and
conclusions of this submission are recording provenance
and the validation unit rather than classifier improvements.
Overlapping datasets and related manuscripts will be
fully disclosed to the journal. A data-custodian clarification and author-approved
independent raw-record verification are prerequisites to finalizing
the description of the public clinical dataset.

[Authors, affiliations, rights, originality, exclusive-submission
declaration, funding, ethics approvals/exemptions, and
corresponding author must be completed by actual authors.]

Sincerely,
[CORRESPONDING AUTHOR TO BE VERIFIED]
