# Public EEG data steward clarification — DRAFT, NOT SENT
Paper 2: When EEG Participant IDs Share Recording Contents
2026-10-13

Subject: Request for clarification about cross-ID EEG signal-content overlap
in the public Liu2024 acute-stroke motor-imagery EEG release

Dear Dataset Curator / Corresponding Author,

We are preparing a methodological reproducibility study about the
independence of released EEG recording contents when evaluating
cross-subject machine-learning models. We are writing to request
technical clarification before finalizing our manuscript's
description of an apparent cross-identifier content overlap.

Using the public dataset through a fixed MOABB 1.7.2 preprocessing
pipeline, we observed that released identifiers 4 and 5 yielded
identical numerical contents and labels across all 40 processed
motor-imagery trials in our pipeline. In a separate public EDF
signal comparison, the corresponding recordings produced
numerically matching values in 32 of 33 MNE-interpreted channels;
the remaining differing channel had an empty reported name.
These are observations about PUBLIC DIGITAL SIGNAL CONTENTS,
not conclusions about the identity of any participant.

Could you please clarify whether these two released identifiers
are intended to refer to independent recordings, whether
there are known repeated files or export/copying conventions,
and whether an updated release, metadata or correction note
exists? We would welcome any public documentation or technical
description that could help us correctly interpret this overlap.
We will not infer clinical participant identity or intent
from numerical signal equality.

For reproduction we can provide a small script specifying
public ID labels, filtering and resampling, the exact
processed trial-content hashes, and an audit receipt without
sharing additional private participant information.

Our manuscript separately measures how a content-safe
49-recording-group evaluation differs from a nominal
50-identifier analysis under a matched fixed-target
two-by-two source fit versus frozen parameter-choice design.
Those differences are descriptive software comparisons,
not causal estimates of an individual patient's data leakage.

We would appreciate an opportunity to incorporate any
technical clarification, and to cite the appropriate
data update or provenance statement if one is available.

Sincerely,

[Actual author/corresponding author — ONLY after approval]

STATUS: UNSENT. Do not send via connected mail or publish
as a claim of duplicate clinical patients. Verify the public
dataset license and metadata, exact signal equality on a
second independent software path, and all authors' approval
before contacting the dataset maintainers.
