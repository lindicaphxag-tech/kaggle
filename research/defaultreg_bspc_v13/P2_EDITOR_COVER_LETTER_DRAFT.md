# P2 BSPC — editorial cover letter, AUTHOR REVIEW ONLY — DO NOT SEND UNSIGNED

Dear Editor-in-Chief,

Please consider our original research manuscript, **"Released EEG identifiers can share numerical recording contents: A full-dataset audit and matched evaluation of cross-subject decoding,"** for *Biomedical Signal Processing and Control*, subject to the authors' final approval.

This study identifies a technical failure mode in biomedical signal benchmarking: different public participant identifiers can index identical processed EEG signal contents even when conventional participant-wise evaluation is used. In the publicly released Liu2024 motor-imagery dataset, a pinned pipeline produced 2,000 labelled epoch instances but 1,960 unique exact contents. A separate, exhaustive census of all 50 published EDF files (1,225 unordered pairs) found 29/29 fully identical time-varying EEG digital streams for released IDs 4/5. Two further pairs, 47/49 and 11/15, shared 6,957/9,280 and 171/9,280 aligned native EEG blocks, respectively, without full-stream equality.

We distinguish numerical release-layer signal identity from clinical patient identity or original acquisition provenance. Corresponding separately published source MAT arrays were not entirely identical, and the current Figshare code archive does not fully specify the MAT-to-EDF transformation. We do not infer that two clinical participants are the same patient.

In a matched 49-content-group retrospective analysis, mean ROC-AUC changed from 0.594000 across 50 nominal IDs to 0.588673 under content-group reporting weights and to 0.582551 with group-disjoint source refitting and source-only parameter choice. The descriptive total difference of -0.011449 must not be interpreted as an identified causal leakage penalty. The matched refit-related mean -0.006122 was concentrated in one overlap-affected held group (-0.2625); the other 48 averaged -0.000781. Fixed-classifier and preprocessing sensitivity tests on the same acquisition support localisation, not population generalisation.

The manuscript contributes an independently traceable biomedical signal-content census, an explicit distinction between clinical, released-ID and digital-content evaluation units, and a matched fit-by-selection audit. We provide the original source products' provenance limitations and do not claim a superior EEG decoder or a new cryptographic method. Four other public EEG datasets were screened for exact processed cross-ID trial overlap under specified processing paths; negative screens are not proof of population-level absence of near-overlap.

The research is distinct in question and primary outcomes from our separate ongoing work on EEG regularisation scoring objectives. Any shared datasets, code or concurrent related manuscripts will be disclosed in the final submission.

**Required before sending:** all authors' agreement on title and findings; authorship/order/affiliations; originality and exclusive submission; ethics and secondary-data terms; funding, CRediT and conflicts; AI-assisted preparation declaration; code/data repository public release; and documented data-custodian correspondence or a clear account of unresolved original provenance. None of these statements is represented as approved here.

Sincerely,

[CORRESPONDING AUTHOR — approval/signature required]
