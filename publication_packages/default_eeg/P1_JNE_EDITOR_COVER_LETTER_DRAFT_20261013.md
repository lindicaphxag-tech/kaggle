# JNE Editor Cover Letter — Paper 1
DRAFT ONLY · Review/approve with ALL authors before submission · 2026-10-13

Dear Editor,

We respectfully submit for consideration our original research manuscript,
"When Predictive Likelihood and Cross-Subject EEG Discrimination Disagree:
An Exact Score-Audit Study," as a Paper in the Journal of Neural Engineering.

We study a narrow but practically consequential methodological question:
why can a covariance model receive a worse predictive Gaussian likelihood
while producing unchanged within-participant pairwise EEG ranking? The
contribution is an explicitly validated measurement study, not a claim
to state-of-the-art accuracy. We distinguish the historically used
validation-class-centred covariance score from a proper source-class-
mean predictive Gaussian score, then give an exact finite-sample
decomposition of the extra mean-shift penalty. We separate common
participant displacement, score-orthogonal class differences and
score-aligned class differences, reporting the exact assumptions
under which parts of that penalty are irrelevant to the ROC-AUC
ordering of a FIXED linear decoder.


An additional retrospective 12-person BNCI2015_001 experiment now
compares the corrected source-trained predictive Gaussian NLL selector
with the historical validation-class-centred score, source-macro AUC,
Ledoit–Wolf and OAS under identical participant-grouped source folds.
Source-macro outperformed the correct NLL selector by +0.035989 AUC
on average, but three participants lost >0.02 AUC and a two-sided
exploratory exact paired sign-flip check gave p=0.1572. This study
reuses an earlier exposed cohort; it is not new prospective confirmation.
Run: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38321486917

Our evidence includes prospectively frozen individual-participant
EEG outcomes, a comparison of source-selection criteria with
both positive and harmful participant-level outcomes, and an
independently audited SOURCE-only explanatory analysis in
three public EEG acquisitions. The reported 98.59%, 99.29% and
99.77% fractions concern only the additional predictive-score
mean-shift penalty; they do NOT measure EEG information,
classification accuracy improvement or unseen-domain utility.
We explicitly report the unsuccessful follow-up regularisation
operators and refrain from claiming a universal source-only
transfer improvement.

We believe this question is of interest to readers designing
neural signal decoders who need to know precisely what their
source-participant regularisation validation metric measures.
We provide original frozen/protocol chronology, person-level
heterogeneity, calculation and reproducibility links to permit
independent inspection. The related article on EEG RECORDING
CONTENT identity concerns a different primary question and is
not simultaneously being presented as a repeat of the same
measurement result; any overlapping data/resources will be
explicitly disclosed.

[Insert verified public CODE release and immutable evidence links here.]
[Insert author-provided prior-publication and exclusivity statement.]
[Insert corresponding author details, affiliations, ethics/data use,
funding, conflicts, CRediT contribution and AI use disclosure.]
No such statements are made without the actual authors' approval.

Sincerely,

[Corresponding author — to be completed and approved]

EDITORIAL GATE: IOP Journal of Neural Engineering Paper normally
<= 12,000 words (~14 journal pages), and requires strong evidence
for hypothesis/technique. Ensure final PDF, supplementary statements
and disclosures comply before sending:
https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/about-journal-neural-engineering/
https://publishingsupport.iopscience.iop.org/journals/journal-of-neural-engineering/
