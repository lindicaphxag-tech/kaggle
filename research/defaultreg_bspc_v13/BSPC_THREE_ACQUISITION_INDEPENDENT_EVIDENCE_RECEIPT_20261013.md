# Default EEG / BSPC — independently audited three-acquisition mean-shift anatomy
Date: 2026-10-13 | original source-only EEG artifacts | all prior targets EXPOSED.
Classification study, not clinical use, submission, fresh target confirmation
or independently preregistered performance superiority.

## Scientific question
The earlier executed CovScore estimated held-SOURCE participant
class-centred residual covariance likelihood, not true
source-trained-class-mean predictive Gaussian NLL.
The EXACT added penalty decomposes into three nonnegative quantities:
 P_loc (both-class common participant translation);
 P_orth (class mean-contrast change orthogonal to fitted LDA score);
 P_score (contrast displacement visible to fitted discriminant).
P_loc and P_orth are invariant with respect to the *fixed* fitted
linear classifier's within-person ROC-AUC; P_score CAN affect its
class mean score gap, but is not by itself sufficient to predict AUC
direction or magnitude because within-person score variance can shift.
This is weighted SPD quadratic algebra, not an unprecedented theorem.

## Actual source-only EEG cohort scope
Both 3-channel studies: same left/right MI task, 8–30 Hz,
160 Hz, C3/Cz/C4, 6 tangent coordinates; original source-only
participant round-robin CV, exactly source-fitted Riemannian reference.
Native BNCI: right-hand versus feet MI, 15 native EEG channels,
120 tangent coordinates, same original grouped source CV path.
These are three separately distributed acquisitions, but all target
results had ALREADY been exposed in related prior studies.
NO HELD TARGET people are read in this new analysis.
All source-validation groups are treated as SOURCE labelled
data, with class labels allowed for source-validation scoring.

## Metrics at the lambda selected by CORRECT SOURCE predictive NLL
| Source acquisition | G | Source trials | p | lambda pred | Location % | Orthog % | Score-aligned % | AUC-invariant combined % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cho2017 | 12 | 2,480 | 6 | 0.10 | 97.3126 | 1.2758 | 1.4115 | 98.5885 |
| Dreyer2023A | 12 | 2,880 | 6 | 0.10 | 98.9209 | 0.3700 | 0.7091 | 99.2909 |
| BNCI2014_002 | 14 | 2,240 | 120 | 0.30 | 98.1845 | 1.5828 | 0.2327 | 99.7673 |

ALL percentages are of the same mean-shift NLL correction term,
NOT percent EEG variance, EEG discriminant information, trial
accuracy or expected subject-transfer benefit.

Original SOURCE full-grid macro AUC selected lambda:
Cho 0.50; Dreyer 0.30; BNCI 0.90.
Historical validation CLASS-centred CovScore lambda:
0.0 on all three sources. The 'contrast-only' diagnostic also
selected 0.0 here, but is NOT proper density prediction or
an accepted novel better classifier.

## Independent, no-refitting SHA-level verification
Original source-only EEG CI for Cho2017 and Dreyer:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38303616022
Original native 15-channel BNCI CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38303571392
Independent frozen-artifact numerical audit:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38304579126

Original JSON SHA256 (immutable CI artifact bytes):
- Cho: db059d83cf23e43ea916bb1194f578ce825157d07ca38eec430aeb1d45f53cbc
- Dreyer: 4fb322119ec62040a7cf0cf8bd404c475a395b6cfd1c9dc9aa05f97fe3aee932
- BNCI: 7a9140ea497a047c7068981aa60ea9781663945cd049d81fb7dadc3957db89af

Independent CI re-read all 3 raw result artifacts, compared byte hashes,
recomputed every feasible lambda and every held-SOURCE group row,
checked C_pred-C_centered=P_loc+P_score+P_orth, checked group arithmetic
against output row means and original source-macro lambdas; SUCCESS.
At the pre-existing source-predictive selected lambda:
- Location contribution exceeded score-aligned contribution on
  all 12/12 Cho, 12/12 Dreyer, 14/14 BNCI grouped held-source people
  (38/38 source group rows; training folds overlap).
- Descriptive leave-one-group-out REAGGREGATION of the same
  source-fold outcomes (no model retraining):
  * Cho AUC-invariant share min/max 98.3784–99.3536%
  * Dreyer share min/max 99.1525–99.6005%
  * BNCI share min/max 99.7355–99.8569%
  These intervals are NOT CIs or independent replication p-values.

## Relation to target transfer
Original frozen EEGMMIDB S021–S040 target n20 historical CovScore
comparison +0.0561 AUC Holm p=.00597, with 5/20 harmed >.02 AUC.
BNCI and separately frozen Shin datasets have the previously documented
distinct primary inferential families; historical CovScore is
NOT proper source-trained-mean likelihood.
Weibo2014 10-person continuous-vs-grid test was a transfer tie
(AUC .73141), source-only new SCE Cho40 combined gain near zero,
and the noise-debiased anisotropic V32 operator selected alpha0 and
had no Dreyer improvement. These null outcomes remain visible.
The three-factor results DO NOT establish a superior new regularizer,
unseen-target prediction, transductive calibration, or an L8/L9
algorithm. V26–V30 recording-content duplicate-group auditing is
a SEPARATE second technical paper.

## Release state
Focused BSPC manuscript is a research draft. Author, author
contribution, ethics/data-use, funding, competing interests,
data/code and AI-use statements require author approval.
Publisher submission has not occurred; acceptance is not guaranteed.
