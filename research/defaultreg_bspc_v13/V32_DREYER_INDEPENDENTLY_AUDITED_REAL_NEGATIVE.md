# V32 Dreyer2023A full real EEG negative: independent immutable audit
2026-10-13 | research evidence | NOT a successful new algorithm.

## Provenance
- First original source-locked experiment GitHub Actions 38299747910,
  all twelve source EEG groups fixed before any held target I/O,
  SOURCE SHA256 9bfc60e1da10575315917aa6662532198a892dfa99f045341bb71adde58a04b9.
  That workflow processed held IDs 13–17 and failed on an OSF download
  timeout for held ID 18. It produced no complete target summary.
- Strict original-SHA transport-only resumption GitHub Actions 38300422475:
  unchanged numerical code, original source groups, 12 target IDs 13–24,
  predeclared hyperparameters, MOABB 1.7.2, channels C3,Cz,C4, 8–30Hz.
  Reproduced byte-identical original SOURCE SHA before target I/O.
  No newly independent experiment is claimed.
- Independent source structural audit CI 38300740581 PASS.
- Independent complete target artifact audit CI 38300999147 PASS:
  rederived all 12 people/methods, exact 2^12 signed permutations,
  subject bootstrap and individual harms without retraining EEG.

## Source-only method novelty STOP (before target results)
Original source-macro grid: lambda=0.3, alpha=0;
V32 risk selector: lambda=0.1, alpha=0;
scalar risk ablation: lambda=0.1, alpha=0;
correct predictive Gaussian source NLL: lambda=0.1, alpha=0;
best mean-AUC matrix candidate: lambda=0.3, alpha=0.
Source-group between-class contrast covariance raw trace 0.0236251334;
finite-trial class-mean sampling-noise trace 0.0083543322;
PSD-debiased heterogeneity trace 0.0159187814.
The V32 chosen classifier had direction cosine 0.9999999999999999
against scalar LDA at the SAME lambda and identical AUCs as the
scalar-risk and proper-Gaussian-NLL choices.
The anisotropic matrix operator was not selected at all.

## All 12 held targets (2,880 EEG motor imagery trials)
| Model | Person-macro AUC |
| --- | ---: |
| Original source-macro lambda0.3 | 0.766452546 |
| V32 risk lambda0.1, alpha0 | 0.765364583 |
| Scalar risk ablation | 0.765364583 |
| Proper source-mean Gaussian NLL | 0.765364583 |
| Mean-AUC candidate alpha0 lambda0.3 | 0.766452546 |
| Riemannian MDM | 0.770277778 |
| Tangent LogisticRegression | 0.762719907 |
| Ledoit–Wolf | 0.764021991 |
| OAS | 0.763929398 |
| Historical class-centred covariance | 0.763975694 |
| Fixed lambda1 | 0.744699074 |

V32 risk minus original grouped source macro mean AUC
= **-0.001087963**, descriptive person-bootstrap 95% interval
**[-0.002968750,+0.000879630]**, exact two-sided paired sign-flip
**p=0.3134765625**, 5 improved persons, 7 declined,
0 with decrease larger than .02 AUC.
No predeclared confirmatory MDM-vs-source-macro p value is claimed.

Signed paired V32 minus original group-macro:
13 -0.0004861; 14 -0.0060417; 15 -0.0041667;
16 -0.0011806; 17 +0.0001389; 18 -0.0058333;
19 +0.0063889; 20 -0.0036806; 21 +0.0009028;
22 +0.0025000; 23 -0.0021528; 24 +0.0005556.

## Scientific decision
BOTH structural novelty and held-target improvement criteria FAILED.
This is not evidence of a new superior anisotropic EEG decoder.
The source variance-component method and new AUC expectation derivation
may be retained as scoped analysis, but not promoted to BSPC headline.
NO post-outcome retuning of lambda/alpha/beta/participants/channels may
convert this held Dreyer acquisition into untouched method validation.
Focus the main BSPC paper on previously frozen criterion-mismatch
research and honest selection failures. Keep Liu recording identity
V26–V30 in a SEPARATE reproducibility paper.
