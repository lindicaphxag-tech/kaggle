# Default Regularisation vs REAL ACCEPTED BSPC papers, 2025–2026
Date 2026-10-13. Purpose: adversarial evidence-based head-to-head audit
for top-range BSPC research papers, not an acceptance prediction.

## Sources and evidence levels
A = accepted-author MANUSCRIPT/PDF directly checked; B = publisher full
web text/Methods/Results checked; C = publisher abstract/highlights only.
Never infer a study's unreported held-target information budget from
a headline. Do not directly compare ROC-AUC and accuracy or protocols
with/without target domain fine-tuning.

### Ten actually published comparators
| Study (published BSPC year/article) | Scientific problem / what is actually new | Breadth and reported result | Target-data budget / limits | Grade |
|---|---|---|---|---|
| Zhang et al., 2025, 102:107213, DOI 10.1016/j.bspc.2024.107213 | 62 wet-to-8 dry channel topology, GCN, teacher-student knowledge distillation and target fine-tuning | Author-accepted PDF: 54 source people, 22 target people plus BCI-IV2a; 8-channel CV 71.19%, cross-session 69.03% after transfer; before transfer 67.67/65.20. Table ablation Dense/Graph/Full 65.79/70.11/71.19, plus training fraction and channel aggregation sweeps. | EXPLICIT LABELLED target train-session samples for fine-tuning: NOT the same zero-target information budget as DefaultReg. Authors' accepted version 13 PDF pages, IEEE format pre-publication, not the BSPC publisher typeset layout. | A |
| Morvay et al., 2026, 113:108892, DOI 10.1016/j.bspc.2025.108892 | multi-source Wasserstein barycenter transport across new subjects, used with common interpretable models | 3 datasets, FOUR tasks across MI and VEP, four classifiers SVM/LDA/LR/MLP and CSP/xDAWN feature extraction; 79.59% balanced VEP accuracy, 53.97% BCIC4D2a MI accuracy; strong crossed-factor generalizability design. | Explicit target domain X (unlabelled) used to transport source to a new person: UNSUPERVISED DOMAIN ADAPTATION rather than strict zero-target domain GENERALISATION. Different task and metric; do not compare numbers directly. | B |
| Qin et al., 2026, 116:109535, DOI 10.1016/j.bspc.2026.109535 | within-versus-between person difference caused by channel spatial normalization; significance and morphology preservation | 9 ERP/classification paradigms (7 binary + 4 class + 16 class). Between-subject Cohen d=1.39, p<.001, NOT similarly positive within-person; effect size and multiple temporal/participant/ERP morphology panels; code/data accessible. | Different EEG task (ERP vs MI); more compelling reproducible multi-paradigm example of mechanism + concrete discriminative benefit. | B |
| Gil & Hernández-Sabaté, 2026, 120:110020, DOI 10.1016/j.bspc.2026.110020 | hierarchical EEG statistical validation prevents window- vs subject-level performance leakage | 6 deep models, seizure CHB-MIT, window/state/person split; recall/specificity 0.96/1.00 at window versus 0.77/0.90 at person with uncertainties. Generalized mixed-effects modeling and heterogeneous groups. | Same family of EVALUATION ESTIMAND paper but a different seizure application; its novel inference framework + model breadth set a higher bar than simply showing a standard ANOVA identity. | B |
| Murtaza et al., 2025, 105:107552, DOI 10.1016/j.bspc.2025.107552 | neural-signature, channel-correlation intersubject distance; use resting-state EEG to find source similarity | Trial, session, person-level distance measurement and statistical/ML validation. Magnitudes and target calibration rules NOT fully verified from accessible abstract, so no numerical ranking. | Closest "explanation/measurement with actionability" direct competitor: their measured distance supports source choice. Our mechanism currently descriptive without proved source-to-new-target decision utility. | C |
| He/Cao/Luo, 2026, 118:109762, DOI 10.1016/j.bspc.2026.109762 | cyclic deep representation/domain adaptation for cross-subject MI | deep representation transfer claim; full target split, ablation and effect magnitudes NOT verified in accessible abstract. | Source/target labels and unlabeled budget UNKNOWN at this evidence level; cannot rank by raw performance. | C |
| Liu et al., 2026, 112:108648, DOI 10.1016/j.bspc.2025.108648 | BioMIFormer neurobiology-inspired parallel temporal/spatial branches, transformer fusion | 2 BCI IV datasets 2a/2b, strong model/ablation and biological interpretability discussions; paper reports better performance, exact matched budget unknown. | Same target information budget unknown; within/across-person comparability must be verified. | B |
| DLSSNet, 2026, 120:109826, DOI 10.1016/j.bspc.2026.109826 | interpretable dual-layer state space and task relevance/attention; 125-ms neurophysiologically interpretable states | BCI-IV2a / High Gamma: 78.51% / 95.49% average classification ACCURACY; stability of sensorimotor rhythm maps across subjects. | No evidence this matches zero-target-person AUC protocol; cannot compare 78.51 with DefaultReg 0.75 AUC. Biological mechanism + predictive merit is an advantage. | B |
| Ren et al., 2026, 124:110650, DOI 10.1016/j.bspc.2026.110650 | ASTA-DAN adaptive spatiotemporal attention and subject adaptation | 2 MI-EEG datasets, cross-subject improved relative to their chosen baselines; precise magnitude and information budget NOT verified. | Domain adaptation may be transductive; don't count algorithm superiority against strict zero-target GID. | C |
| Zhang et al., 2025, 102:107213 (same GCN study, not duplicate sample) | calibration fraction experiment: 25/50/75/100% available target source-training-session trials; 12-trial setting and graph ablations | 25% -> 59.23% and 75% -> 65.06% and 100% -> 69.03% target scenario. This is an EXPERIMENTAL axis of paper #1, NOT another published comparator. | We have no matched budget-vs-benefit curve; our Z0 (zero target calibration) is narrower and preferable if it achieves the required decision performance. | A axis |

Other accepted related 2026 work: subject-independent ensemble representation
learning, BSPC 118:109504 DOI 10.1016/j.bspc.2026.109504; memory-augmented
meta-learning 2025 BSPC 110:108246 DOI 10.1016/j.bspc.2025.108246.
These establish competitive algorithmic families, but detailed protocols
were not verified; the live shortlist above prioritizes stronger evidence.

## Direct comparisons: NO dishonest leaderboard
| Axis | Real accepted-paper high-end evidence | DefaultReg PR #140 verified | Gap / what to do |
|---|---|---|---|
| Scientific motivation | EEG transfer, statistics, calibration, neuro-interpretability | important contradiction: proper predictive covariance logscore can mostly penalize changes invisible to *fixed* within-person rank AUC | Strong scientific question, retain and foreground |
| Mechanistic math | Gil hierarchical mixed effect analysis; Qin cross-paradigm normalization effect; neuro-signature distance related to subject variability | exact 3-factor weighted SPD quadratic identity, held-source anatomy: 98.59/99.29/99.77% invariant shift penalties in 3 exposed acquisitions | Need novelty beyond CLASSIC ANOVA/projection: prove selection diagnostic has a measurable prospective consequence |
| Training/data breadth | WBT 3 datasets/4 tasks/4 classifiers; Qin 9 paradigms; GCN 62->8 electrodes & multiple physical devices | 3 **post-outcome source diagnostic acquisitions** plus older frozen target families; many prospective cohorts only 3 channels, binary MI | Multiple independent target acquisitions, 15/22/62 channel and task sensitivity; do not count SOURCE validation as fresh target replication |
| Classifier/comparators | GCN includes CSP/FBCSP/EEGNet/ATCNet and graph/Dense ablations; WBT crosses classifiers | core EEG LDA + LW/OAS/MDM/TangentLR; CSP3 matched on later GID held Lee only, FBCSP not matched over original frozen target suites | Need precommitted strong matched budget baseline matrix and target AUC wins before new-method superiority claims |
| Actionability | Qin normalized preprocessing improves BETWEEN decoding d=1.39; WBT gives practical performance gains with target X; neural distance offers source selection | three-factor decomposition is primarily explanatory; V31 Cho40 no gain, V32 Dreyer alpha0/no gain | Single decision rule must work prospectively on at least a new acquisition; negative new rule is not promoted |
| Neural plausibility | DLSSNet 125ms SMR patterns, BioMIformer anatomical routing | mathematical feature projections but no validated sensorimotor topography/stability claim | Carefully study alpha/beta bands, channel localization C3/C4 and across-session robustness WITHOUT invention of physiological proof |
| Statistical quality | Gil subject-stratified hierarchy; Qin 9-paradigm paired effects; GCN ablation and p tests | original p values+participant adverse effects, exact sign flips, immutable per-person audit, rigorous failures public | Our advantage: transparency; missing pre-registered independent across-acquisition *predictions*, multiplicity-aware head-to-head deep comparisons |
| Implementation/budget | deep papers higher GPU cost; WBT cheap interpretability & target unlabeled | source fixed tangent LDA very low compute and strict Z0 budget | Measure actual CPU wall time, memory, energy and equal-score success; low compute is only advantage if comparable utility |
| Manuscript | published polished figures, well-supported experiments, complete authored declarations | 18-page compile success, editable source, cohesive scientific message; prior abstract source-only percentages strong | Real referee review, editorial condensing, data/code/ethics/CRediT declarations, designed Fig 1 pipeline / Fig 2 mechanism / Fig3 per-person harms, no invented results |

## Rigorous counterexample to misleading claims
- Morvay 79.59% **BALANCED ACCURACY** is VEP, not MI. MI BCIC4D2a
  result 53.97% accuracy and can use unlabeled target X. DefaultReg
  ROC-AUC about .75 on binary MI three-channel/other acquisition is
  NOT numerical superiority.
- Zhang cross-dataset 71.19% is 8-channel ACCURACY after target-labelled
  FINE TUNING, not held-person zero-target ROC-AUC.
- DLSSNet 95.49% High Gamma ACCURACY is not direct comparison
  to zero-calibration participant-macro ROC-AUC.
- Qin nine paradigms are ERP not MI; robustness breadth is the
  meaningful comparison, not AUC.
- Three DefaultReg diagnostic cohorts all have previously EXPOSED
  target results. Forty-person Cho plus twelve-person Dreyer are
  not independent successful methods. Do not pool for a novel effect.

## Reviewer demand — objective ACCEPT/REJECT conditions
R1. Demonstrate source-statistic -> future-target transfer RISK PREDICTION
    on at least ONE fully untouched external new acquisition with
    frozen binary direction/failure thresholds before target access.
R2. Matched source-only, ZERO target X or labels, source dimension /
    participant count / filtered epochs and channels for each
    comparator; report CSP and filter-bank as well as tangent-LDA,
    Riemannian MDM, Tangent LR; no cross-budget scoreboard.
R3. One algorithm candidate with unequivocal nonparallel source-fitted
    classifier and risk improvement over original, including each
    held-source person's harm; both across-person and
    across-acquisition reproducibility essential.
R4. For the methods/estimand paper lacking R3, foreground
    falsifiability and the *scientific measurement finding*,
    include exactly interpreted classic mathematics and negative
    method trials, and avoid SOTA performance claim.
R5. Compute analytical decomposition and share shifts per
    individual group with honest influence sensitivities and
    bootstrap only at group, never at trial independence.
R6. Include uncertainty/status for Lee2019 source/held B SHA
    mismatch before publication; a failed source lock is a
    pipeline reproducibility failure and NOT an outcome.
R7. All citation-backed paper claims have DOI and status; if
    only abstract inspected, do not attribute unverified
    detailed ablations/statistics to paper.

## Editorial positioning and current judgement
DefaultReg is stronger than a bare hyperparameter study because it
demonstrates a reproducible mismatch between different estimands
with an exact mechanism and multiple honest negative trials.
It is NOT established as superior to BSPC top-range accepted
papers in substantive NEW ALGORITHM, multiple-paradigm replication,
and prospective target utility. It is potentially competitive for
a focused BSPC methods/measurement contribution if presented as
an estimand-audit rather than a SOTA domain-generalisation model.
No numerical acceptance probability is justified.

## Publisher/author sources; verification levels
A. Author accepted full PDF, Zhang et al. 2025:
https://eprints.whiterose.ac.uk/id/eprint/236823/1/Highlighted-accepted%20on%2016%20Nov%202024.pdf
B. Morvay et al. 2026:
https://www.sciencedirect.com/science/article/pii/S174680942501403X
B. Qin et al. 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426000893
B. Gil et al. 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426005744
C. Murtaza et al. 2025:
https://www.sciencedirect.com/science/article/pii/S1746809425000631
C. He et al. 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426003162
B. BioMIFormer 2026:
https://www.sciencedirect.com/science/article/pii/S1746809425011590
B. DLSSNet 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426003800
C. Ren et al. 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426012048
C. ERL 2026:
https://www.sciencedirect.com/science/article/pii/S1746809426000583
C. MAgML 2025:
https://www.sciencedirect.com/science/article/pii/S1746809425007578
