# ClinGeom vs Medical Image Analysis / MICCAI / ICCV accepted papers — editorial contrast

**Current decision: NO-GO for superiority-based IEEE TMI or Medical Image Analysis segmentation work.** Retain the negative cohort replication and redesign the contribution rather than drawing prettier fusion diagrams.

| Peer paper (confirmed accepted/published) | What it does | Why we do not yet compete |
|---|---|---|
| Santhirasekaram et al., *A geometric approach to robust medical image segmentation*, **Medical Image Analysis 97, 103260 (2024)**, https://pubmed.ncbi.nlm.nih.gov/38970862/ | Shape equivariance + texture-invariant features and a learned shape dictionary; proves/discusses group-order robustness and reports high-performing prostate and cardiac MRI segmentation | Our geometry-space correctness is a mechanism, **not a trained feature representation with positive cross-domain segmentation** |
| Wu et al., *SicTTA: Single image continual test time adaptation for medical image segmentation*, **Medical Image Analysis (2025)**, https://www.sciencedirect.com/science/article/pii/S1361841525004050 | Source-friendly images, aligned-batch enhancement, feature fusion; comparative tests against 7 methods on fundus/heart | Any new ClinGeom feature repair must beat a strong *no-extra-fusion / canonical-first* control, and should also compare with credible adaptation/fusion methods |
| Su et al., *UniFuse: A Unified All-in-One Framework for Multi-Modal Medical Image Fusion Under Diverse Degradations and Misalignments*, **ICCV 2025**, https://openaccess.thecvf.com/content/ICCV2025/html/Su_UniFuse_A_Unified_All-in-One_Framework_for_Multi-Modal_Medical_Image_Fusion_ICCV_2025_paper.html | Joint modeling of misalignment, restoration and fusion across datasets, not mere affine bookkeeping | Need a genuinely new pre-feature-mix correction **and measured full-volume downstream benefit**, including cross-architecture and misalignment conditions |
| Zhang et al., *Progressive Test Time Energy Adaptation for Medical Image Segmentation*, **ICCV 2025**, https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Progressive_Test_Time_Energy_Adaptation_for_Medical_Image_Segmentation_ICCV_2025_paper.html | Adaptive shape energy under dataset shift | Distinguish a *typed spatial coordinate contract* from generic shape-prior/energy adaptation and demonstrate novel utility |

## Non-negotiable facts from real experiments

Two original author-run official pretrained-model CI workflows:
- First 6 persons: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827612217
- Next 6 source-ID-disjoint persons: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37975340905
- Separate exact reproducible raw-JSON check (source SHA-256, checkpoint identity, patient hashes): https://github.com/lindicaphxag-tech/kaggle/actions/runs/37983549282

| Mean patient Dice | First 6 | Next 6 | Combined 12 |
|---|---:|---:|---:|
| Strong canonical-first | 0.186391697 | 0.115957955 | **0.151174826** |
| Frozen reliability-gated prediction fusion | 0.186314510 | 0.115891135 | 0.151102822 |
| Naive output fusion | 0.088261879 | 0.051992991 | 0.070127435 |
| Gate minus canonical-first | -0.000077187 | -0.000066820 | **-0.000072003** |
| Positive gains | 0/6 | 0/6 | **0/12** |

**36 ROI, 12 patients, one public source, one pretrained checkpoint.** Not 36 independent clinical cases. Negative sign is tiny and **not evidence of clinically meaningful harm**; it DOES show the current method has not improved accuracy.

## Paper-ready novelty fork

1. **Correctness-first / geometry software method**: typed coordinates, explicit source/target physical affine composition, proof that patch-wise inference plus metadata-preserving resampling commutes under precise assumptions; failures caused by inserting the change after an irreversible mix. Unlike simple header correction, prove distinct semantic worlds produce observational aliasing, and create open reproducible mistakes.
2. **Feature-level algorithm**: modify ACTUAL pretrained network upstream of feature mixing with a differentiable physical-frame operator. Compare canonical-first and correct spatial harmonization with *the exact same weights and labels*, then multiple architecture families and patient-domain OOD.
3. **Negative-transfer paper**: if feature repair still fails, clearly scope as physical-coordinate correctness versus learned representation fragility. Existing geometric adaptation papers must be cited; zero improvement alone is not a novel algorithm or enough for a high-impact paper.

## Layout borrowed in scholarly *structure*, never copied

- Abstract: numerical falsifiable benefit/correctness question, mechanism, exact test family, and limitations. **Never say clinical deployment.**
- Figure 1: patient physical-space coordinate diagram with world/voxel/array frame, exact affine examples.
- Figure 2: competing pipelines (*canonical first*, *naive fuse*, *gate*, *actual pre-mix correction*) including where information is irreversibly lost.
- Figure 3: paired patient Dice/HD95 plots **with 12 individual points** and separate cohort labels; do not bar-plot 36 correlated ROIs as IID.
- Figure 4: genuinely full-volume independent multi-domain evidence, only when performed.
- Table 1: true family-level competing methods; Table 2: ablations on trust/geometry; Table 3: failure conditions and statistical uncertainty.
- Main text should explain *what moved in physical space and why the trained feature responded*, not just reproduce code snippets.

### Venue constraints
TMI official https://ieeetmi.org/authors-instructions/ initial regular <=10 pages **including references**, <=250-word one-paragraph abstract. Because the current fusion advantage is negative, a TMI efficacy claim is unjustified regardless of fitting the page limit.
