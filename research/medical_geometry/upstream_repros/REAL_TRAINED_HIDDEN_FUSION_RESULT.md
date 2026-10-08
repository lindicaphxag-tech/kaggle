# Original pretrained U-Net: actual internal-feature transport before a frozen learned classifier

**Evidence generated 2026-10-09. Author-operated, exact trained TensorFlow execution; no outside endorsement.**

## Immutable first-party execution

- [GitHub Actions 37821718961](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37821718961) at PR #70 code head `dcf076c5662144dce75cb36d9b81de313595d8fa`: **success**.
- Original TensorFlow CPU job `113464142761`: **success**. All source/official weight checks, unchanged original 3-plane inference, original-vs-native-patched NIfTI export, weak-reference measurement, and new hidden-feature experiment passed.
- New orientation/pointwise contract tests: **9 passed**.
- Four-file run artifact [11570235231](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37821718961/artifacts/11570235231), including `authentic_hidden_feature_fusion.json`. Uploaded ZIP digest SHA-256:
  `0f94a718f86fb6af6543709430a1692e30f72f1658c8e24d20f2a53b9dbe9d72` (the hosted attachment has finite retention).
- Frozen external source `ecalabr/brain_mask` commit `ca789fb26b70e359b61a8d52968813dd772fb4b1`.
- Original MRI Git blob `81b1e5260ee4a579d8ac490cac113df9c1cb942f`;
  original **nonexpert** mask blob `e0dd089b3544da4352e751a24dc27a828904bad7`;
  published T1 pretrained HDF5 blob `6ac54600739585f54d9f612d13264b74965ffd33`.
- Reproducible code: [brainmask_true_hidden_fusion.py](brainmask_true_hidden_fusion.py)
  and [adversarial unit tests](test_brainmask_true_hidden_fusion.py).

## Fixed experimental intervention

The architecture is the true upstream `Unet3dBneck` 3D U-Net. Its
public pretrained parameters are loaded without partial matching or training.
Separate axial/coronal/sagittal runs process the **same acquired 64³ T1 MRI
crop** through the unchanged model. The last hidden feature tensor *before*
the original pretrained `Conv3D(1x1x1)` + sigmoid has **32 channels**. For
each plane, the experiment verifies all three independent equalities to
a numerical tolerance of at most `2e-5`:

1. `frozen_classifier(true_hidden_features)` reproduces the unmodified network's
   real forward probability output;
2. the new frozen network forward reproduces the prior source-native
   trained TensorFlow export;
3. `frozen_classifier(inverse_plane(true_hidden_features))` reproduces the
   prior **patched** NIfTI output on the acquisition voxel grid.

Failure in any oracle refuses the experiment. The same **untouched original
1x1 classifier** is then used for both hidden-feature fusion arms: arithmetic
mean before classification, either uncorrected (wrong-index) or correctly
reoriented. As a fair standard baseline, the third arm does **native inverse
orientation, classify separately, then average probabilities**. All three
use equal plane weights and the predeclared 0.5 threshold.

This is an experimental cross-view composition. It was **not** the original
pretrained network's fusion pathway and was never re-trained to produce
cross-plane-invariant features.

## Observed first-party real learned-feature data

All comparisons are against the same original **upstream-supplied nonexpert
brain mask**, not against a verified expert label.

| Single-study arm | Weak-mask Dice | Physical HD95 (mm) | Physical centroid displacement (mm) | Foreground voxels |
|---|---:|---:|---:|---:|
| Raw-index hidden-feature fusion, spatially wrong | 0.0502044591 | 52.9622507 | 33.9066378 | 6,947 |
| **Exact orientation transport BEFORE frozen learned classifier** | **0.2109570529** | 49.5378643 | **27.6771251** | 28,529 |
| Standard corrected-grid POST-prediction fusion | 0.1700203499 | **40.9023227** | 28.6229273 | 22,591 |

The only difference between the first two arms is known spatial alignment
of actual learned **hidden features**. The raw-index arm is an intentionally
invalid baseline (not a state-of-the-art competitor). This is a real
downstream change in a trained classifier, **not** another label-only transpose.

**Important negative result:** correct pre-head hidden-feature fusion produces
a larger weak-mask Dice than standard post-prediction fusion on this MRI, but
it produces a **worse HD95** by ~8.64 mm. We must not present Dice alone
or claim universal task-level improvement. The confidence/foreground-volume
change may reflect out-of-distribution latent averaging.

The learned 32-channel features from different model-plane inputs need
not have the same representation meaning; correct spatial orientation is
necessary for patient-space mixing but **insufficient to certify learned
channel-semantic equivalence**. This experiment measures hidden-feature
disagreement after orientation in the attached JSON, but it does not
establish a learned channel gauge, cross-patient reliability, or safe
pre-head fusion authorization.

## Controls and scientific boundaries

- Authentic original HDF5 weight load and weight SHA match against previous
  native learned-model run; external source MRI and config are pinned.
- Unmodified original real 3-D TensorFlow preprocessing/forward and native
  source NIfTI exporter are independently run; no synthetic model outputs.
- Original HDF5 classifier weights, source MRI, centered crop, threshold,
  plane set and arithmetic mean are frozen. Not even the classifier bias is
  fine-tuned.
- 9 adversarial unit controls verify axis permutations keep channel identity,
  commute with pointwise classifiers, reject NaNs and unknown planes, and
  **disprove the assumption** that identical tensor shape implies latent
  semantic equivalence.
- A head-extraction numerical oracle is checked for *each* of three real
  model orientations, not just one hand-picked successful branch.
- No expert test masks, multi-patient validation, cross-modal generalization,
  native learned MONAI DiNTS multi-phase branch, repaired trained encoder,
  independent investigator reproduction or external maintainer acceptance.

## Best next experiment (the actual publication gate)

Move to a second independent **trained architecture with naturally
heterogeneous channel sampling supports** (e.g., a publicly pretrained DiNTS
model) and independent expert-annotated held-out CT. Compare a native
geometry-aware baseline, controlled spatially incorrect fusion, chart-only
repair, channel-support-aware authorized pre-mix repair, and the conservative
post-prediction/refusal fallback, with patient-level paired uncertainty.

Stop claiming scientific novelty if ordinary MONAI/NiBabel orientation
tracking and post-prediction fusion are just as correct and effective. No
additional analytic certificate substitutes for learned, labeled
multi-patient performance.
