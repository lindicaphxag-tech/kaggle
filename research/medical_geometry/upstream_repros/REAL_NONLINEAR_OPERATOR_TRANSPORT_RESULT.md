# Authenticated nonlinear spatial-operator transport in an actual pretrained 3-D U-Net

**Author-operated primary evidence; 2026-10-09. This is not independent external replication nor a demonstrated generalization gain.**

## Primary immutable executable evidence

- [Authenticated trained-source TensorFlow CI run 37823599961](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37823599961) on commit `520d2071d0b345dac47351f93eb6a2518edfe08f`: **SUCCESS**.
- True pretrained CPU TensorFlow job `113470602816`: source-trained checkpoint hash, reconstructed original source network, original NIfTI exporters, original/native-patched 3-plane learned predictions, original 32-channel pre-head cut, logit-equivalence oracle, **new original residual nonlinear cut**, and the independently computed operator conjugation all passed.
- `4 + 6 + 16` dedicated model/space tests passed. The additional chart/operator tests are explicitly *controlled unit tests*, not trained-model predictions; the separate true TensorFlow job makes the latter evidence.
- Actual workflow's evidence ZIP SHA-256: `086c34fb985172bb56766f02e564d6e523cc6421e4d8383fcf2377507b1d889b`; JSON `authentic_hidden_feature_fusion.json` in the Actions artifact. Artifact retention is time-limited.
- External model and acquisition MRI pinned and verified exactly as in [PR #70](https://github.com/lindicaphxag-tech/kaggle/pull/70), a previously passed authentic original-source TensorFlow experiment.
- Exact executing source: [brainmask_true_hidden_fusion.py](brainmask_true_hidden_fusion.py); tests [test_brainmask_true_hidden_fusion.py](test_brainmask_true_hidden_fusion.py).

## Actual learned nonlinear intervention

The pretrained original Keras `Unet3dBneck` **final residual block** has three BatchNorm operations. The cut is the INPUT tensor of the first of those three normalizers, thereby retaining the original complete final residual block's actual skip branch, original nonlinear activations, trained convolutions including spatial 3×3×3, the same trained pointwise classifier and sigmoid **inside** the tail. This is not a synthetic operator invented for the results.

A separate Keras partial model runs the unchanged pretrained encoder up to that cut. The original trained tail is then reattached and its full *per-orientation* output must numerically reproduce the unchanged model's independently obtained original output within the frozen `2e-5` tolerance. All three replays passed.

At the **pre-final-residual-block hidden tensor**, perform a frozen equal mixture of exactly the same source MRI's three native-plane views. The three experimental arms are:

1. Known-invalid raw co-indexed mixture and original tail (coordinate-error positive control).
2. Real hidden tensors spatially mapped to acquisition grid, then original *axial* spatial-kernel operator is applied without transformation (incorrectly assumes the learned spatial operator itself is direction-agnostic).
3. The same spatially mapped hidden tensor mixture, but apply **conjugated original trained nonlinear tails**: map the common feature tensor to each native chart; evaluate the unchanged original spatially learned kernels and nonlinearities in that native chart; map its actual probability output back to the acquisition chart; then equally average resulting probabilities.

Strong established baseline: original independent 3-plane predictions, **native** spatial reorientation and equal probability averaging. Prior frozen-logit ensemble is also retained. All original weights remain frozen, the same threshold 0.5 is used, and no held-out mask values affect choice of arms.

## Real one-study results

**Upstream-provided nonexpert brain mask only**, centered 64³ acquired-grid crop. This is not expert-labelled NFBS, whole-volume validation or a patient-disjoint dataset.

| Arm | Weak-mask Dice | HD95 mm | Centroid distance mm | Foreground voxels |
|---|---:|---:|---:|---:|
| Invalid raw-index hidden mix → trained nonlinear tail | 0.0427342315 | 55.9732079 | 34.8652805 | 5,957 |
| Aligned hidden mix → original untransported trained axial tail | **0.1981383889** | 50.0899191 | 28.0991250 | 26,620 |
| Aligned hidden mix → spatially conjugated trained nonlinear tails | 0.1980722067 | 50.3040746 | 28.1290661 | 26,617 |
| Ordinary corrected-grid **post-prediction** mean | 0.1700203499 | **40.9023227** | 28.6229273 | 22,591 |
| Aligned pre-head 1×1 feature mean (=logit mean to 4.17e-7) | **0.2109570529** | 49.5378643 | 27.6771251 | 28,529 |

**Learned original nonlinear operator noncommutation**, holding input canonical learned hidden features fixed (mean absolute probability differences between common-axial trained tail and properly conjugated trained native tails):

- Coronal: **0.003534034146456041**
- Sagittal: **0.0038702730142047576**

These are genuine trained nonlinear operator outputs and nonzero orientation-transport sensitivity. Nevertheless, conjugation on THIS MRI does not improve either Dice or HD95 over the naive aligned-axial-tail arm. Its stronger claim is **not task improvement**, but demonstration and quantified audit of why geometric feature consistency does not imply learned-operator rotation equivariance.

## Interpretation / refutation / novelty gate

- PR #70 proved that its last-layer feature mix can be reduced exactly to a standard logit average. This new cut instead retains the real learned nonlinear spatial residual block. Unlike a pointwise linear head, a spatial kernel generally cannot commute with a spatial coordinate permutation; 16 unit controls and empirical operator commutator distinguish those two questions.
- **The source model was not pretrained to fuse across plane views at this cut.** Channel indices in the three independent input rotations have no proven representation-gauge equivalence. Neither correct chart alignment nor conjugating pretrained spatial kernels establishes safe latent mixing.
- No clinician-grade or independent reference labels, no multi-subject validation, no retraining of a clinically effective algorithm, no statistical uncertainty, no superiority over standard post-prediction fusion, no external review or third-party maintainer adoption.
- The published counterexample is worthwhile as a **reproducible mechanistic artifact**, not as a stand-alone high-impact algorithmic paper.
- Next real paper acceptance gate: patient-disjoint **expert-annotated independent data** such as NFBS (125 manually corrected T1 brains; [project](https://preprocessed-connectomes-project.org/NFB_skullstripped/), [peer-reviewed dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC5080782/)), true whole-brain protocols and acquisition grids, independent model/architecture, matched compute/thresholds, and predeclared primary Dice/physical HD95 endpoints. A genuine mechanism must improve a strong orientation-aware MONAI baseline or explicitly abstain; no feature cherry-picking.

**Bottom line:** Real frozen model + real MRI + actual nonlinear learned block + spatial operator conjugation + observed noncommutativity + explicit negative task outcome. All necessary, still insufficient for a top-tier original algorithm claim.
