# ClinGeom prospective six-patient mechanism and reliability-gate falsification

**Status: Completed original pre-trained-model, source-revision-frozen, truly new patient-cohort *ROI pilot*. All authentic public TensorFlow steps passed. Negative efficacy: the proposed gate does NOT beat the original unpermuted RAS branch.**

## Independent provenance / preregistration boundary

- Prospective branch [PR #73](https://github.com/lindicaphxag-tech/kaggle/pull/73), after the [three-subject discovery dataset PR #72 (merged)](https://github.com/lindicaphxag-tech/kaggle/pull/72) had already demonstrated a three-view fusion failure.
- [**Official-original-checkpoint model CI #37827612217: SUCCESS**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827612217). Original Keras native 3-D U-Net HDF5, no retraining; all six reserved subjects x three fixed source-image-only 64³ ROIs x seven fixed alternatives completed. All original-model nonlinear tail/head replay equalities are verified on each plane.
- Original approved source image SHA list is independently frozen in [PROSPECTIVE_SIX_FROZEN_SEEDSET.md](PROSPECTIVE_SIX_FROZEN_SEEDSET.md); the initial [dataset-only acquisition + model-count assertion failure CI #37827321553](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827321553) proves dataset access/affine checks *preceded* any six-subject model outcome. The hardcoded `!=3` check was fixed without reading outcome scores.
- Source dataset [MedOtter/SynthStrip](https://huggingface.co/datasets/MedOtter/SynthStrip) exact HF SHA `d5a92c65c549b2a099bdb5a6e61ebeed6cb6eca3`. Cases are the next **six distinct metadata-reported T1 subject identifiers** after the two SynthStrip discovery subjects. Actual image/mask files are never published or attached. The dataset describes its masks as manually refined; we did not independently audit rater quality or cross-study subject identity overlap.
- [Every patient/ROI source-replayed output and score JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827612217/artifacts/11571812747) (ZIP SHA-256 `47398d898cdd87f8302aa2e64a322476086878d9a9bcef22bbf0b5676ec61835`); [six-case file-hash provenance artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37827612217/artifacts/11571413713). Uploaded artifact bytes are available only for the provider's limited retention.
- **No** post-result optimization: `view_disagreement_scale=0.10`, `trained_spatial_operator_scale=0.015`, maximum blend weight `0.5` committed before unseen cases' model results.

## Mechanism — the new non-supervised gate

From three independently oriented original trained outputs mapped to the canonical RAS grid `p_ax, p_cor, p_sag`, and the original trained nonlinear residual spatial operator applied with and without exact chart conjugation:

```text
d_view(x) = max(|p_cor(x)-p_ax(x)|, |p_sag(x)-p_ax(x)|)
d_kernel(x) = max(|T_cor(f)(x)-T_ax(f)(x)|, |T_sag(f)(x)-T_ax(f)(x)|)
r(x) = exp[-min(50, d_view(x)/0.10 + d_kernel(x)/0.015)]
p_gated(x) = p_ax(x) + 0.5*r(x)*(mean(p_ax,p_cor,p_sag)(x)-p_ax(x))
```

These are actual frozen learned-model outputs; no labels or additional trainable coefficients enter the gate. **The method is not a conformal safety certificate**; a decrease in label overlap is possible and must count as a failure.

## Prospective primary results (patient-macro average of each patient's 3 ROIs)

| Method | Source-manual-mask Dice ↑ | Physical HD95 mm ↓ |
|---|---:|---:|
| **Canonical-RAS unpermuted first-view model (strongest baseline)** | **0.186392** | **41.5667** |
| **Label-free operator/view-risk gated probability fusion (candidate method)** | 0.186315 | 41.5681 |
| Correct patient-grid equal-probability ensemble | 0.088262 | 49.5186 |
| Correct patient-grid equal-logit ensemble | 0.088829 | 52.8774 |
| Exact aligned latent features → original trained nonlinear block | 0.082872 | 52.3850 |
| Aligned latent and physically conjugated trained residual operators | 0.083284 | 52.1286 |
| Invalid same-array-index latent mean → original residual block | 0.036461 | 69.0992 |

**Prospective failure:** gated minus first-view patient-macro Dice = **−0.0000772**; gated minus first-view HD95 = **+0.00137 mm**. The gate did **not** win. It did preserve nearly all of the strong first-view performance when ordinary fusion cut mean Dice by ~53%.

### Is the near-tie due to the gate never activating?

No: the true per-case output artifact records an **average reliability factor r of 0.7435 across all 18 fixed ROIs**, with **79.04%** of voxels satisfying `r>0.2`. The maximum blend weight is 0.5; therefore reliability was not simply zero everywhere. The gated method differed from first-view in **all 18 ROI Dice scores** (usually by tiny margins). These statistics are computed from the committed original output JSON; reliability represents a per-voxel *blend multiplier*, not a segmentation-change or conditional acceptance rate. We cannot claim meaningful risk reduction from large r alone.

All six patient-level Dice deltas (gate minus original first view) are negative: −0.0001833, −0.0000630, −0.0000485, −0.0000237, −0.0001002, −0.0000446. There is **no** one case with demonstrated positive gain. HD95 differences are 0 in five patients at recorded precision, and +0.00823 mm in one.

## Limitations / critical reviewer judgment

1. Six independent source IDs, 18 overlapping diagnostic ROIs, **not whole-brain independent clinical scans**. A shared original public pretrained source model may be OOD for this multi-source cohort; preprocessing/sequence mismatches are not fully studied.
2. The reference label is manually refined according to dataset's own description, not independently adjudicated by this project.
3. The baseline is the first *unpermuted canonical-RAS* orientation, NOT guaranteed to be the actual network's original scanner anatomical plane: this distinction is essential when interpreting network anisotropy.
4. The early 3-subject *discovery* HD95 analysis had a **posthoc missing-surface correction**, transparently recorded in [the discovery report](REAL_EXTERNAL_ANNOTATED_HOLDOUT_RESULT.md). The new six-case cohort uses that policy fixed before data access.
5. No strong reference trained multidomain multi-orientation architecture yet; comparisons cannot establish improved mainstream medical AI beyond already existing orientation-safe sampling or model adaptation.
6. Gated fusion matched the strongest method only approximately, with **slightly worse Dice on all 6 held-out subjects**. No claim of clinical benefit, SOTA method, confidence-calibrated guarantee, external adoption, research-independent reproduction, or publication acceptance is made.

## Scientific decision

The robust finding across the discovery cohort and **the separately locked six-person extension** is a negative-transfer boundary: *geometric chart consistency does not authorize mixing domain-shifted pretrained latent representations or outputs*. Physical correctness solves one class of bug; it does not imply downstream task utility.

**Do not scale the current hand-weighted heuristic as a claimed winning paper method.** Next potential high-level work should be channel-gauge calibration trained with source-only/unlabelled data, or a genuinely trained multi-view network with selective risk/uncertainty grounded on a separate labelled training cohort. That mechanism must then be frozen and tested on new, **unused** patients, with the unpermuted original model and orientation-safe native output as strong controls. A full-volume annotated external benchmark and independent reproduction remain essential.
