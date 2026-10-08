## Copy-paste native regression patch (2026-10-08)

**Maintainer-native submission now ready:** [`brainmask_export_with_native_tests.patch`](brainmask_export_with_native_tests.patch) applies directly to the exact original `predict.py` and adds `tests/test_predict_export_space.py` under the original project's own tree. It requires only NumPy, NiBabel and pytest for the focused regression, no ClinGeom or TensorFlow.

[Public original-RED / patched-GREEN matrix](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37749615927): Python **3.10 and 3.12** each produced **8 failing nonaxial tests on original unmodified upstream** and **17 passing tests after applying the two-file patch**. This validates that the regression catches a real source defect, not simply a test that would pass regardless of the implementation.

Maintainer quick replay:

```bash
git checkout ca789fb26b70e359b61a8d52968813dd772fb4b1
git apply --check /path/to/brainmask_export_with_native_tests.patch
git apply /path/to/brainmask_export_with_native_tests.patch
python -m pip install numpy nibabel pytest
python -m pytest -q tests/test_predict_export_space.py
```

This test file compiles the real `predictions_2_nii` method from its AST and stubs only model-patch reconstruction, keeping the unit test lightweight. The separate pinned public full-TF and original-checkpoint experiments remain available for independent scrutiny. **No third-party review or upstream merge is claimed.**

# Maintainer-ready issue — brain_mask nonaxial export attaches a stale source affine

**To submit:** copy the section from **Issue title** through **Maintainer question** into a new issue on https://github.com/ecalabr/brain_mask/issues/new using an account authorized to post. Current GitHub integration returned HTTP 403; **this is NOT already submitted to upstream**.

## Issue title

`Coronal/sagittal prediction export retains source NIfTI affine without undoing axis permutation`

## Problem

In `utilities/input_fn_util.py::_load_single_study()`, `data_plane='cor'` transposes the spatial tensor with `(0,2,1,3)` and `'sag'` with `(1,2,0,3)`. `reconstruct_infer_patches_3d()` reconstructs in that model-space axis order, but `predict.py::predictions_2_nii()` serializes the output with the **unchanged original source affine**. For non-axial planes, this puts voxel data under the wrong physical index-to-world mapping; it may also change the output NIfTI array shape.

The original `trained_models/t1/t1.yml` specifies `data_plane: ax`; default-axis tests do not cover the non-axial branches already accepted by the reader.

## Pin the exact upstream source

- Repo: https://github.com/ecalabr/brain_mask
- Upstream SHA: `ca789fb26b70e359b61a8d52968813dd772fb4b1`
- Original exporter Git blob: `d4b03f5b851928ead5ef014af75cd1dd5fbd64dc`
- Original input/reconstruction Git blob: `1c2ce1a9f13f690cdaf28eac5b003e44ec06d192`
- Real T1 MRI Git blob: `81b1e5260ee4a579d8ac490cac113df9c1cb942f`
- Existing example brain-mask Git blob: `e0dd089b3544da4352e751a24dc27a828904bad7`

## Reproduction 1 — native loader/exporter on the original real MRI

Author-run exact-head [passing CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715729072) · [real MRI overlays and JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715729072/artifacts/11522779447) · [source-pinned runnable reproducer](brainmask_plane_export_repro.py).

| Plane | Original fixed-patient-grid Dice | After native patch | Original centroid deviation |
|:--|--:|--:|--:|
| Axial | 1.000 | 1.000 | 0 mm |
| Coronal | 0.781130 | 1.000 | 5.661 mm |
| Sagittal | 0.673054 | 1.000 | 12.379 mm |

The existing brain mask is a **transport sentinel** and not an expert annotation or model prediction. This first test intentionally stubs the patch-reconstruction seam to isolate the original loader/exporter contract.

## Reproduction 2 — original native TensorFlow reconstruction, no stub

Author-run [passing TensorFlow 2.13 CPU CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37716306589) · [native-TF reproduction code](brainmask_full_tf_reconstruction.py).

On a **predeclared non-cubic central 48×56×40 crop of the same real MRI**, the original `_tf_patch_3d` and original `reconstruct_infer_patches_3d` both execute and restore the mask sentinel exactly in the working plane. Then the original sagittal exporter writes `56×40×48` while claiming the unmodified source affine. The same code with the surgical patch exports `48×56×40` with exact voxel equality (Dice 1.0), and axial behavior remains exact.

Therefore the misregistration is not a consequence of substituting a reconstruction routine in Reproduction 1.

## Reproduction 3 — genuine pretrained model, **equal-shape silent failure**

The strongest test now executes the **original public pretrained T1 3D U-Net**,
with its original TensorFlow patch extractor/reconstructor and an actual
acquired MRI center crop. It strictly authenticates upstream `Networks` model
source, original HDF5 checkpoint and the acquired MRI by Git hashes; **340 model
parameter variables change on weight load**.

[Green final-head workflow](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979)
· [public original-vs-repaired **real learned-model MRI image / JSON**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979/artifacts/11524838113)
· [merged reproducible source PR #46](https://github.com/lindicaphxag-tech/kaggle/pull/46)
· [executable code](brainmask_pretrained_native_export.py).

Here the 64³ crop means **all original and patched NIfTI shapes are identical**,
making the original error *silent to shape-only checking*. On the same genuinely
learned probability tensor for each model plane, the original exporter attaches
the original acquisition affine before reversing the plane permutation.

| Plane | Original export Dice vs **its own** correctly inverse-oriented prediction | After source patch |
|:--|--:|--:|
| `ax` (negative control) | **1.0000000000** | **1.0000000000** |
| `cor` | **0.0041171965** | **1.0000000000** |
| `sag` | **0.3630825519** | **1.0000000000** |

**Important interpretation:** each row compares an original vs patched
**export of the same exact learned model result**, not the different predictions
produced by feeding different anatomical orientations to the network.
Therefore this detects **serialization frame loss** without claiming the
network should be rotation-equivariant. The source's brain mask is not used
for this trained-model output test; it is neither ground-truth clinical
validation nor a trained model accuracy uplift.

This equal-shape example is especially important: merely checking
`predictions.shape == source.shape` does **not** certify equal physical
coordinate frames. Only transform provenance or independently checked
world-coordinate semantics can establish that.

## Suggested minimal patch

Apply [one-file standard unified diff](brainmask_export_plane.patch) against the pinned upstream `predict.py`:

```python
if params.data_plane == 'cor':
    predictions = np.transpose(predictions, (0, 2, 1))
elif params.data_plane == 'sag':
    predictions = np.transpose(predictions, (2, 0, 1))
```

The complete patch executes inverse permutation **before source-mask multiplication and serialization**. It validates that output shape matches the source and conservatively refuses to attach the old affine after `resample_spacing`/`load_shape` until an inverse crop/resample mapping exists. The original axial behavior is unchanged. `git apply --check`, `python -m py_compile` and the same real-source before/after tests passed.

## Maintainer question

Is the intended output contract that predictions exported to NIfTI should occupy the **original input image's patient-space grid**, regardless of `data_plane` selected for network inference? If yes, could this inverse-plane regression be retained upstream? For resize/crop options, would fail-closed export or a fully implemented inverse warp be preferable?

**Claim boundary:** The reproducer uses genuine acquired anonymized MRI and the source's existing example mask, but no clinical ground truth is established. The full-TensorFlow path uses the real patch extractor/reconstructor but not a learned network forward in this isolated exporter audit. This is author-run verification, not maintainer approval or a prospective first-seen event. An AI assistant participated in the source analysis and patch preparation.
