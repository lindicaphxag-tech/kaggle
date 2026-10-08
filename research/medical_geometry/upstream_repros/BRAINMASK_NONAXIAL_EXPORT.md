# High-value real upstream issue: nonaxial NIfTI export reuses source affine

**Target:** [ecalabr/brain_mask](https://github.com/ecalabr/brain_mask), pinned commit `ca789fb26b70e359b61a8d52968813dd772fb4b1`.

## Real source-coupled code path

`utilities/input_fn_util.py::_load_single_study()` explicitly permutes model inference inputs:

- `data_plane='ax'`: identity;
- `data_plane='cor'`: axes `(0,2,1,3)`;
- `data_plane='sag'`: axes `(1,2,0,3)`.

`utilities/input_fn_util.py::reconstruct_infer_patches_3d()` reconstructs a tensor in the loader's working order; there is no inverse plane transform.

`predict.py::predictions_2_nii()` then calls that reconstruction, optionally multiplies a source mask and serializes the array using **the original MRI's unchanged affine**. On non-axial input the spatial array dimensions/positions no longer correspond to that affine. This is a natural supported-code-path geometry bug, NOT injected into the source. The original published `t1.yml` uses axial mode, so a passing default pipeline does not eliminate the non-axial risk.

## Exact-source baseline / surgical fix

The reproducibility runner downloads genuine anonymized T1 MRI and existing brain mask from the same upstream version and pins both source code Git blobs and data Git blobs.

To **isolate the export boundary**, the model's patch reconstruction is stubbed to identity, and the existing source brain-mask array is passed as a deterministic segmentation transport sentinel. The actual *original* upstream `_load_single_study()` and `predictions_2_nii()` functions are compiled unchanged from their AST and executed. Therefore this is a direct **production loader/exporter contract test**, not a claim of actual learned-model segmentation output.

The two-line inverse conventions are:

```python
if params.data_plane == 'cor':
    predictions = np.transpose(predictions, (0, 2, 1))
elif params.data_plane == 'sag':
    predictions = np.transpose(predictions, (2, 0, 1))
```

The [standard upstream-ready patch](brainmask_export_plane.patch) additionally rejects attempts to attach the unmodified source affine after untracked `resample_spacing` or `load_shape` changes, rejects unknown plane codes, and checks the resulting array shape before saving. This is a **narrow safe patch**, not a claim that resampling cannot eventually be properly supported.

## Controls

- Original native axial loader/export path must already have exact reference-grid Dice 1; patched axial path must remain identical.
- Both original cor/sag exported arrays must fail same-patient-reference-grid equality on the real source mask.
- Both patched cor/sag arrays must recover the exact original source mask under the unchanged source affine.
- Supported non-axial correction must occur BEFORE optional multiplication by a source mask.
- Unknown or nontracked spatial transformations are refused rather than silently attaching stale metadata.

## Honest scope and release

No neural network checkpoint is executed **in this isolated audit**. The independently verified trained T1 3-D U-Net experiment is in public [PR #39](https://github.com/lindicaphxag-tech/kaggle/pull/39), and is distinct.

The existing mask from the anonymized source is not verified expert clinical ground truth. This audit uses it solely to detect identity failures in data transport. The real MRI is a real acquired example, not generated anatomy.

Numerical failure evidence and a five-panel real-MRI overlay are emitted only after both original and patched paths have been tested in publicly visible CI. No claim of an upstream PR/merge or independent maintainer adoption is made until it actually occurs.

## Maintainer submission (not yet authorized)

Open [ecalabr/brain_mask](https://github.com/ecalabr/brain_mask) issue/PR under an authenticated account with access, link this exact-baseline report and apply [the two-site safety patch](brainmask_export_plane.patch). The project's owner controls the expected semantics of nonaxial export and whether resampling should fail closed or be inverted. **Do not spam or represent the author's own public CI as upstream endorsement.**
