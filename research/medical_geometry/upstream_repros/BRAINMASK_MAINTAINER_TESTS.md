# Native upstream tests for brain_mask spatial export

This is the **minimal adoption packet** for `ecalabr/brain_mask` on original
source commit `ca789fb26b70e359b61a8d52968813dd772fb4b1`.

The evidence was first found through native source audit and then checked with
real MRI and actual pretrained T1 U-Net inference:

- [real MRI, original vs fixed source loader/exporter](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715729072);
- [real TensorFlow patch extraction/reconstruction](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37716306589);
- [real original HDF5 pretrained model, nonaxial original vs corrected export](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979).

**However, project maintainers should not need any of that framework to accept
a regression.** This packet provides a standard unified diff that changes
`predict.py` plus a standalone `tests/test_predict_export_space.py`.

## Inputs and native test cases

Run this directly on the original source, after applying the patch:

```bash
git apply --check /path/to/brainmask_export_with_native_tests.patch
git apply /path/to/brainmask_export_with_native_tests.patch
python -m pip install pytest numpy nibabel
python -m pytest -q tests/test_predict_export_space.py
```

The tests use only NumPy, NiBabel, pytest and Python stdlib.
They call the **actual upstream exporter method from its source AST**; the
TensorFlow model-patch reconstruction is intentionally stubbed *only in this
fast unit test*, to keep it isolated and suitable for project maintainers.
The independent full-real-TF and real-trained-model runs above close that gap.

Cases cover:
- axial negative control;
- coronal/sagittal inverse permutations on noncubic asymmetric arrays;
- oblique/sheared and non-sheared affine matrices;
- alignment of source-space optional mask after inversion;
- explicit refusal of resample/crop configurations with no recorded inverse;
- unrecognized plane and source-shape mismatch refusal.

The GitHub workflow `clingeom-brainmask-native-regression.yml` additionally
proves the **pre-patch tests fail against the pinned original code** and
then proves the **patch + tests pass**, across Python 3.10 and 3.12.

## Scope and publication conditions

The fix corrects patient-space index/affine binding. It does **not** change
the learned segmentation probabilities, guarantee clinical correctness, or
infer NIfTI affine identity from same-shaped tensors. Proper resampling/crop
inversion needs a separate, explicit implementation; fail-closed is a design
proposal for maintainer consideration.

The public `lindicaphxag-tech/kaggle` repository is author-operated.
**No project owner has reviewed or adopted this yet.** The GitHub integration
cannot open a third-party issue here (403); preserve that distinction.
The research repository's reuse license decision is also still open.

## Data provenance

The fast tests use non-clinical numeric arrays only; they do not redistribute
medical data. Real acquired MRI, the original example mask, pretrained weights
and full-source execution are available via pinned public workflows above.
This keeps the upstream patch small and avoids adding copyrighted imaging
data to the patch.
