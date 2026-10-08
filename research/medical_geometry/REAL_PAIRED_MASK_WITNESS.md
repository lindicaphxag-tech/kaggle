# Real MRI + pre-existing brain-mask geometry and surface metric witness

This experiment uses the real example MRI and its paired *existing* brain-mask
file from [ecalabr/brain_mask](https://github.com/ecalabr/brain_mask), pinned
at commit `ca789fb26b70e359b61a8d52968813dd772fb4b1`.

Inputs:

- `example_data/example001/example001_t1.nii.gz`: 11,972,670 bytes,
  Git blob `81b1e5260ee4a579d8ac490cac113df9c1cb942f`.
- `example_data/example001/example001_brain_mask.nii.gz`: 310,064 bytes,
  Git blob `e0dd089b3544da4352e751a24dc27a828904bad7`.

The source repository's README says its **example data was already masked for
anonymity**, so output mask quality may not be correct. No expert/manual
ground-truth or lesion diagnosis is asserted here. This is a *real-input,
paired-provided-mask representation-correctness* experiment.

## Deterministic protocol

1. Fail closed unless both input files match their pinned Git content IDs,
   both are 3-D on the same nondegenerate declared patient/world affine, and
   the source mask is neither empty nor full.
2. Use the real supplied example brain mask, not a synthetic ROI.
3. Flip the stored mask along the **predeclared axis 0**, leaving the affine
   unchanged (controlled metadata corruption); do not pick the largest-error axis
   after seeing results.
4. Independently compare both the corrupted and analytically repaired
   representation to the *unchanged source MRI patient-space grid*.
5. Report fixed-grid Dice, three-dimensional physical centroid error (mm),
   symmetric HD95, symmetric mean surface error and Hausdorff (mm).
   Surface distances are computed in the real affine's world coordinate frame,
   including anisotropy and oblique basis, using KDTree on boundary voxel centers.
   They are **not** clinical mesh-based HD95 estimates.
6. Require exact restoration of the original mask by `A_out = A_in H`.
7. Confirm affine and data permutation separately using native NiBabel
   `as_reoriented()` as an independent implementation.
8. Render the *actual MRI + actual existing mask*, with before/after label
   contours on the same underlying anatomical pixels, and save a JSON sidecar.

No model checkpoint runs in this protocol. **No segmentation accuracy uplift,
clinical safety, external adoption or expert validation is implied.**

## Exact reproduction

```bash
python -m pip install "numpy>=1.26,<3" "scipy>=1.11" "nibabel>=5.2,<6" "matplotlib>=3.8"
curl -fL "https://raw.githubusercontent.com/ecalabr/brain_mask/ca789fb26b70e359b61a8d52968813dd772fb4b1/example_data/example001/example001_t1.nii.gz" -o example001_t1.nii.gz
curl -fL "https://raw.githubusercontent.com/ecalabr/brain_mask/ca789fb26b70e359b61a8d52968813dd772fb4b1/example_data/example001/example001_brain_mask.nii.gz" -o example001_brain_mask.nii.gz
test "$(git hash-object example001_t1.nii.gz)" = 81b1e5260ee4a579d8ac490cac113df9c1cb942f
test "$(git hash-object example001_brain_mask.nii.gz)" = e0dd089b3544da4352e751a24dc27a828904bad7
python -m research.medical_geometry.real_paired_mask_witness --image example001_t1.nii.gz --mask example001_brain_mask.nii.gz --out artifacts/paired
```

The [public GitHub Actions workflow](../../.github/workflows/clingeom-real-paired-mask.yml)
performs the same steps and uploads its raw PNG and JSON upon success.

## Licensing/provenance

The upstream README identifies **CC BY** for the source project. Verify the
specific data rights and preserve attribution before republishing the MRI itself.
This workflow downloads inputs on demand and archives only derived PNG/JSON.
The public research repository currently has no general code license.
