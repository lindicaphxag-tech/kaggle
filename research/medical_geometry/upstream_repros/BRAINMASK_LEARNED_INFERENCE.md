# Final full learned-model causal-chain test

This companion experiment to the [validated native nonaxial exporter defect](BRAINMASK_UPSTREAM_ISSUE_READY.md) executes the *same original*, publicly released, pretrained TensorFlow 2.13 T1 3-D U-Net checkpoint under strict original model architecture and native source.

It runs the original native source NIfTI loader (`ax/cor/sag`), nonzero z-score normalization, TensorFlow patch extraction, the actual learned forward pass and original TensorFlow patch reconstruction. From the **same exact learned probability tensor**, it compares the original unmodified NIfTI exporter with the surgical inverse-plane native source patch.

The original source, model source files, HDF5 weights, config and real acquired MRI are authenticated by exact Git blob hashes before running. The model weights must actually change pre-load variable values. Hashes of each real prediction are recorded.

**What it measures:** whether the same trained-model result is preserved when output is attached to the acquisition affine. The oracle is the mathematically inverse axis transform of the *same* model output. It does not compare `ax` and `sag` predictions as if a medical network were orientation-equivariant, and does not claim segmentation improvement.

The model patch is a **predeclared 64³ center crop** of the acquired anonymized MRI. The cropped anatomy and output are real. `ax` is the negative control. `cor/sag` test previously unseen same-model export failures (nonaxial support already exists in the original loader). The fixed patch must exactly reproduce the original model prediction in the original acquisition coordinate frame. Empty/all-one network outputs cause explicit failure.

**Caveat:** One anonymized MRI and one original checkpoint, with a predeclared crop and per-crop upstream z-score. The source example mask is not used as ground truth, and no clinical validity or patient-safety benefit is claimed. A maintainer may reasonably want different regression coverage or an inverse full-resample contract.

Public CI: `.github/workflows/clingeom-brainmask-pretrained-planes.yml`.
