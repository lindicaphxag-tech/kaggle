# Genuine frozen pretrained 3-D T1 U-Net inference

Upstream: ecalabr/brain_mask, commit ca789fb26b70e359b61a8d52968813dd772fb4b1.

The workflow pins all original model source modules, configuration, real anonymized T1 MRI, accompanying brain mask and 10.7 MB official T1 Keras HDF5 checkpoint to immutable Git blob IDs.

Runs the original Unet3dBneck model under TensorFlow 2.13 CPU and loads the original weights strictly, not with skip_mismatch or by_name. Uses a predeclared center 64x64x64 crop and the source's nonzero whole-volume z-score preprocessing. The crop location does not use the provided brain mask.

Actual output evidence: probability array SHA-256, min/max/mean, foreground voxel count at a frozen 0.5 threshold, source hashes, crop world affine and an actual learned prediction image if nondegenerate.

A predeclared axis-0 orientation corruption and analytically corrected affine are tested on the true learned output. NiBabel native as_reoriented provides an independent reorientation oracle. The code refuses to invent a meaningful binary segmentation or repair improvement if the model gives an empty/full mask.

The included example brain mask is not verified expert ground truth; the source declares its example MRI pre-masked for anonymity. Agreement against it is descriptive only, not a segmentation-accuracy benchmark.

This is an actual frozen checkpoint forward experiment, not patient safety validation, external adoption, prospective issue discovery or proof of trained-model Dice improvement.

Run the public GitHub Actions workflow: .github/workflows/clingeom-pretrained-t1.yml. Its successful artifact, if one is produced, contains the original JSON metrics and PNG.
