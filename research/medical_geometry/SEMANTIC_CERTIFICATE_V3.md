# ClinGeom v3: physical-semantic certificate binding

## Failure mode

Earlier `physically_align_then_mix()` committed its output SHA-256 provenance
fingerprint to convolution parameters, each discrete source-to-target sampling
plan and receptive support widths. Yet the same byte-valued voxel lattice
can be interpreted under **different patient millimetre scales** or
different DICOM FrameOfReferenceUIDs. The earlier fingerprint omitted both.

In the optional Hessian-gated path this omission is more serious: the
same integer-index resampling under a 1×1×1 mm versus 1×1×5 mm
voxel-to-world chart has different **numerical interpolation-error limits**.
The previous SHA-256 fingerprint could still be the same even when the
derived per-output value-error bounds were different.

This is a semantic-provenance collision, **not a demonstrated SHA-256
cryptographic collision**: the original bytes hashed simply omitted an
essential field. Nor is an unhashed/signed SHA-256 digest proof that the
caller-supplied patient frame or curvature bounds are medically true.

## Corrected contract

After the existing source-plan / weight / receptive-support fields, the
v3 deterministic provenance digest now includes:

- input patient voxel-to-world 4×4 affine in millimetres (or explicit unknown);
- declared DICOM FrameOfReferenceUID (or explicit unknown);
- source and target tensor shapes;
- declared per-fiber world-Hessian bounds and allowed fiber error budget;
- declared maximum output interpolation-value error budget;
- calculated per-output error bounds, if conditional proof was requested;
- original model weights, original per-fiber spatial repair digests,
  contributing receptive support widths and derived output envelope.

The patient affine must be finite with invertible 3×3 linear part and
a correct homogeneous fourth row. A provided frame UID must be nonblank.
Unknown frame identity is represented as **unknown**, never silently
substituted for a specific patient frame.

The digest remains a **non-authenticated fingerprint** of declared
inputs and certified computations. A dishonest caller can supply false
frame/curvature metadata or recompute the digest; this layer has no
external provenance or patient-data authentication.

## Falsification

`test_semantic_certificate_binding.py` checks:

1. Equal source tensor, spatial index mapping, convolution weights and
   source plan digests under different patient mm scales → different
   output numerical bounds **and different v3 semantic digests**.
2. Changing a trusted Hessian premise without changing any tensor bytes
   → a different digest.
3. Changing only FrameOfReferenceUID → a different digest, even if all
   pixel/affine/numerical error values remain identical.
4. Unknown frame identity does not impersonate declared identity.
5. Noninvertible/nonfinite input affine and blank UID fail closed.
6. Identical inputs remain deterministic across runs.

Run:

```bash
python -m pytest -q research/medical_geometry/test_semantic_certificate_binding.py
```

## External use

A consuming library should not treat the digest as a digital signature
or as proof that the DICOM frame matches a patient examination. It may
compare digests to detect a mismatch in declared physical-semantic
configuration. Independent source provenance, original-image UID checks
and assumption validation remain the caller's responsibility.
