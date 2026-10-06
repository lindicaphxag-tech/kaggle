# ClinGeom public core artifact v0

ClinGeom is a research prototype for **proof-carrying patient-space semantics in medical AI**.

This branch exposes a minimal executable core:

- exact patient-space reindex certificates;
- voxel-Dice-perfect / patient-space-wrong counterexamples;
- model-derived feature lattices;
- channel-fibered `LatticeBundle` semantics for multi-phase tensors;
- PyTorch / torch.fx geometry propagation;
- tensor-valid but patient-space-invalid merge detection;
- constructive common-lattice repair synthesis;
- executable certificate-bound crop/resampling;
- automatic FX merge rewriting;
- prompt-language representation closure certificates for physical chart changes;
- minimal exact AABB -> OBB -> parallelepiped -> target-grid-mask compilation.

A source-level motivating case is MONAI DiNTS `FactorizedReduceBlock`, whose channel groups can sample complementary spatial phases while sharing one tensor spatial shape. This is **not claimed as a MONAI bug**. It motivates the type-system question: when does a medical feature tensor admit one patient-space affine, when does it require channel-conditioned lattices, and when must fusion be repaired or refused?

Claim boundary: self-authored public method evidence only. This is not clinical validation, external adoption, a prospective mechanism match, or an L8/L9 claim.

The prompt module also exposes a deterministic NumPy-only assay comparing the
analytic tight-AABB physical-region fidelity certificate against direct target-
chart sampling. This is representation evidence only; it is not a claim about
segmentation accuracy or clinical validity.
