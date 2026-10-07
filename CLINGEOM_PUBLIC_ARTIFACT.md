# ClinGeom public research artifact

ClinGeom is a research prototype for **typed patient-space semantics in medical AI**.

> Tensor validity is not physical validity.

The public core asks whether images, feature lattices and prompts still denote
the same patient-space object after indexing, resampling, fusion and coordinate
changes—and synthesizes an exact repair or representation lift when that is
possible.

## Start here

Frozen evidence pack: [`research/medical_geometry/EVIDENCE_V0.1.0.md`](research/medical_geometry/EVIDENCE_V0.1.0.md) · machine-readable manifest: [`EVIDENCE_MANIFEST_V0.1.0.json`](research/medical_geometry/EVIDENCE_MANIFEST_V0.1.0.json)

- Research entry point: [`research/medical_geometry/README.md`](research/medical_geometry/README.md)
- Frozen theorem/claim boundary: [`research/medical_geometry/CLINGEOM_THEOREMS.md`](research/medical_geometry/CLINGEOM_THEOREMS.md)
- Independent reproduction task: [Issue #9](https://github.com/lindicaphxag-tech/kaggle/issues/9)
- Prompt-closure falsification task: [Issue #5](https://github.com/lindicaphxag-tech/kaggle/issues/5)

## 60-second NumPy core

```bash
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
```

## Current method surface

- exact patient-space reindex certificates;
- voxel-Dice-perfect / patient-space-wrong counterexamples;
- model-derived feature lattices and channel-conditioned `LatticeBundle` semantics;
- PyTorch / `torch.fx` geometry propagation;
- tensor-valid but patient-space-invalid merge detection;
- constructive common-lattice repair synthesis;
- executable certificate-bound crop/resampling;
- automatic FX merge rewriting;
- prompt representation-closure certificates;
- minimal exact `AABB -> OBB -> parallelepiped -> target-grid mask` lifting.

A source-level motivating case is MONAI DiNTS `FactorizedReduceBlock`, whose
channel groups can sample complementary spatial phases while sharing one tensor
spatial shape. This is **not claimed as a MONAI bug**. It motivates the type
question: when does a medical feature tensor admit one patient-space affine,
when does it require channel-conditioned lattices, and when must fusion be
repaired or refused?

## Evidence state

- public executable method artifact: **yes**;
- deterministic analytic-vs-sampled prompt assay: **yes**;
- independent third-party reproduction: **not yet**;
- external maintainer adoption: **not yet**;
- prospective external defect: **not yet**;
- clinical validation: **not claimed**.

ClinGeom deliberately distinguishes self-authored public evidence from external
validation. A future successful reproduction, maintainer confirmation or
upstream integration should be recorded separately rather than inferred from
repository visibility or self-merged pull requests.
