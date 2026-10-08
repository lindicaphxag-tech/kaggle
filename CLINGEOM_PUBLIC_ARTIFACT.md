# ClinGeom public research artifact

ClinGeom is a research prototype for **typed patient-space semantics in medical AI**.

> Tensor validity is not physical validity.

The public core asks whether images, feature lattices and prompts still denote
the same patient-space object after indexing, resampling, fusion and coordinate
changes—and synthesizes an exact repair or representation lift when that is
possible.

## Latest authentic pretrained 3-D MRI execution

[**Merged PR #68**](https://github.com/lindicaphxag-tech/kaggle/pull/68) connects the *same pinned public T1 U-Net checkpoint's* true three-plane NIfTI predictions to a fixed downstream prediction-level ensemble, comparing incorrect same-index fusion against explicitly inverse-reoriented physical-grid fusion. [Public genuine-model CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37819406496) and the [source / result / negative control dossier](research/medical_geometry/upstream_repros/REAL_PRETRAINED_PLANE_FUSION_RESULT.md) record one-example paired weak-reference Dice 0.0458 vs 0.1700 and physical HD95 50.93 vs 40.90 mm. All six new adversarial source-provenance tests passed. Native exporter/orientation libraries remain fair alternative baselines.

This is **not** an expert clinical segmentation gain, learned internal encoder fusion, multisubject validation or a third-party accepted contribution. Further scientific novelty depends on showing the compiler's channel-dependent pre-mix transport on *actual trained internal features* across independently annotated studies and above standard spatial transform tracking.

## Start here

Frozen evidence pack: [`research/medical_geometry/EVIDENCE_V0.2.0.md`](research/medical_geometry/EVIDENCE_V0.2.0.md) · machine-readable manifest: [`EVIDENCE_MANIFEST_V0.2.0.json`](research/medical_geometry/EVIDENCE_MANIFEST_V0.2.0.json)

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
- exact minimal patient-lattice type/fiber certificates and exact minimax single-affine collapse radius for parallel lattice families;
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
