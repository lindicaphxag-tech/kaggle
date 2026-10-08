# ClinGeom public research artifact

ClinGeom is a research prototype for **typed patient-space semantics in medical AI**.

> Tensor validity is not physical validity.

The public core asks whether images, feature lattices and prompts still denote
the same patient-space object after indexing, resampling, fusion and coordinate
changes—and synthesizes an exact repair or representation lift when that is
possible.

## Latest independently sourced annotated MRI audit — real pretrained 3-D U-Net

The strongest available *medical imaging* experiment is now [merged #72](https://github.com/lindicaphxag-tech/kaggle/pull/72), executed on the **exact original public pretrained** TensorFlow T1 3-D U-Net and **three independent public subject identifiers** (NFBS + 2 SynthStrip) with externally prepared manual/mask-refined references, three predeclared 64³ patches per subject, six method variants and unchanged original weights.

[Original learned-model external MRI CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37826696442) · [all-case report](research/medical_geometry/upstream_repros/REAL_EXTERNAL_ANNOTATED_HOLDOUT_RESULT.md). A clear **negative finding**: the unpermuted canonical-RAS first branch performs *better* than every evaluated multi-view fusion on this extremely small heldout ROI pilot. Patient-macro Dice/physical HD95: first branch **0.3524 / 30.52 mm**, correctly reoriented probability fusion **0.1816 / 44.36 mm**, original nonlinear spatially conjugated feature fusion **0.1373 / 48.00 mm**. Empty-mask HD95 aggregation was explicitly corrected *after initial inspection* and labeled as such; the report includes the original measurement and all failure arms.

The mechanistic chain is independently inspectable: [#68 original source exporter and corrected coordinate fusion](https://github.com/lindicaphxag-tech/kaggle/pull/68), [#70 authentic learned 32-channel feature cut and exact reduction to ordinary logit fusion](https://github.com/lindicaphxag-tech/kaggle/pull/70), [#71 authentic original pretrained nonlinear residual feature/operator conjugation and nonzero directional operator commutator](https://github.com/lindicaphxag-tech/kaggle/pull/71), [#72 real separate annotated subjects and the strong single-view control](https://github.com/lindicaphxag-tech/kaggle/pull/72). **External maintainers have NOT endorsed or independently reproduced this research.**

**Unresolved central scientific question:** Spatial/provenance correctness is not sufficient to authorize mixing latent representations or original learned nonlinear operators from distribution-shifted views. A pre-frozen source-label-free reliability gate is undergoing genuinely new [six-subject prospective verification #73](https://github.com/lindicaphxag-tech/kaggle/pull/73), and no performance benefit is asserted before its actual results. No whole-volume clinical benchmark, broad patient-level statistical confidence, native multi-fiber DiNTS learned-feature repair or external adoption has been established.

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
