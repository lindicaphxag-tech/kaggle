# Public Research Artifacts

[![ClinGeom public core artifact](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml/badge.svg?branch=main)](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml)

This repository is a public evidence and reproducibility surface for two research lines:

1. **ClinGeom** — typed patient-space semantics for medical AI pipelines.
2. **SemRepair evidence transport** — prospective, timestamped semantic-reliability observations for embodied software.

They are deliberately kept separate in claims: public self-authored artifacts are **not** external adoption, and prospective cases are **not** counted before their frozen criteria are met.

## ClinGeom — start here

ClinGeom asks a narrow question with broad consequences:

> When a medical image, prompt, annotation, feature tensor or deformation field changes representation, does it still denote the same patient-space object?

The public core currently contains:

- exact patient-space reindex certificates;
- voxel-Dice-perfect / patient-space-wrong counterexamples;
- model-derived feature lattices and channel-fibered lattice bundles;
- PyTorch / `torch.fx` spatial-semantics propagation and merge checking;
- constructive common-lattice repair synthesis;
- prompt representation-closure certificates;
- exact AABB → OBB → parallelepiped → target-grid-mask lifting;
- a pinned MONAI DiNTS runtime witness used as an architecture-level motivating case, **not** as a MONAI bug claim.

### 3-minute NumPy reproduction

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

The first benchmark should expose a case where voxel Dice is exactly `1.0` while patient-space centroid error is non-zero, plus a negative control where shape-only geometry is valid. The prompt benchmark must keep analytic-vs-sampled representation error below its frozen threshold. It also includes the exact 45° rotation → inverse-rotation witness: deferred projection returns prompt IoU `1.0`, while eager intermediate AABB projection returns `0.25`.

### PyTorch compiler reproduction

```bash
python -m pip install numpy pytest
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q \
  research/medical_geometry/test_torch_geometry.py \
  research/medical_geometry/test_torch_fx_geometry.py \
  research/medical_geometry/test_torch_fx_bundle.py \
  research/medical_geometry/test_torch_repair.py \
  research/medical_geometry/test_fx_geometry.py \
  research/medical_geometry/test_fx_rewrite.py
PYTHONPATH=. python research/medical_geometry/benchmark_channel_fibers.py
```

See [`CLINGEOM_PUBLIC_ARTIFACT.md`](CLINGEOM_PUBLIC_ARTIFACT.md), [`REPRODUCE_CLINGEOM.md`](REPRODUCE_CLINGEOM.md), and [`research/medical_geometry/CLINGEOM_THEOREMS.md`](research/medical_geometry/CLINGEOM_THEOREMS.md) for the full claim boundary.

## SemRepair prospective evidence transport

The `transport/` surface is intentionally narrower: it freezes first-seen public issue evidence, prediction digests, deadlines and later resolution evidence. It does **not** award itself research credit or retroactively convert known cases into prospective discoveries.

## Evidence status

- ClinGeom public method artifact: **available**.
- ClinGeom external adoption / independent reproduction: **not yet claimed**.
- ClinGeom prospective external mechanism hit: **not yet claimed**.
- SemRepair prospective transport: **active public evidence surface**.

## Contributing / independent reproduction

Independent reproductions are welcome. Please report exact commit, Python/framework versions, commands, full outputs, and negative results. A reproduction only counts as independent evidence when it is performed from public artifacts without private thresholds or hidden expected outputs.

## License

No repository-wide open-source license has been declared yet. Do not infer permissions beyond GitHub's repository access terms until an explicit license is added.
