# ClinGeom — Typed Physical Geometry for Medical AI

> **The tensor shape can be valid while the patient-space meaning is wrong.**

ClinGeom is a public research artifact for making spatial semantics explicit in
medical-AI pipelines. It focuses on representation changes that preserve the
same physical patient-space object but can silently alter tensor-, prompt-, or
feature-lattice semantics.

## Three claims under test

### C1 — Patient-space geometry is a type, not just a tensor shape

Objects that share one array shape do not necessarily share one sampling lattice.
ClinGeom tracks spatial lattice semantics through model operations and rejects
merges that are tensor-valid but not patient-space compatible.

### C2 — Repairs must preserve the represented physical object

When two feature lattices can be reconciled, ClinGeom synthesizes a common-lattice
repair instead of silently fusing incompatible samples. For prompts, the same
principle becomes **representation closure**:

- AABB stays AABB only under monomial chart maps;
- otherwise lift to the least expressive exact supported primitive:
  `AABB -> OBB -> parallelepiped -> target-grid mask`.

A 45° in-plane rotation of a unit box has a tight target AABB prompt-region IoU
of exactly **0.5**; replacing the true transported region by that AABB therefore
doubles represented volume before the model even runs.

### C3 — Exact physical counterexamples are preferable to visual heuristics

The artifact uses analytic geometry and deterministic sampling to produce
falsifiable witnesses. It does not claim that visual similarity, segmentation
accuracy, or a valid affine alone proves physical semantic equivalence.

## 60-second reproduction

From the repository root:

```bash
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
```

Then run the two deterministic public assays:

```bash
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
```

`benchmark_prompt_representation.py` independently samples a 65^3 target grid
for rotations from 0° to 45° and requires the sampled physical-region fidelity
to agree with the analytic certificate within 0.002 absolute error.

## What to inspect first

| File | Purpose |
| --- | --- |
| `CLINGEOM_THEOREMS.md` | frozen theorem/claim boundary |
| `clingeom.py` | patient-space lattice semantics |
| `lattice_bundle.py` | channel-conditioned lattice bundles |
| `repair_synthesis.py` | constructive common-lattice repair |
| `prompt_representation.py` | representation closure and minimal exact lifts |
| `benchmark_clingeom.py` | deterministic geometry witness |
| `benchmark_prompt_representation.py` | analytic-vs-sampled prompt certificate |

## Current evidence state

| Evidence | State |
| --- | --- |
| algebraic/unit witnesses | **implemented** |
| NumPy-only deterministic benchmark | **implemented** |
| PyTorch / FX propagation and rewrite tests | **implemented** |
| MONAI DiNTS source/runtime motivating case | **implemented as a motivating case; not claimed as a MONAI bug** |
| public artifact | **yes** |
| independent third-party reproduction | **not yet** |
| external maintainer adoption | **not yet** |
| prospective external defect discovered by ClinGeom | **not yet** |
| clinical validation | **not claimed** |

## Claim boundary

ClinGeom is a research prototype, not a clinical safety system. It does **not**
claim that every tensor-lattice mismatch changes task accuracy, that prompt
fidelity bounds final segmentation quality, or that the cited motivating
framework code is defective. The strongest current claims are executable
representation/geometry statements that can be independently falsified.

## Independent reproduction

If you can reproduce—or falsify—one of the public witnesses on a clean
environment, please report the exact commit, Python/NumPy/PyTorch versions, and
the command/output. Negative results are useful and should be reported.

See the repository issue **Independent reproduction: ClinGeom physical semantics**
for the smallest third-party validation task.
