# ClinGeom v0.1.0 — Frozen Public Research Artifact

Release branch: `release/clingeom-v0.1.0`

Frozen commit:

`016b09d1d88f86ee20da5ddb7dab5fd471c50a27`

Public core CI run:

`37701895349` — conclusion: **success**

## Frozen method surface

This version freezes the public ClinGeom method core through theorem T13:

- patient-space reindex certificates;
- voxel-Dice-perfect / patient-space-wrong witnesses;
- feature-lattice propagation;
- channel-conditioned `LatticeBundle` semantics;
- minimal point-affine partition certificate;
- PyTorch / torch.fx geometry propagation;
- tensor-valid / patient-space-invalid merge detection;
- constructive common-lattice repair;
- representation-closure certificates for prompt primitives;
- exact `AABB -> OBB -> parallelepiped -> target-grid mask` lifting;
- prompt wrapping-debt analysis under repeated restricted projection.

## T13 addition

For point-valued channel fibers, one global patient-space affine is exact iff
every fiber carries the same affine center map. The artifact separately reports:

1. the number of distinct patient-lattice affine types;
2. the minimum number of channel-contiguous fibers required by channel order;
3. whether any channel is already multi-support, in which case no point-affine
   channel partition is exact and the semantic representation must be lifted.

## Reproduction

Use:

```bash
git clone https://github.com/lindicaphxag-tech/kaggle.git
cd kaggle
git checkout release/clingeom-v0.1.0
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
```

See `REPRODUCE_CLINGEOM.md` for PyTorch / torch.fx and pinned-MONAI checks.

## External-evidence state at freeze

- public artifact: **yes**;
- public CI: **green**;
- independent third-party reproduction: **0**;
- external maintainer adoption: **0**;
- prospective ClinGeom hit after public activation: **0**;
- clinical validation: **not claimed**.

This file intentionally freezes the negative evidence state as well as the
implemented claims. Future external reproduction/adoption belongs to later
evidence records and must not be retroactively attributed to v0.1.0.