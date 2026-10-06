# ClinGeom Independent Reproduction

This document minimizes the cost of independently checking the public ClinGeom claims.

## A. NumPy-only semantic core

```bash
git clone https://github.com/lindicaphxag-tech/kaggle.git
cd kaggle
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
```

Required reproduction outcomes:

- all selected tests pass;
- the `dice_lies` synthetic witness reports voxel Dice `1.0` with non-zero patient-space centroid error;
- the pooling control reports near-zero affine mismatch;
- prompt analytic-vs-sampled error remains below the benchmark's frozen assertion.

## B. PyTorch / torch.fx semantic compiler

```bash
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

Required reproduction outcomes:

- graph propagation / merge checks pass;
- factorized reduction carries more than one patient-space sampling lattice;
- a single-affine representation is rejected when the channel phases disagree;
- repair/rewrite tests preserve the declared semantic invariants.

## C. Pinned MONAI runtime witness

The public CI installs MONAI from the exact audited revision:

`efb240ac97fcb619d842b2c7872d7db942eb7cd9`

and runs:

```bash
python -m pytest -q research/medical_geometry/test_monai_dints_runtime.py
```

This is an architecture-level phase/lattice witness. It is **not** a claim that MONAI DiNTS is inaccurate, unsafe or buggy.

## Reporting an independent reproduction

Please include:

1. repository commit SHA;
2. OS and Python version;
3. NumPy / PyTorch / MONAI versions where relevant;
4. exact commands;
5. complete pass/fail output;
6. any divergence from the public protocol.

Negative results are explicitly useful. Do not tune thresholds after seeing the result.

## What does not count as external validation

- self-merged pull requests in this repository;
- rerunning CI under the author's account;
- stars, forks or downloads alone;
- historical bugs used only as motivating examples;
- unpublished private experiments.
