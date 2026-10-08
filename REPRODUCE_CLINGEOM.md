# ClinGeom Independent Reproduction

**Frozen public version:** `release/clingeom-v0.2.0` at `7e570b05071ae65dc1abc636701c5a75ca1424a3`.

Use that branch for independent reproduction; `main` is the continuing research branch.

This document minimizes the cost of independently checking the public ClinGeom claims.

## A. NumPy-only semantic core

```bash
git clone https://github.com/lindicaphxag-tech/kaggle.git
cd kaggle
git checkout release/clingeom-v0.2.0
# frozen anchor: 7e570b05071ae65dc1abc636701c5a75ca1424a3
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_lattice_collapse.py
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

## D. Adversarial repair verification and postcondition

The public evidence is not only a synthetic mismatch. These tests attempt
to fool the verification/execution boundary with a correctly signed but
out-of-domain sampling map, incorrect source shapes, inaccurate atomic
repair-cost claims, and corrupted post-rewrite spatial phases.

```bash
python -m pytest -q \
  research/medical_geometry/test_repair_certificate.py \
  research/medical_geometry/test_torch_repair_certificate.py \
  research/medical_geometry/test_fx_postcondition.py
```

Expected: all three test modules pass. No threshold tuning is allowed.
Associated public source/evidence: [PR #16](https://github.com/lindicaphxag-tech/kaggle/pull/16)
and [PR #20](https://github.com/lindicaphxag-tech/kaggle/pull/20).

## E. Pinned external highdicom witness

[Execution record](research/medical_geometry/upstream_repros/HIGHDICOM_454_RESULT.md)
and [minimal patch proposal](research/medical_geometry/upstream_repros/HIGHDICOM_454_PATCH_PROPOSAL.md)
are public. The pinned CI workflow installs the audited highdicom #454 PR head
and MONAI 1.6.1. It observes exact affine round-trip but lost automatic
`FrameOfReferenceUID`. This is a third-party-code *observation*, not
third-party *confirmation* or adoption.


## F. Conditional physical interpolation-value certificates

ClinGeom does **not** infer that interpolating two spatially aligned neural
features makes their values semantically equivalent. It now offers an
optional, fail-closed error budget under a declared **uniform patient-space
Hessian operator norm bound** on each interpolated feature field.

```bash
python -m pytest -q research/medical_geometry/test_interpolation_certificate.py
python -m pytest -q research/medical_geometry/test_interpolation_fx_gate.py
```

Expected: an analytic quadratic field on 2 mm voxels attains the sharp
0.3 output-value-unit bound for curvature 0.2 units/mm², both with direct
NumPy eight-corner interpolation and the actual PyTorch grid_sample repair.
Unknown curvature, excessive curvature, and label-nearest interpolation
must fail when a numerical value-error budget is requested.

This is an assumption-conditional guarantee, not a measured reliability
score on pretrained neural features. For pretrained models, external
curvature envelopes must be established before promoting this claim.

## G. Curvature-gated nonlinear MONAI channel-fiber transport

**Source/evidence:** [merged PR #35](https://github.com/lindicaphxag-tech/kaggle/pull/35)
· [author-operated five-job successful CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37711065982).

Run with the pinned MONAI source revision used by the public CI workflow:

```bash
PYTHONPATH=. python research/medical_geometry/benchmark_monai_dints_quadratic_bound.py
python -m pytest -q research/medical_geometry/test_channel_fiber_runtime.py
```

The first script runs the real pinned MONAI DiNTS `FactorizedReduceBlock`
on an **analytic quadratic** in millimetre patient space. The input
curvature is known from the chosen synthetic function rather than
estimated from trained activations.

Falsifiable first-party values on float32 CPU:

- spacing `[0.7, 0.7, 5.0]` mm;
- two distinct channel-phase fiber supports;
- declared world Hessian operator-norm bound `0.02` value units/mm²;
- sharp physical trilinear bound `0.19485` value units per fiber;
- naïve shared-index readout MAE `1.49385`;
- type-directed repaired MAE `0.19485`, max error about `0.1948514`;
- omit one channel's curvature bound → fail closed;
- tighten allowed value error below `0.19485` → fail closed.

**Critical interpretation:** geometry alignment does not imply zero feature
error. This experiment intentionally retains a nonzero residual and
demonstrates its analytic bound is attainable on an external library's
real forward path. It does not establish the required curvature bounds
for learned medical features or patient outcomes, and does not claim
MONAI implements the wrong topology.

Independent reports should include exact Git commit, MONAI source SHA,
PyTorch version, full JSON and failures, and should not copy the
author-operated CI results as third-party validation.

## H. Repair-before-fusion identifiability challenge

**Merged implementation:** [ClinGeom PR #44](https://github.com/lindicaphxag-tech/kaggle/pull/44).
**Complete final author-operated run:** [GitHub Actions 37717636413](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37717636413).

The experiment controls the input fields and conv weights but uses real
MONAI DiNTS `FactorizedReduceBlock` and real PyTorch dense `1x1x1` fusion.
The two analytic input worlds produce indistinguishable post-mix tensors
despite different required co-located outputs. A successful run therefore
validates **where** an admissible physical repair must be executed,
not clinical segmentation accuracy.

```bash
python -m pytest -q research/medical_geometry/test_pre_mix_compiler.py
PYTHONPATH=. python research/medical_geometry/benchmark_monai_nonidentifiable_mixing.py
```

Falsification checks:

- max difference between the two *already mixed* tensors <= 2e-6;
- true patient-space targets differ by 0.7 units;
- any post-mix-only estimator has worst-case lower bound 0.35;
- each independently corrected pre-mix output matches its target
  within 2e-6;
- any already mixed/multi-support feature is refused by point-affine repair;
- a non-pointwise/grouped fusion fails, rather than borrowing this proof;
- conditional output interpolation budget, when requested, transports
  via the actual mixer weights and rejects over-budget repair.

External reviewers should use their own fork and quote **their own**
run URL and exact dependencies. The merged project's author-operated
green CI alone does not establish independent reproduction.

**Scope:** this is elementary nonidentifiability in a specific physical
channel-fusion topology. We do not claim a new abstract information-theory
theorem, downstream trained medical-model improvement, or a bug in MONAI.

## I. Bitwise-exact fusion indistinguishability

**Merged PR:** [#45](https://github.com/lindicaphxag-tech/kaggle/pull/45).
**Full successful CI:** [Actions 37718214998](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718214998).

The previous 0.7 mm case (#44) had analytically identical postmix signals,
but an observed float32 residual of 4.77e-7. This tightened 0.5 mm
(binary-representable) case checks **IEEE-754 int32 bit reinterpretations**
of the two real MONAI/PyTorch postmix tensors, not just numerical equality.

```bash
python -m pytest -q research/medical_geometry/test_monai_dints_runtime.py
PYTHONPATH=. python research/medical_geometry/benchmark_monai_bitwise_identical_fusion.py
```

The independent run must verify:

- `post_fusion_tensors_bitwise_equal: true` (not just allclose);
- `post_fusion_max_abs_difference: 0.0` exactly;
- `correct_patient_target_difference: 0.5` value units;
- post-fusion deterministic-estimator worst-case lower bound `0.25`;
- pre-mix repair max error `0` for the zero world and at most `1e-6`
  for the signed analytic nonzero world (author run: `5.96e-8`).

Report your own SHA, MONAI/torch/python environment, raw JSON and failures.
These are constructed analytic fields, not real CT/MRI test images.
