# ClinGeom: Global normalization makes local receptive-support claims unsound

## The falsifying example

PyTorch's `InstanceNorm3d` (without tracked running statistics) computes spatial
mean and variance from the *same volume being normalized*. `GroupNorm`
computes per-example group statistics across all spatial positions. Training
`BatchNorm3d` also uses input-dependent statistics.

Take a 9³ MRI-like feature field `x`, with center value 1. Form `x'`
by changing **only one distant corner** from 0 to 8. The center input is
unchanged, yet the normalized center output changes because the per-volume
statistics changed. The dependency spans the spatial volume, not the
identity lattice's reported receptive support width `[1,1,1]`.

This is classical normalization behavior; it is **not a new discovery about
neural networks, a MONAI defect, or a clinical segmentation result**.
It exposes an unsound internal assumption in an emerging proof compiler.

## Corrected trust boundary

The ClinGeom compiler must **fail closed** rather than reuse a local support
annotation when exact global dependency typing is not implemented.

| Operator state | Correct certification policy |
|---|---|
| `GroupNorm` in train or eval | Refuse local support |
| `InstanceNorm3d` training | Refuse local support |
| `InstanceNorm3d` eval, `track_running_stats=False` | Refuse local support |
| `BatchNorm3d` training | Refuse local support |
| `BatchNorm3d` eval, `track_running_stats=False` | Refuse local support |
| `InstanceNorm3d` or `BatchNorm3d` eval, tracked frozen running stats | Admit as pointwise in space |

The statement is about spatial *information dependency*, not whether the output
tensor shape or grid-center affine changes.

## Second previously unsound transition: residual addition

The bundled FX compiler's old pointwise addition required center-grid matching,
then returned the **left input's full bundle unchanged**. An identity branch and
a same-center `AvgPool3d(k=(3,5,1))` branch therefore returned nominal support
`[1,1,1]` when the left operand was first even though the output contained
information from `[3,5,1]`.

The corrected operator retains the per-fiber union of both input supports,
and the test verifies the same result for both operand orders, with an actual
off-center impulse that changes the output.

## Reviewer-verifiable execution

```bash
python -m pip install numpy pytest scipy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
PYTHONPATH=. python -m pytest -q research/medical_geometry/test_nonlocal_spatial_support.py
PYTHONPATH=. python -m pytest -q research/medical_geometry/test_torch_geometry.py research/medical_geometry/test_torch_fx_bundle.py
```

The new tests execute actual PyTorch operators and demand:
- a distant perturbation changes the center under global normalization;
- the compiler rejects a spurious local support certificate;
- frozen running-stat normalization remains a legitimate spatially local rule;
- pointwise residual addition preserves the wider branch's spatial support;
- branch order cannot erase support.

The next research step **must not** call a trained default MONAI network's
normalized feature output 'local' without a more expressive dependency type,
such as a global-support field, an explicit volume-context dependency graph
or a provably valid lower-dimensional sufficient statistic.

## Flagship-paper relevance and limits

This closes a potentially paper-invalidating gap in ClinGeom's representation
contracts. A legitimate flagship contribution is *compositional patient-space
data-dependency typing before irreversible operators*, with numerical
authorization and concrete high-profile medical-network evidence.

However, this patch is **first-party proof soundness**, not a newly accepted
external contribution, a novel normalization theorem, a trained-model accuracy
uplift, or a top-tier publication result. No promise of clinical certainty
follows from these tests.
