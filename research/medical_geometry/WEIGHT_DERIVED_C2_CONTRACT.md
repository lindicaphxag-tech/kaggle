# ClinGeom — audited weight-derived physical Hessian propagation

## What was added

Earlier the nonlinear MONAI proof used a manually supplied output-fiber
Hessian upper bound. This procedure derives the **output bound from actual
model weights**, while requiring a genuine bound on the underlying **input**
continuous scalar field. It is a deliberately narrow proof-carrying execution
contract; no bound for a nonlinear pretrained network is inferred.

## Statement and derivation

Let each input channel be a C² scalar field `f_c(p)` in the same patient-space
coordinate system. Suppose for every patient-space location touched by the
model's finite sampling support:

    ||Hess(f_c)(p)||_2 <= B_c

For a linear `1×1×1` Conv3d branch with real weights and constant bias:

    y_o(p) = sum_c W[o,c] f_c(p) + bias[o]

Differentiation commutes with the fixed linear combination, so:

    ||Hess(y_o)(p)||_2
      = ||sum_c W[o,c] Hess(f_c)(p)||_2
      <= sum_c abs(W[o,c]) ||Hess(f_c)(p)||_2
      <= sum_c abs(W[o,c]) B_c.

A stride-2 branch or one-voxel source crop changes the output **sample
locations**, not this continuous-field curvature inequality. ClinGeom's
existing channel-fibered lattice type preserves those independent phases.

Every output channel gets its own derived bound. The runtime takes the
maximum over the output channels in each fiber before applying the
physically calibrated trilinear error certificate. Thus it never silently
substitutes a nominal feature channel's bound for its neighbor.

## Narrow executable theorem preconditions

The current derivation accepts the **exact** pinned MONAI DiNTS
`FactorizedReduceBlock` class, only when:

- both convolution branches are native `nn.Conv3d` with 1×1×1 kernels,
  stride 2, no padding, no dilation and no groups;
- `.act` and `.norm` are explicitly `nn.Identity` (a nonlinear ReLU/PReLU
  can destroy C² differentiability, and active normalization has not been
  proved by this rule);
- every input channel has an externally justified, finite, nonnegative
  uniform patient-space Hessian bound over the complete touched region;
- the model's weights/biases are finite and current at execution time;
- a valid channel-fiber lattice map and the existing bounded in-domain
  resampling authorization both succeed.

Any unsupported model/activation, missing input premise, different source
support or output numerical budget violation **rejects**. This is not an
automatic input-Hessian estimator and it does not certify real nonlinear
learned networks.

## Quantitative pinned MONAI witness

For input `f(p) = (B/2) ||p||²`, `B=0.02` output units/mm² and input
spacing `[0.7, 0.7, 5.0]` mm, choose two live MONAI branch weights
`+2` and `−3`. The provably valid output bounds are respectively
`0.04` and `0.06`, with source-grid interpolation error limits
`0.38970` and `0.58455` output units.

The benchmark compares the actual frozen MONAI block's forward tensor
against that known continuous ground truth, and reports:

- both derived output bounds and the actual per-fiber interpolation errors;
- a SHA-256 digest of the model's current conv weights and input assumptions;
- a naïve shared-index baseline and the physically repaired error;
- failed authorization at budget 0.58 vs permitted budget 0.60;
- failed derivation under unsupported nonlinear activation/normalization;
- changed output bound *and* provenance digest when a weight changes.

Reproduce using exactly the MONAI Git SHA and CPU PyTorch dependency
versions in `.github/workflows/clingeom-core-artifact.yml`:

```bash
python -m pytest -q research/medical_geometry/test_weight_curvature.py
PYTHONPATH=. python research/medical_geometry/benchmark_weight_derived_monai.py
```

## What this does NOT prove

This inequality is classical linear operator norm calculus, not a new
theorem. Its novel *candidate integration* is type-directed physical
compiler authorization across separately sampled channel fibers. It
does not prove an input-Hessian premise for real CT/MRI values or a
pretrained feature encoder; it does not certify non-C² activations,
learned representation equivalence, clinical safety or downstream Dice.

Since there is no third-party reproduction yet, all passing runs remain
**author-operated software evidence**, not independent adoption.
