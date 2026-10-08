# Repair-before-fusion: irrecoverability witness and authorized execution

## Concrete question

Given a channel tensor with independent patient-space sampling charts,
can a post-fusion interpolator reconstruct the *correct* co-located
channel sum after a dense convolution has mixed those channels?

**Not in general.** This artifact gives two continuous fields that are
indistinguishable after an actual MONAI/PyTorch mixed forward yet require
different answers at every co-located target point. The failure is not
a numerical tolerance issue; it is missing information.

## Paired-world witness on pinned MONAI DiNTS

Let the patient-space first-axis voxel spacing be s = 0.7 mm. MONAI's
`FactorizedReduceBlock` has two stride-2 branches. With controlled
one-hot 1×1 weights, branch A samples its selected input f at even
source indices 2i, whereas B samples g at odd indices 2i+1.

World 0:

    f0(p)=0,   g0(p)=0

World 1:

    f1(p)=p_x, g1(p)=s-p_x.

The ordinary `Conv3d(2,1,kernel_size=1)` with weights (1,1) observes:

    y0[i]=f0(2is)+g0((2i+1)s)=0
    y1[i]=f1(2is)+g1((2i+1)s)=2is + s-(2i+1)s=0.

Thus **the entire post-fusion tensor is identical**, while for every
physical target location p, the intended common-space sum differs:

    target0(p)=f0(p)+g0(p)=0
    target1(p)=f1(p)+g1(p)=s=0.7.

Any deterministic post-fusion-only algorithm receives exactly the same
input in both worlds and must produce the same answer. At each target
location, one of its two absolute errors must be at least s/2 = 0.35.

This is an elementary indistinguishability proof, not a newly discovered
information theory theorem. Its importance here is the precise topology
and physical chart where semantic repair authorization is still possible.

## Executable pre-mix repair

`physically_align_then_mix` changes the order to:

    raw MONAI fiber outputs
      -> individually verified physical target-grid re-sampling
      -> admitted 1x1 dense Conv3d
      -> a single patient-space output lattice.

The MONAI branch outputs still exist at the insertion point. As both
adversarial fields are linear in patient coordinates, their individual
resamples are exact in real arithmetic; source-dependent additive
cancellation is correctly reproduced only after alignment.

After mixing, the existing `dense_conv_mix_bundle` deliberately carries
**multiple supports**; `repair_channel_fibers` rejects an invented
post-hoc unique affine.

## Optional numerical error authorization

Given verified per-source-fiber interpolation error limits `E_c`,
for a pointwise linear mixer with weights W the output interpolation
error of channel o is conditionally bounded by:

    E_out[o] <= sum_c abs(W[o,c]) E_c.

This bound is composed with the existing geometry/domain certificate and
rejects a requested output budget that is too small. We read and hash
the actual mixer parameters and all source repair digests. This law is
classical linear error propagation, not a new result in itself.

## Reproduce and falsify

On the same CPU PyTorch / pinned MONAI runtime specified by
`.github/workflows/clingeom-core-artifact.yml`:

```bash
python -m pytest -q research/medical_geometry/test_pre_mix_compiler.py
PYTHONPATH=. python research/medical_geometry/benchmark_monai_nonidentifiable_mixing.py
```

Check all of the following:

- two adversarial postmix tensors differ by at most 2e-6 (ideally 0);
- the known target patient-space sums differ by 0.7;
- the postmix minimax lower bound is 0.35;
- both admissible pre-mix results match their respective target within 2e-6;
- attempting a postmix point-affine repair is refused;
- unsupported strided/grouped/non-pointwise mixer is refused;
- under B=0.02 values/mm² and 0.7×0.7×5 mm voxel spacings,
  source fiber sampling bounds are 0.19485; a mixer (2,-3) propagates
  0.97425, and (2,-4) propagates 1.1691 (value units).

## What is not established

This uses analytic, constructed functions and controlled but real
MONAI/PyTorch operators; it does not show a clinical benchmark
improvement, an error in MONAI's published implementation, or a novel
general identifiability theorem. The original architecture does not
promise physically co-located channel fusion, so this is a proposed
type-directed intervention with changed computation, not a drop-in
equivalent reimplementation. Third-party reproduction is outstanding.
