# Receptive-support closure after physical alignment and dense fusion

## Previously unverified assumption

ClinGeom #44/#45 proved that *sample-center* alignment must occur before
irreversible channel mixing. Its first compiler output nevertheless copied
the caller's `target.receptive_field` into the mixed output, ignoring
the original receptive footprints and neighbors used by trilinear
interpolation. Thus a returned single-affine output could silently claim a
smaller patient-space support than the executed operation actually read.

**Centers and receptive supports are different typed properties.**

## Conservative construction

For an input fiber with center index `k`, receptive width `R_j > 0` along
input-voxel axis `j`, source sampling step `J_j > 0`, and one target-center
source coordinate `s_j`, trilinear interpolation uses integer centers
`floor(s_j)` and `ceil(s_j)` with nonzero weights when `s_j` is noninteger.
The input source footprint for each contributing center has radius `R_j/2`.
Consequently an upper bound on the *union* of all physical supports,
centered at the target sampling location, has width

    R_out,j <= R_source,j + 2 J_j max_{target i_j} d(frac(s_j(i_j)))

where `d(alpha)=0` at exact integer alpha=0, otherwise
`d(alpha)=max(alpha,1-alpha)`. This expression is an **admissible envelope**
when the declared feature receptive field bounds its actual contributing
input support. The construction is axis-separable; it checks all target
indices on each axis, not just the first and last grid points.

Since an admitted dense 1x1 convolution may use every channel, ClinGeom
sets the mixed output's declared receptive support to the componentwise
maximum of the envelopes of *all* incoming channel fibers. This does not
claim the true mixed receptive support is exactly the displayed box; it
is a conservative uniform bound.

## Independent falsifiers

1. Equal center maps, different input R=(1,1,1) vs (3,5,1): an exact
   pre-mix convolution's conservative output R is (3,5,1), **not**
   the target's nominal (1,1,1).
2. With target half-cell centers and a source jump of 1, interpolation
   raises the conservative mixed bound to R=(4,6,2), because both
   neighboring source receptive footprints contribute.
3. Modifying only the declared input receptive width must alter both
   the output support type **and** the output certificate digest, even
   when sample centers and the actual tensor values are unchanged.
4. Invalid/nonfinite input widths fail closed before mixed execution.
5. On the real pinned MONAI DiNTS factorized-reduction → dense 1x1
   `Conv3d` sequence from #44, the same target-center tensor has
   nominal R=(1,1,1), but interpolation expands the conservative
   output R to (4,4,4) in input-voxel index units.

## Reproduce

```bash
python -m pytest -q research/medical_geometry/test_support_envelope.py
PYTHONPATH=. python research/medical_geometry/benchmark_monai_nonidentifiable_mixing.py
```

Both are run by the public PyTorch and pinned-MONAI CI jobs. These
are first-party mechanical checks, not independent reproduction.

## Limits

Receptive-support enclosure is **not** feature equivalence. Combining a
wide-context channel and a pointwise channel on matching *centers* does
not make their semantics identical. The box describes potential source
support, not an inverse of pooling/blur nor a performance guarantee.
Future research can use these types to decide whether a requested fusion
must retain multi-support semantics or whether a common information-scale
projection can be safely constructed. No claim about a pretrained model's
clinical accuracy follows from this result.
