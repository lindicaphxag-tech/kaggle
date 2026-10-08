# ClinGeom: preserve receptive-support unions under same-center deduplication

## Original failing safety invariant

Previous `lattice_bundle.py::_deduplicate_supports()` grouped channel fibers
only by output `(shape, jump, start)`. It then applied `dict.setdefault()`
to keep the **first** feature's complete `FeatureLattice`, including its
`receptive_field`. Equal *center grids* do not imply equal source support.

A physically plausible 3-D network block has:

```text
MRI voxels --- identity branch --------------------------              center grid G; receptive width [1,1,1]     +-- dense 1x1 Conv3d
MRI voxels --- AvgPool3d(kernel=(3,5,1),padding=(1,2,0))--/
              center grid G; receptive width [3,5,1]
```

`all_bundle_supports()` previously returned RF `[1,1,1]` whenever the
identity branch happened to be first, instead of `[3,5,1]`. Merely
swapping channel order changed the exported proof, although the output
could still depend on spatially distant MRI voxels. The same bug appeared
in `dense_conv_mix_bundle()`, `factorized_reduce_bundle()` and related
bundle transformations that call the same deduplication helper.

This directly violated the stronger receptive-support guarantee introduced
in [PR #48](https://github.com/lindicaphxag-tech/kaggle/pull/48):
the support annotation must be a valid conservative bound on actual input
voxels contributing to each exported feature.

## Mathematically admissible correction

If supports have an identical `(shape, start, jump)`, their center lattice
can be represented once, but their receptive support footprints must be
coalesced using the componentwise supremum:

```text
R_union[j] = max(R_1[j], R_2[j], ..., R_k[j]),  j in {0,1,2}.
```

This enclosing box is correct for same-centered axis-aligned input-index
footprints. It is **not** a claim of identical embeddings, equal receptive
functions, or the ability to recover lost information after dense fusion.
Invalid (NaN/infinite/nonpositive) receptive widths fail closed.

Distinct center lattices remain distinct and continue to require channel
fiber provenance or prior spatial alignment. Deduplicating centers still
serves point-affine type counting; it must not erase support metadata.

## Reproducible red-to-green proof

- [Original failing commit](https://github.com/lindicaphxag-tech/kaggle/commit/ea9cee77dcb15e2eb7a27bacc47293fa2b24f8cc):
  the deliberately added same-center/narrow-vs-wide regressions revealed
  first-fiber-order dependence. [Actual red CI #37750826065](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37750826065)
  reported **4 expected failures and 71 previous tests passing** in the
  Python 3.12 NumPy core job.
- Corrective commit: preserve the conservative union of receptive supports
  for each common center signature.
- Independent execution check: a real PyTorch `AvgPool3d(k=(3,5,1))`
  response changes at a queried center because of an **off-center input
  impulse** even when the identity-path value at that center is zero.
  A dense `Conv3d(2,1,1)` fuses those paths. The type must report RF
  `[3,5,1]`, not `[1,1,1]`.
- Standard tests cover both fiber orders, point-affine center equality,
  dense convolution, and MONAI-style factorized reductions.
- The core public GitHub Actions matrix must pass on the corrected exact
  head before the patch can be merged.

## External recognition and evidence boundary

This is a genuine *first-party* semantic-certificate bug discovered by
adversarial internal auditing; its fix is a model-independent safety proof
for representation metadata. **It is not a third-party MONAI/TorchIO code
defect, a new mathematical theorem, real-data model improvement, clinical
safety certification, nor an externally accepted PR.**

The main benefit is proof integrity: after this change a patient-space
certificate will not silently describe a narrower input-receptive region
solely because channel fibers were enumerated in a different order.
