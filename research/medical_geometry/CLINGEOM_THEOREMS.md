# ClinGeom theorem notes — patient-space feature lattices

These statements define the mathematical core currently implemented by ClinGeom.
They are research notes, not yet a peer-reviewed proof artifact.

## T1 — voxel overlap is blind to joint reindexing

For prediction and reference voxel sets `P` and `G`, and any bijection `T` of the voxel
index lattice:

    Dice(T(P), T(G)) = Dice(P, G).

Therefore voxel Dice alone cannot certify that the represented anatomy remains at the
same patient-space coordinates.

## T2 — exact representation-reindex certificate

Let `H` map output voxel indices to source voxel indices for a signed axis
permutation/flip. If `A_in` and `A_out` map their respective voxel indices to patient
coordinates, then the reindex preserves patient-space meaning iff

    A_out = A_in H.

This is an exact, no-interpolation relation.

## T3 — equal tensor shape is insufficient for feature fusion

There exist two convolutional branches with identical tensor output shape and different
patient-space sampling lattices.

Constructive witness along one spatial axis:

- branch A: `Conv3d(kernel=(1,1,3), padding=(0,0,1), stride=1)`;
- branch B: left-pad one cell, then `Conv3d(kernel=(1,1,2), padding=0, stride=1)`.

For an input of length `N`, both outputs have length `N`. Branch A has feature-center
start `0`; branch B has start `-1/2` input cell. Hence elementwise addition or
concatenation is tensor-valid but does not fuse corresponding physical locations.

For 5 mm spacing on that axis, the mismatch is 2.5 mm.

## T4 — corner sufficiency for affine-lattice merge certification

Let two same-shape feature lattices map index vector `i` to patient coordinates

    p(i) = A [i,1]^T
    q(i) = B [i,1]^T.

Their displacement is

    d(i) = ||p(i) - q(i)||_2 = ||(A-B)[i,1]^T||_2.

`d` is a convex function of `i` because it is the norm of an affine function.
Over the continuous axis-aligned box containing the discrete feature grid, a maximum
of a convex function is attained at an extreme point. The extreme points are exactly
the eight spatial corners, and those corners belong to the discrete grid.

Therefore:

    max_{i in voxel grid} d(i) = max_{c in 8 corners} d(c).

ClinGeom can thus compute the exact worst corresponding-cell patient-space displacement
with eight coordinate evaluations, independent of feature-volume size.

## T5 — exact repair class for integer lattice translation

Assume two same-shape lattices have equal jump/spacing and their starts differ by an
integer vector `s` measured in feature cells. Then corresponding physical locations
can be fused without interpolation on the maximal common domain by slicing:

    i_ref = i_candidate + s.

The retained extent on each axis is `N - |s|` when `|s| < N`.

This repair is exact on the retained domain but discards non-overlapping boundary cells.
It must not be described as full-FOV equivalence.

## T6 — fractional lattice translation requires interpolation

Under the same equal-jump assumption, if the start difference divided by jump has any
non-integer component, no pure integer reindex/crop maps all candidate cell centers to
reference cell centers. A full-grid fusion therefore requires interpolation/resampling
or must fail closed.

ClinGeom currently chooses fail-closed certification and emits a `resample` repair class;
it does not silently invent an interpolation policy.

## Falsification boundary

These theorems concern affine sampling lattices. They do not by themselves certify:

- deformable or content-dependent sampling;
- attention with learned continuous offsets;
- interpolation error or aliasing;
- clinical validity of a model;
- semantic equivalence of features with different receptive-field content.

Those require additional contracts rather than extending these claims by analogy.

## T7 — one tensor can require multiple channel-conditioned patient-space lattices

Consider an input lattice with center map

    p(i) = A(i),

and a factorized stride-2 reduction that concatenates two channel groups:

    y_even(i) = f_even(x(2i))
    y_odd(i)  = f_odd(x(2i + 1)),

where `1=(1,1,1)` for the 3-D shifted branch.

The two channel groups share the same output tensor index `i` and output spatial
shape, but their source-center maps are

    p_even(i) = A(2i)
    p_odd(i)  = A(2i + 1).

If the source affine has nonzero spatial linear part, then

    p_odd(i) - p_even(i) = A_linear * 1 != 0.

Therefore no single point-valued affine map from output tensor index to patient
coordinate can be exactly correct for both channel groups.

The correct spatial type is channel-conditioned:

    Tensor -> {(channel range_g, lattice_g)}.

ClinGeom calls this a **LatticeBundle**.

### Sharp lower bound for collapsing a bundle to one affine

At any tensor index, let two channel fibers correspond to patient-space points
`p0` and `p1`, with separation `D = ||p0-p1||`. For any common coordinate `q`
assigned to both groups, the triangle inequality gives

    D <= ||p0-q|| + ||q-p1||

and therefore

    max(||p0-q||, ||q-p1||) >= D/2.

Thus the best possible single-coordinate representation has worst-fiber error
at least half the channel-fiber separation.

For spacing `0.7 x 0.7 x 5.0 mm` and a one-voxel phase shift on all axes:

    D = sqrt(0.7^2 + 0.7^2 + 5.0^2) = 5.0970579749 mm
    single-affine minimax error >= 2.5485289875 mm.

This lower bound is representation-theoretic; it does not depend on trained
weights or segmentation accuracy.

## T8 — dense channel mixing preserves multi-lattice structural support

Let an input tensor have channel fibers with distinct spatial center lattices
`L1,...,Lm`. A groups=1 convolutional output channel has the form

    y_k(i) = sum_c sum_delta w[k,c,delta] x_c(i+delta).

Structurally, before inspecting or assuming exact zero learned weights, each
output channel may depend on every input channel fiber. Its patient-space
support is therefore the union of the spatial supports induced by all incoming
lattices through the convolution.

If at least two incoming center lattices are distinct, ordinary dense channel
mixing does not by itself prove that they have collapsed to one exact
point-sampling lattice. A tensor-level affine may at best be an approximation
or convention unless an additional equivalence proof is supplied.

ClinGeom therefore propagates a finite support set per output channel group
instead of silently erasing phase provenance at a convolution.

### Important boundary

This is a **structural support** statement. A particular trained kernel could
contain exact zeros that remove some dependencies. ClinGeom v1 deliberately
does not use floating-point weight sparsity to certify semantic geometry;
doing so would require weight-bound certificates and robustness to numerical
perturbation.


## T9 — affine prompt representation ceiling for axis-aligned box prompts

Let a source AABB have positive side-length vector `d` and let `A` be the
invertible source-to-target affine linear chart map. The exact transported
physical region is a parallelepiped with volume

    V_exact = |det(A)| prod_j d_j.

The tightest target-axis-aligned enclosing box has widths

    w_i = sum_j |A_ij| d_j

and volume

    V_AABB = prod_i w_i.

Therefore the best physical-region fidelity achievable by an enclosing target
AABB is

    IoU_prompt(A,d) = V_exact / V_AABB.

This equals one exactly for monomial `A` (one non-zero per row and column),
the affine maps under which the AABB family is closed. For a unit cube under
a 45-degree in-plane rotation, `V_AABB=2`, so `IoU_prompt=0.5`.

This is a prompt-representation statement before model inference. It is not an
upper bound on final segmentation Dice/IoU; a model may use image evidence to
refine beyond its prompt.

## T10 — minimal exact representation lift hierarchy

For one canonical source AABB, the transported edge matrix is

    E = A diag(d).

The least expressive exact target representation obeys:

1. **AABB** iff `A` is monomial.
2. **OBB** iff the transported edges are mutually orthogonal, equivalently
   `A^T A` is diagonal for positive diagonal `d`.
3. **Parallelepiped** for every invertible affine `A`.
4. **Target-grid mask** when no supported continuum parametric primitive is
   available and physical point-membership is explicitly rasterized.

The instance-level OBB condition must not be confused with global OBB-family
closure. Every arbitrarily oriented OBB remains an OBB only under similarity
maps satisfying `A^T A = s^2 I`.

A compiler should therefore choose the first *supported exact* primitive in

    AABB -> OBB -> parallelepiped -> mask

before considering any inexact enclosing-box approximation. If an API supports
only AABB, approximation requires an explicit fidelity budget based on T9 or
must fail closed.

### Prompt-theorem falsification boundary

T9/T10 concern physical prompt denotation under affine image-chart changes.
They do not claim generic bounding-box robustness, prompt-selection optimality,
or segmentation-performance bounds.
