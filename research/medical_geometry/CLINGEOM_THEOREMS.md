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


## T11 — intermediate AABB projection creates path-dependent approximation debt

Let invertible affine linear maps `A1,...,Ak` be applied in order to a source
AABB with positive side lengths `d`.

If exact geometry is retained until the final boundary, its tight AABB widths are

    d_direct = |Ak ... A1| d.

If the region is replaced by its tight AABB after every intermediate stage,

    d_eager = |Ak| ... |A1| d.

The elementwise triangle inequality

    |AB| <= |A||B|

implies `d_direct <= d_eager` componentwise. Hence eager intermediate AABB
projection cannot improve final AABB representation fidelity relative to one
deferred projection after exact transport.

### Exact round-trip witness

For a unit cube, a 45-degree in-plane rotation `R`, then `R^-1`,

    R^-1 R = I.

Deferred projection therefore returns the exact unit AABB with prompt-region
IoU `1`. But eager projection gives widths

    |R^-1| |R| (1,1,1)^T = (2,2,1)^T,

volume `4`, and prompt-region IoU `1/4`.

Thus an exactly cancelling physical transform pair can create 0.75 avoidable
prompt-region IoU loss purely from intermediate restriction of the
representation. The compiler consequence is to retain exact richer geometry
through internal chart transforms and project to a restricted API primitive
only at the final boundary.

This is a representation statement, not a prediction that model Dice changes
by the same amount.


## T12 — dimensionless prompt wrapping debt and refinement monotonicity

For one fixed physical transform path with exact final map

    A = Ak ... A1,

define

    V_direct = prod(|A| d)

for one deferred final AABB projection and

    V_eager = prod(|Ak| ... |A1| d)

for eager AABB projection after every stage.

ClinGeom records the dimensionless prompt wrapping debt

    delta_path = log(V_eager / V_direct) >= 0.

Because exact transported volume is identical for both representations,

    IoU_eager / IoU_direct = V_direct / V_eager = exp(-delta_path).

Thus exp(-delta_path) is the exact fraction of direct AABB prompt fidelity
retained after intermediate AABB restrictions.

### Monotonicity under adding projection boundaries

Suppose one coarse stage B A is split into two stages A then B. The
elementwise inequality

    |B A| <= |B| |A|

shows that replacing a coarse projection boundary by two finer projection
boundaries cannot decrease the enclosing widths. Repeating this argument gives:

> refining a transform path with additional intermediate AABB projections
> cannot reduce delta_path and cannot increase fidelity retention.

The public tests check this property over deterministic randomized invertible
3-D affine maps.

### Prior-art boundary: classical wrapping effect

Repeated axis-aligned enclosure of transformed sets is a classical wrapping
effect in interval/set propagation. ClinGeom does not claim that phenomenon,
the absolute-matrix inequality, or box-volume inflation as new mathematics.

The medical-AI contribution is narrower:

1. interpret the loss as prompt-language semantic debt in patient space;
2. bind it to a typed prompt representation contract;
3. use the certificate to decide whether approximation is permitted;
4. prefer an exact richer primitive (OBB -> parallelepiped -> mask) when the
   target model API supports one;
5. defer lossy projection to the final model boundary when internal transforms
   can retain exact geometry.

The debt certificate is therefore an execution-semantic decision object, not a
new interval-arithmetic theorem.


## T13 — exact completeness and minimality of point-affine channel bundles

Assume each channel fiber g of one feature tensor is point-lattice representable
by one affine patient-space center map F_g(i), with a common tensor spatial shape.

Then a single global patient-space affine F is exact for the entire tensor if
and only if

    F_g = F

for every channel fiber g.

The forward direction is immediate: if one global F exactly represents every
channel, each fiber map must equal it. The reverse direction is constructive:
if all fiber maps are identical, that common map is an exact tensor-level affine.

Therefore the number

    r_patient = | { F_g } |

of distinct patient-space affine-map equivalence classes is the minimum number
of distinct point-lattice semantic types required by the tensor.

ClinGeom additionally requires channel fibers to be contiguous ranges. For the
ordered type-id sequence c_1,...,c_m, the minimum number of contiguous fibers is

    m_min = 1 + sum_{j=1}^{m-1} [c_j != c_{j+1}].

Hence semantic type count and contiguous storage complexity are different.
For example, the channel order A/B/A has two distinct patient-lattice types but
requires three contiguous fibers.

### Strong boundary: multi-support channels

If any output channel structurally depends on more than one distinct patient-space
support lattice, then no partition of channels into point-valued affine fibers
is exact, regardless of how many channel groups are introduced. This is the
case after dense mixing of distinct phase fibers in T8.

The semantic type must then be lifted from a point-affine fiber to a richer
support-set representation rather than increasing the number of point-affine
fibers indefinitely.

### Executable certificate

`certify_minimal_point_affine_partition` returns:

- whether a point-affine partition exists at all;
- the number of distinct patient-lattice affine types;
- the minimum number of channel-contiguous fibers;
- the ordered affine-type ids;
- whether one global affine is exact.

The public tests include homogeneous, factorized-reduction, A/B/A and dense-mix
witnesses.

### Prior-art boundary: polyphase sampling and shift equivariance

ClinGeom does **not** claim discovery of polyphase components, phase-sensitive
downsampling, or the fact that ordinary stride breaks shift equivariance.
Adaptive Polyphase Sampling (APS; Chaman & Dokmanić, CVPR 2021) explicitly
enumerates/selects polyphase sampling grids to obtain shift invariance, and
Learnable Polyphase Sampling (LPS; Rojas-Gomez et al., NeurIPS 2022) learns
polyphase down/up-sampling rules for shift-invariant/equivariant networks.

T7/T13 make a different claim. They treat channel-dependent sampling phases as
**patient-space semantic types** inside an otherwise ordinary medical feature
tensor and ask:

1. whether one tensor-level patient-space affine is exact;
2. the minimum number of distinct patient-lattice affine types;
3. the minimum contiguous channel-fiber representation under fixed channel order;
4. when dense mixing produces a multi-support channel for which *no*
   point-affine channel partition is exact.

Thus the target contribution is not a new anti-aliasing or shift-equivariant
architecture. It is an executable spatial type/minimality system for auditing
existing medical-AI graphs whose downstream code may otherwise assign one
patient-space affine to a tensor that contains multiple sampling phases.

References:
- Chaman & Dokmanić, *Truly Shift-Invariant Convolutional Neural Networks*, CVPR 2021.
- Rojas-Gomez et al., *Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks*, NeurIPS 2022.

### Falsification boundary

T13 assumes affine center maps and a fixed numerical equivalence tolerance for
the executable certificate. It does not claim that every learned feature can be
semantically reduced to a point sample; multi-support channels are explicitly
outside that family and trigger semantic lifting.


## T14 — exact minimax collapse radius for parallel patient lattices

Let a finite family of patient-space feature-lattice affines share one linear
part L and differ only by translations t_1,...,t_m:

    F_g(i) = L i + t_g.

Approximate the entire family by one affine with the same linear part and a
single translation q:

    F_q(i) = L i + q.

Because the linear parts cancel, the worst patient-space center error for
fiber g is independent of feature index:

    ||F_g(i) - F_q(i)|| = ||t_g - q||.

Therefore the optimal single-affine minimax error is exactly

    r* = min_q max_g ||t_g - q||,

which is the radius of the Euclidean minimum enclosing ball of the translation
vectors {t_g}. Its center q* gives the optimal approximate affine translation.

This upgrades the two-fiber T7 bound. For m=2, the minimum enclosing ball is
the midpoint and

    r* = ||t_1 - t_2|| / 2.

For the eight unit-cube phase offsets {0,1}^3,

    q* = (1/2, 1/2, 1/2)
    r* = sqrt(3) / 2.

### Compiler consequence

A bundle of parallel patient lattices may be collapsed to one approximate affine
under a declared tolerance epsilon **iff**

    r* <= epsilon.

Otherwise the compiler must preserve the LatticeBundle (or lift to a richer
support representation). This turns an informal 'close enough' affine choice
into an exact millimetre-valued authorization certificate.

### Relation to the diameter bound

The familiar diameter lower bound

    diam({t_g}) / 2 <= r*

is exact for two fibers but can be non-tight for more general point sets.
ClinGeom reports both the diameter/2 lower bound and the exact minimum-enclosing-
ball radius.

### Prior-art boundary

Minimum enclosing balls are classical computational geometry, and ClinGeom does
not claim that optimization problem as new. The contribution target is narrower:
identify polyphase / channel-conditioned feature lattices in patient coordinates
and use the exact enclosing-ball radius as a semantic type-collapse certificate
for medical-AI feature tensors.

The executable implementation enumerates boundary sets of at most four
translations, sufficient in three dimensions, and is intended for the small
support cardinalities carried by one feature tensor.

### Falsification boundary

T14 requires equal patient-space linear parts. If support lattices differ in
jump/direction/scale, the parallel-lattice theorem is inapplicable and the
certificate must refuse rather than report a misleading radius.

## T14 — conditional physical interpolation error for feature-lattice repair

A geometry-correct interpolation is **not** a proof of functional feature
recovery. Let f be one twice continuously differentiable scalar patient-space
field sampled at a regular source feature lattice. Denote the three physical
source-cell edge vectors by v_j. For a target feature center whose coordinates
inside one source cell have fractions alpha_j in [0,1], tensor-product linear
interpolation I satisfies

    |f(x) - If(x)| <= (1/2) sum_j alpha_j (1-alpha_j) M_j,

provided a *uniform, externally justified* bound

    |v_j^T Hessian(f)(z) v_j| <= M_j

holds throughout all source cells used by the interpolation. Repeated 1D
linear-interpolation error bounds prove the inequality; no mixed-derivative
term is needed because tensor-product linear interpolation reproduces
multilinear cross terms exactly.

If a conservative world-space spectral Hessian norm bound B is known, then

    M_j = B ||v_j||²

is sufficient. The source physical edge is obtained from the patient affine
and model-derived lattice jump:

    v_j = L(A_patient) * diag(jump_source) * e_j.

Since ClinGeom v1 feature-lattice maps are axis-separable, the maximum of
the stated bound across the complete target lattice is computed without
enumerating 3D voxels: find the largest alpha_j(1-alpha_j) among target
indices on each axis and sum the three maxima.

### Sharp quadratic witness

For f(p) = (B/2)||p||² on a 2 mm isotropic source grid, targets at half
a cell on every axis satisfy alpha_j=1/2, v_j=2 mm, M_j=4B.
For B=0.2 in output-value units/mm², each axis contributes 0.1
and the maximum error is exactly

    (1/2) * 3 * (1/4) * (4 * 0.2) = 0.3.

Both a NumPy eight-corner interpolator and the real PyTorch grid_sample
executor have regression witnesses for this equality, up to documented
floating-point tolerance.

### Critical assumption and falsification boundary

- This is a **classical conditional interpolation bound**, not a new
  approximation-theory theorem.
- A finite set of feature samples cannot establish a global Hessian bound.
  For example f(i)=0 and f(i)=sin(pi*i) agree at integer sample nodes but
  differ at half-integer locations. Missing curvature evidence **fails closed**
  when a value-error budget is requested.
- The bound applies only to interpolating *one declared underlying scalar
  field*. It does not prove that distinct learned branches represent the same
  feature field, share receptive-field content, or preserve segmentation Dice.
- Exact integer reindex/crop requires no smoothness assumption and has zero
  interpolation error **on the retained domain**, not full-FOV equivalence.
- Nearest-neighbor interpolation of class labels is not authorized by a
  C² scalar-field bound.
- The physical Hessian envelope is user-provided and must be externally
  justified over **every touched cell**; the compiler does not infer it from
  a neural-network output or certify its provenance.

### Compiler consequence

A request for a maximum allowed value-error budget adds a separate
interpolation-authorization gate to the existing spatial type checker.
Each resampled FX input branch must provide an explicit uniform physical
curvature envelope, otherwise the rewrite is refused. Successful geometry
postconditions and conditional interpolation-error bounds are reported
**separately**, never conflated into a clinical safety guarantee.
