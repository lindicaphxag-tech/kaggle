# ClinGeom Flagship — Research Decision and Evidence Dossier

**Version:** 2026-10-09, evidence-audited research planning dossier.  
**Status:** engineering research brief for author/adviser evaluation; **not** a peer-reviewed or submission-ready manuscript.  
**Proposed positioning:** *ClinGeom: Proof-Carrying Patient-Space Dependencies and Repair Before Irreversible Fusion in 3-D Medical AI.*

## Executive decision

**One flagship, not a portfolio of shallow variants.** Focus on a single research
question: *When is a spatially shape-compatible medical-model tensor
physically meaningful, and can a compiler repair the representation **before**
an irreversible operation while issuing a reproducible proof obligation?*

This is NOT simply an affine recorder, a benchmark-only contribution or an
appeal to clinical risk. The computational object is

```text
physical patient-space chart + channel-dependent sampling lattices
  + structural dependency/support + operation-specific numerical semantics
                  |
                  v
       certified transport or explicit refusal
                  |
                  v
         downstream fusion/prediction
```

The strongest *candidate* methodological advance is **typed pre-mix transport
of per-channel physical sampling fibers, with operation-aware support and
conditional error proofs**. The word "candidate" matters: independent novelty
assessment and actual task-level improvement have not yet been established.

### Why this is the best near-term ROI

1. There is already a public repository with executable NumPy, PyTorch,
   torch.fx and pinned MONAI tests; rebuilding a baseline would waste time.
2. A genuine pretrained external 3-D medical U-Net checkpoint and a natural
   nonaxial NIfTI export defect are both pinned to upstream commits.
3. The impossibility witness for **post-mix-only recovery** executes on MONAI
   and PyTorch and gives bitwise-identical outputs for physically distinct
   intended values. This cleanly justifies why action must move **before**
   the irreversible mixer rather than appear as a cosmetic post-hoc checker.
4. Work is CPU-executable for geometry and reproducibility; an RTX 3080 Ti
   12 GB is sufficient for *small 3-D crop fine-tuning and frozen inference*.
5. One newly caught correctness problem—global normalization dependencies
   falsely treated as local support—has its own adversarial regression and
   separates structural support from fixed-sample conditional transport.

### Prior work / non-novel elements

- MONAI already represents tensor-level metadata through `MetaTensor` and
  supports transform tracing, spatial resampling and inversion:
  https://github.com/Project-MONAI/MONAI/blob/dev/monai/data/meta_tensor.py
  and https://monai.readthedocs.io/en/latest/transforms.html
- NiBabel already supports mathematically correct voxel orientation
  permutations and their affine transformations:
  https://nipy.org/nibabel/reference/nibabel.orientations.html
- Classical linear algebra already establishes null-space identifiability
  and linear mapping criteria; it must **not** be presented as a new theorem.
- Classical box wrapping, numerical curvature/Lipschitz propagation and
  convolution receptive-field arithmetic are all established prior art.
- Input-dependent normalization has global spatial context. The recent
  fail-closed correction is a **necessary validity fix**, not a novel type
  of neural layer or a new mathematical discovery.
- The pinned MONAI DiNTS FactorizedReduceBlock defaults to an InstanceNorm
  configuration; replacing its nonlinearity/norm with Identity in some
  linear analytic witnesses is explicitly a controlled model variant,
  not a default-network trained-model result:
  https://monai-dev.readthedocs.io/en/stable/_modules/monai/networks/blocks/dints_block.html

## Scientific claims and evidence provenance

Do **not** aggregate these rows into one fabricated "clinical accuracy" score.

| ID | Narrow claim / experiment | Grounded result | Evidence class |
|---|---|---|---|
| E1 | Same genuine pretrained T1 3-D U-Net output, original vs inverse-plane-corrected NIfTI exporter | axial 1.000→1.000; coronal **0.0041172→1.000**; sagittal **0.3630826→1.000** | **real learned output**, controlled comparison of exporter semantics, **not** learned accuracy uplift |
| E2 | MONAI FactorizedReduce + PyTorch dense pointwise mix; two distinct physical inputs with identical post-fusion binary32 representation | bitwise output collision, postmix difference 0, physically intended target separation **0.5** in analytic units; pre-fusion repair errors below numerical tolerance | controlled **analytic** inputs and weights, not trained model |
| E3 | Existing real MRI + project-provided brain mask under controlled stale affine | fixed-grid Dice 0.834866; physical HD95 11.4455 mm; centroid drift 0.7338 mm; exact known-transform recovery Dice 1.0 | genuine image and **unverified existing mask**, controlled representation corruption |
| E4 | Source-native `brain_mask` original nonaxial exporter bug and minimal upstream patch | original-source 8 failing nonaxial tests, patched 17 passing tests on Python 3.10 & 3.12 | original external library, **author-hosted**, not upstream-accepted |
| E5 | Receptive support under normalization and residual fusion | global-statistic nonlocal influence; fixed-statistics inference counterexamples; support union must be order invariant | internal compiler correctness; not external validation |
| E6 | Independent third-party maintainers have adopted ClinGeom fixes | **NOT OBSERVED** | **missing external recognition** |
| E7 | Improvement in segmentation accuracy on annotated held-out patient volumes | **NOT MEASURED** | **missing task-level study** |

**Public evidence:**

- E1: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979
- E2: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718214998
- E3: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37711455720
- E4: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37749615927
- E5: https://github.com/lindicaphxag-tech/kaggle/pull/54 (only cite green exact-head CI once confirmed)

## Technical formulation

Let `A` be the raw input voxel-center to patient-mm chart; a tensor
`X in R[N,C,D,H,W]` carries a family of channel-fiber maps:

`L_g(i) = A [ diag(j_g) i + s_g, 1 ]^T`.

A physical-support annotation `S_g(i)` must enclose **every source voxel
that can affect the value** of fiber `g` at output position `i`,
including nonlocal normalization or statistics when used.

The compiler must enforce THREE separate invariants:

1. **Spatial identity:** if output voxel indices are permuted by exact signed
   transform `H`, the source/world identity requires `A_out = A_in H`.
   Shape parity or equal numeric tensor values cannot substitute for this.
2. **Support soundness:** after residual sum or other aggregation, support is
   the union of **all** contributing branches. GroupNorm and input-statistics
   InstanceNorm/BatchNorm cannot silently inherit a local support.
3. **Pre-irreversibility:** before a spatially incompatible dense channel mixer,
   either synthesize a role-aware, in-domain, explicitly recorded lattice
   alignment or **refuse**. Post-fusion output alone cannot generally recover
   every spatially intended target.

For a particular fixed input sample, an operator like InstanceNorm may be
analyzed conditionally by substituting its *measured* per-channel scale and
offset into a separate post-execution interpolation inequality. That is a
**sample-conditioned field-transport** statement, not a structural proof that
the original model's output lacked global input dependencies.

### Concrete algorithm deliverable

```
input real 3-D MRI, physical affine and tensor graph
  -> torch.fx channel-fiber propagation
  -> track center lattice AND exact/conservative support
  -> recognize unsupported data-dependent/global-statistic assumptions
  -> for each required merge, check exact crop or role-aware resampling
  -> prove per-fiber admissibility + numeric tolerance when premises exist
  -> execute transport before dense fusion
  -> preserve digest bound to source geometry, model weights and proof premises
  -> emit output or REFUSE with a minimal, readable witness
```

**Explicitly forbidden claims:** automatically inferred Hessian bounds for a
learned MRI encoder; local support for default InstanceNorm; clinical
segmentation improvement from the E1 export-fix Dice; "upstream merged"
for author repository PRs; ability to reconstruct arbitrary post-mix signals.

## Candidate 4-figure narrative — all figure assets must be real/reproducible

**Fig. 1: Actual acquired MRI and a real pretrained model output.** Use the
pinned four-panel pretrained T1 evidence and the source-frame correction
results. Annotate source affine, model crop, actual frozen probability map,
original versus corrected NIfTI export. Do not draw artificial anatomy.
Source: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979

**Fig. 2: Real operator graph and bitwise collision.** Diagram the actual
MONAI factorized/phase-shift reduction and dense PyTorch 1x1 mix; link the
exact code and report the bitwise-equality check rather than implying trained
model recovery. Source:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718214998

**Fig. 3: Compiler action with dependency provenance.** Show two incoming
sampling fibers, the recognized support envelope, conditional interpolation
authorization, repair before mixer, and refusal branch. This panel must use
real measured tensors/affines; no invented clinical image.

**Fig. 4: Audit and ablation plot.** Include measured task-level outcome
only after *actual frozen inference on independently annotated data*. Until
then, use a factual red/green regression matrix and label it **software
evidence**, not clinical-model accuracy.

Avoid infographic cards or ornamental model drawings. Work from real NIfTI
slices, native PyTorch graph diagrams, compact axis labels, vector overlays,
and consistent small typography; make SVG/PDF from reproducible source.

## Fastest experiment matrix that would materially upgrade the work

The *only* additions worth prioritizing before submission are:

| Experiment | Baseline(s) | Target comparison | Minimum credible evidence |
|---|---|---|---|
| R1. Real labelled medical volumes | original input-frame export; standard native reorientation; ClinGeom pre-fusion repair | physical-space Dice, HD95, reference-frame preservation | frozen checkpoint; independent annotated set; patient-level count and intervals; no tuning on test |
| R2. Two different architectures / spatial operator families | shape-only; tensor-level affine metadata; channel-fiber centers only; full support-aware compiler | false authorization rate; correct refusal rate; value error; runtime overhead | one MONAI/nnU-Net-family 3-D encoder plus another materially different stack |
| R3. Actual natural wrong-space pipeline | unpatched upstream; minimal native patched; metadata-only workaround | physical export correctness and regression stability | original-source reproducible red/green plus **official** reviewer response or independent reproduction |
| R4. Ablations that identify mechanism | w/o support union; w/o nonlocal-stat gate; post-mix-only; w/o numeric authorization | false certificate count; admissible repair fraction; impact on task metrics | identical data/checkpoints, no post-hoc subset selection |

A one-case synthetic/controlled benchmark cannot satisfy R1–R4. Two separate
public expert-annotated domains, e.g. CT organ and MRI tissue segmentation,
are preferable to many variants of one anonymized example case. Freeze
dataset IDs, patient-level splits, model weights and geometry corruption
policy **before** measuring results. This is a proposal, not a claim these
experiments have run.

## Stop conditions and risk gates

**Stop adding mechanisms** if any core certificate admits global-statistic
locality without proof, if a source data license prohibits release, or if
the real-model improvement cannot beat ordinary native inverse-orientation
handling. A simple correct library call is a fair baseline; the method needs
to demonstrate a *beyond-metadata* case to justify a research paper.

**Falsifiers/reviewer concerns to invite now:**

- `MetaTensor` + MONAI invertible transforms already solve many ordinary
  export errors; demonstrate exactly which multi-support tensor semantics
  they do and don't express.
- A fixed-sample InstanceNorm curvature certificate does not imply the
  network itself has a local receptive field; trace a separate context
  dependency.
- An exact known-transform fix does not claim that an unknown affine can be
  inferred from intensities alone.
- A dense mixer may become practically invertible on a restricted data
  manifold; the current collision only establishes *no universal recovery*.
- Learned-model task performance needs clinical masks and patient-level
  statistics, not duplicate numerical transformations of the same scan.

## Dissemination strategy and ethics

Primary ambition: a **methods** paper for a strong medical-image computing
venue, only after R1/R2 are completed and robustness/no-claim problems are
closed. Candidate venue MICCAI 2027 (submission deadline not yet officially
announced on the society's website as of October 2026). Journal follow-on
should be considered only with broad real data and genuinely independent
external evaluation.

**Important MIDL 2027 policy conflict:** the MIDL 2027 author instructions
restrict LLM use to minor language/style/formatting edits and prohibit LLM
generation of research ideas, study designs, analyses, results or substantive
manuscript text. This AI-assisted dossier must **not** be pasted into an
MIDL submission or presented as solely human-authored scientific work.
https://2027.midl.io/author-instructions

For any submission, the human researchers must independently formulate,
verify, own and write the final scientific contributions, and comply with
the chosen venue's current AI and authorship policies.

## External adoption action

One narrow offer per active maintainer, supported by complete reproducer,
a conventional minimal patch and a native upstream test. **Do not** frame
green author CI as maintainer endorsement. Target active MONAI/TorchIO/
highdicom semantics with a genuinely relevant bug or requested extension,
not mass unsolicited feature PRs.

The current linked GitHub installation can write author-owned repos but
returned HTTP 403 for multiple third-party actions. No official upstream
issue, reviewer acceptance or merge is implied by this dossier.

**Decision rule:** a first-party paper with honest scope and strong held-out
expert-mask experiments is worth more than ten additional author-hosted
patch submissions. The next publication-grade evidence must be a
cross-model/cross-patient result, not another synthetic figure.
