# Public Research Artifacts

[![ClinGeom public core artifact](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml/badge.svg?branch=main)](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml)

This repository is a public evidence and reproducibility surface for two research lines:

1. **ClinGeom** — typed patient-space semantics for medical AI pipelines.
2. **SemRepair** — authority-optimal identification of hidden embodied interface semantics, independently checkable frozen CPU artifact, real-stack evidence, and timestamped prospective transport.

They are deliberately kept separate in claims: public self-authored artifacts are **not** external adoption, and prospective cases are **not** counted before their frozen criteria are met.

## SemRepair — start here

**Flagship research question:** When robot interfaces have hidden or mutually
compensating physical semantics, what is the **minimum admissible evidence**
needed to authorize one concrete implementation-bound repair?

- [**One-page reviewer / falsification packet**](research/semrepair/README.md)
- [**Cross-repository CPU verifier**](research/semrepair/reproduce_crossrepo.py):
  `python research/semrepair/reproduce_crossrepo.py`
- [Frozen Semantic ABI source](https://github.com/lindicaphxag-tech/Isaac-GR00T/commit/f69254a1e3a621a10236902ac77e9c1b1c7de668)
  · [origin CI](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37471167749)
  · [external-hosted cross-repo CI](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/semrepair-crossrepo-reproduction.yml)
- [Real ManiSkill regression PR #1495](https://github.com/mani-skill/ManiSkill/pull/1495)
  · [current-head 13 / 18-test validation](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37687749068)
  · [GR00T PR #786](https://github.com/NVIDIA/Isaac-GR00T/pull/786)

The frozen finite counterexample requires expected diagnostic cost **3.5**
to identify the entire hidden ABI, but **1.0** to identify a single authorized
repair class. A compensated fault requiring a different local repair must still
be rejected under external-only observations. Lean proves the structural
stopping boundary, not arbitrary robot safety.

**Claims:** No maintained external SemRepair adoption, confirmed prospective I2,
or independent third-party SemRepair reproduction is asserted. This repo's
cross-repository CI is still author-operated, not independent endorsement.

---

## ClinGeom — start here

**Independent reviewers wanted:** [5-minute highdicom #454 before/after reproduction task](https://github.com/lindicaphxag-tech/kaggle/issues/28) — fixed upstream SHA, public passing author-operated CI, exact success/falsification criteria, and credit for an independent report. **Open request; no external reproduction yet.**

Frozen evidence pack: [`research/medical_geometry/EVIDENCE_V0.2.0.md`](research/medical_geometry/EVIDENCE_V0.2.0.md) · machine-readable manifest: [`EVIDENCE_MANIFEST_V0.2.0.json`](research/medical_geometry/EVIDENCE_MANIFEST_V0.2.0.json)

**Frozen public version:** [`release/clingeom-v0.2.0`](https://github.com/lindicaphxag-tech/kaggle/tree/release/clingeom-v0.2.0) at `7e570b05071ae65dc1abc636701c5a75ca1424a3` (public core CI green; 54 NumPy-core tests). Use the frozen branch for independent reproduction; `main` is the continuing research line.

ClinGeom asks a narrow question with broad consequences:

> When a medical image, prompt, annotation, feature tensor or deformation field changes representation, does it still denote the same patient-space object?

The public core currently contains:

- exact patient-space reindex certificates;
- voxel-Dice-perfect / patient-space-wrong counterexamples;
- model-derived feature lattices and channel-fibered lattice bundles;
- T13 minimality certificate: exact patient-lattice type count, minimum contiguous channel-fiber count, and explicit detection when no point-affine partition can represent a multi-support channel;
- T14 exact collapse-radius certificate: optimal millimetre-valued single-affine approximation for parallel patient lattices;
- PyTorch / `torch.fx` spatial-semantics propagation and merge checking;
- constructive common-lattice repair synthesis;
- prompt representation-closure certificates;
- exact AABB → OBB → parallelepiped → target-grid-mask lifting;
- a pinned MONAI DiNTS runtime witness used as an architecture-level motivating case, **not** as a MONAI bug claim.

### 3-minute NumPy reproduction

```bash
python -m pip install numpy pytest
python -m pytest -q \
  research/medical_geometry/test_clingeom.py \
  research/medical_geometry/test_lattice_bundle.py \
  research/medical_geometry/test_repair_synthesis.py \
  research/medical_geometry/test_prompt_representation.py
PYTHONPATH=. python research/medical_geometry/benchmark_clingeom.py
PYTHONPATH=. python research/medical_geometry/benchmark_prompt_representation.py
```

The first benchmark should expose a case where voxel Dice is exactly `1.0` while patient-space centroid error is non-zero, plus a negative control where shape-only geometry is valid. The prompt benchmark must keep analytic-vs-sampled representation error below its frozen threshold. It also includes the exact 45° rotation → inverse-rotation witness: deferred projection returns prompt IoU `1.0`, while eager intermediate AABB projection returns `0.25`.

### PyTorch compiler reproduction

```bash
python -m pip install numpy pytest
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

See [`CLINGEOM_PUBLIC_ARTIFACT.md`](CLINGEOM_PUBLIC_ARTIFACT.md), [`REPRODUCE_CLINGEOM.md`](REPRODUCE_CLINGEOM.md), and [`research/medical_geometry/CLINGEOM_THEOREMS.md`](research/medical_geometry/CLINGEOM_THEOREMS.md) for the full claim boundary.

## Three inspectable external-review entry points

These are public, pinned, executable research artifacts—not third-party adoption claims.

1. **Real medical-library interoperability witness:** [highdicom #454 / MONAI physical-frame identity](research/medical_geometry/upstream_repros/HIGHDICOM_454_RESULT.md), [minimal upstream patch proposal](research/medical_geometry/upstream_repros/HIGHDICOM_454_PATCH_PROPOSAL.md), [reproducer](research/medical_geometry/upstream_repros/highdicom_454_frame_identity.py). On the pinned open highdicom PR head, voxel-to-world affine round-trips (max error `0.0`) but automatic DICOM FrameOfReferenceUID does not; explicit caller-provided UID does. [Public successful run](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37476416622).
2. **Adversarial certificate repair:** [PR #16](https://github.com/lindicaphxag-tech/kaggle/pull/16) rejects even a *freshly signed* resampling plan when it extrapolates beyond source feature support; the PyTorch executor independently checks the signed source shape and valid sampling domain. [Full green CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37702733728).
3. **Post-rewrite executable proof obligation:** [PR #20](https://github.com/lindicaphxag-tech/kaggle/pull/20) independently re-types the *actual rewritten* `torch.fx` graph and requires zero patient-space fusion obligations. Tests deliberately corrupt the repaired phase/axis mapping and check fail-closed behavior. [Full green CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37703217555).

To run the two focused correctness regressions:

```bash
python -m pytest -q research/medical_geometry/test_repair_certificate.py
python -m pytest -q research/medical_geometry/test_torch_repair_certificate.py research/medical_geometry/test_fx_postcondition.py
```

**Evidence boundary:** these are first-party reproducibility artifacts and a pinned observation on an external codebase. highdicom maintainer confirmation, independent reproduction, external adoption, clinical performance improvement, and prospective new-defect status remain separate, unclaimed milestones.
## New physical-value and upstream compatibility evidence (2026-10-08)

**ClinGeom conditional value repair:** [PR #24](https://github.com/lindicaphxag-tech/kaggle/pull/24) extends the previously verified patient-space lattice repair with a **separate conditional feature-interpolation error budget**. Without a valid uniform world-space Hessian envelope for a resampled branch, a requested value-error guarantee fails closed. A quadratic 2 mm patient-space witness attains the stated 0.3-value-unit bound, including the real PyTorch interpolation executor. This uses classical approximation theory; no pretrained-model or clinical robustness claim is made. [Full public CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37706126491).

**highdicom upstream review-ready paired proof:** [PR #25](https://github.com/lindicaphxag-tech/kaggle/pull/25) replays the same pinned highdicom #454 installation **before and after** a two-site FrameOfReferenceUID patch, with unchanged reproduction criteria. Before: two automatic UIDs are lost. After: both UIDs preserved, explicit arguments override metadata, missing legacy metadata stays backward-compatible, and LPS geometry still round-trips with affine error `0.0`. [Paired passing CI and downloadable artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37706342122). See [minimal patch script](research/medical_geometry/upstream_repros/patch_highdicom_454.py) and [post-patch regression](research/medical_geometry/upstream_repros/highdicom_454_patch_regression.py). **This is author-operated validation, not upstream highdicom adoption.**

**Direct upstream-ready source diff:** [`highdicom_454_frame_uid.patch`](research/medical_geometry/upstream_repros/highdicom_454_frame_uid.patch) is now on public `main`, with a [two-site application guide](research/medical_geometry/upstream_repros/HIGHDICOM_454_UPSTREAM_PATCH.md) and [successful exact-commit, bytewise-equivalence CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37707822713). It cleanly applies to highdicom #454 head `4524bb0...` and produces byte-identical code to the patch already used in the paired runtime reproduction. **There is still no highdicom maintainer adoption or upstream PR.**

**Reuse-license decision is open:** [Issue #30](https://github.com/lindicaphxag-tech/kaggle/issues/30). This public repository currently lacks an explicit license; external redistribution/derivative adoption must not be presumed.
## SemRepair prospective evidence transport

The `transport/` surface is intentionally narrower: it freezes first-seen public issue evidence, prediction digests, deadlines and later resolution evidence. It does **not** award itself research credit or retroactively convert known cases into prospective discoveries.

## Evidence status

- ClinGeom public method artifact: **available**.
- ClinGeom external adoption / independent reproduction: **not yet claimed**.
- ClinGeom prospective external mechanism hit: **not yet claimed**.
- SemRepair prospective transport: **active public evidence surface**.

## Contributing / independent reproduction

Independent reproductions are welcome. Please report exact commit, Python/framework versions, commands, full outputs, and negative results. A reproduction only counts as independent evidence when it is performed from public artifacts without private thresholds or hidden expected outputs.

## License

No repository-wide open-source license has been declared yet. Do not infer permissions beyond GitHub's repository access terms until an explicit license is added.
