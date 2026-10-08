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
- [**Bounded-noise diagnostic correction**](research/semrepair/noisy_authority_game.py):
  `python research/semrepair/noisy_authority_game.py` — the old v0.1
  overlapping-ball quotient can **falsely reject** feasible active diagnosis;
  the new exact minimax belief-set game restores a two-probe solution in a
  frozen four-world counterexample. This is not a false-authorization finding.
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

**Repair-before-fusion — information-loss witness (2026-10-08):** [merged PR #44](https://github.com/lindicaphxag-tech/kaggle/pull/44) executes two genuine, phase-shifted MONAI DiNTS branches followed by a dense `Conv3d(2,1,1)`. Two **different** analytic input worlds yield almost indistinguishable post-mix tensors (max diff `4.77e-7`) but require physical target outputs separated by `0.7` value units. Any post-mix-only estimator has a minimax worst-case absolute-error lower bound `0.35`. The new [`physically_align_then_mix`](research/medical_geometry/pre_mix_compiler.py) repairs each admissible spatial fiber **before** the dense mixer, reducing the controlled reconstruction max error to `4.17e-7`, with a certified target lattice and optional mixer-weight error budgets. [Final five-job successful CI + raw JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37717636413) · [full proof scope](research/medical_geometry/PRE_MIX_IDENTIFIABILITY_CONTRACT.md) · [reproduction](research/medical_geometry/benchmark_monai_nonidentifiable_mixing.py). This is a synthetic information-loss witness with real MONAI/PyTorch operators, **not a clinical segmentation improvement or MONAI defect claim**.
**First non-injected upstream geometry defect (2026-10-08):** On the exact **original** [`ecalabr/brain_mask`](https://github.com/ecalabr/brain_mask) NIfTI inference loader/exporter, supported `data_plane='cor'` and `'sag'` convert spatial data axes but the exporter then reattaches the **unmodified source affine**. [Verified native-source before/after CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715729072) pins the upstream source, genuine acquired T1 MRI and project-provided mask by Git blobs: same-grid Dice **ax 1.000→1.000**, **cor 0.781→1.000** (original centroid drift **5.66 mm**), **sag 0.673→1.000** (original drift **12.38 mm**). [Actual 5-panel real MRI + numeric JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715729072/artifacts/11522779447) · [standard one-file native patch](research/medical_geometry/upstream_repros/brainmask_export_plane.patch) · [**copy-ready maintainer issue**](research/medical_geometry/upstream_repros/BRAINMASK_UPSTREAM_ISSUE_READY.md) · [reproducer and exact code audit](research/medical_geometry/upstream_repros/BRAINMASK_NONAXIAL_EXPORT.md) · [merged author-hosted PR #42](https://github.com/lindicaphxag-tech/kaggle/pull/42).

**Stronger no-stub corroboration:** [merged public PR #43](https://github.com/lindicaphxag-tech/kaggle/pull/43) · [successful native TensorFlow 2.13 CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37716306589) now runs **the original upstream patch extractor, GradientTape-based reconstruction and NIfTI exporter** without replacing reconstruction. The same real MRI-derived noncubic input `48×56×40` becomes an incorrectly labeled `56×40×48` original sagittal output; the native patch restores exact original shape/voxels while axial remains unchanged. The mask remains a **transport sentinel, not a learned prediction**.

**Exact claim boundary:** this uses the original production loader and exporter with the model patch-reconstruction boundary explicitly stubbed to isolate spatial transport; a genuine existing example mask is used as the test signal, **not a learned segmentation output or expert truth**. It is an established nonaxial export-code-path defect, not a prospectively frozen discovery or accepted upstream repair. Direct issue submission to `ecalabr/brain_mask` was attempted but returned GitHub integration **403**; external maintainer reception remains unconfirmed.


**Default MONAI activation/normalization proof (2026-10-08):** [merged PR #41](https://github.com/lindicaphxag-tech/kaggle/pull/41) runs the genuine `FactorizedReduceBlock` with **its default ReLU and InstanceNorm3d** (not identity-replaced). Conditioned on an externally established non-crossing input sign interval and continuous-field Hessian bound, the compiler propagates curvature through ReLU's stable region, actual frozen conv weights and this sample's measured normalization scaling. In the controlled positive quadratic, each fiber's predicted vs observed max interpolation errors were `0.05565005` vs `0.05565012`, and `0.04250343` vs `0.04250368` (float32 tolerances). Crossing-zero sign intervals, unsupported norm modes, or unknown premises fail closed. [Five-job green CI + raw JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37715655720) · [reproduce](research/medical_geometry/benchmark_default_monai_curvature.py). **Does not certify pretrained clinical models or infer input intervals from real scans.**
**Weight-derived model-level repair evidence (2026-10-08):** [merged PR #38](https://github.com/lindicaphxag-tech/kaggle/pull/38) now derives *per-output* physical Hessian bounds **from the live pinned MONAI Conv3d weights**, starting from explicit uniform input-field curvature premises. For audited `FactorizedReduceBlock` branches with controlled weights `+2`/`−3` and input bound `0.02` value units/mm², output bounds are `0.04`/`0.06`, yielding per-fiber interpolation limits `0.38970`/`0.58455`. Model-weight mutation changes the derived bound and SHA-256 provenance digest; unsupported nonlinear activation or normalization fails closed. [Five-job green CI + reproducible JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37713447511) · [direct benchmark](research/medical_geometry/benchmark_weight_derived_monai.py) · [proof obligations and novelty audit](research/medical_geometry/WEIGHT_DERIVED_C2_CONTRACT.md). **No claim of certified default pretrained MONAI networks or clinical Dice improvement.**
**Verified pretrained-medical-model result (2026-10-08):** [Frozen original TensorFlow 2.13 T1 3-D U-Net experiment](research/medical_geometry/PRETRAINED_T1_WITNESS.md) on a genuine anonymized MRI now has [green model CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37713444946), [green core CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37713444893), and [real full-brain-context prediction figure + JSON evidence](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37713444946/artifacts/11523305602). We pin the official upstream HDF5 weights and three model source modules by Git hashes and verify **340 model variables change from initialization**. At frozen threshold 0.5, the actual 64³ prediction contains **136,748 positive voxels**. Controlled stale-affine export gives fixed-grid Dice **0.848042**; restoring the known exact representation map gives **1.0**; native NiBabel independently verifies the affine/index map. [Merged research PR #39](https://github.com/lindicaphxag-tech/kaggle/pull/39).

**Correct evidence boundary:** genuine model and genuine MRI, but controlled injected geometry error. The existing provided mask is *not independently verified expert annotation*; no learned-model accuracy uplift, naturally occurring geometry bug, clinical safety certification or third-party adoption is claimed.


**Second real-data proof (2026-10-08):** [paired anonymized T1 MRI and existing 3-D brain mask](research/medical_geometry/REAL_PAIRED_MASK_WITNESS.md) from `ecalabr/brain_mask` was verified using pinned upstream Git blobs and [green public CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37711455720). [Download actual MRI overlay + full numerical sidecar](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37711455720/artifacts/11521614213). For a controlled fixed-axis reindex with stale metadata: **fixed-grid Dice 0.834866, physical HD95 11.4455 mm, physical centroid drift only 0.7338 mm**; restoring the known exact index-map affine gives **Dice 1.0, HD95 0 mm**. Native NiBabel `as_reoriented()` independently agrees with the affine and voxel map. [Merged PR #36](https://github.com/lindicaphxag-tech/kaggle/pull/36).

**Critical boundaries:** the upstream sample is intentionally pre-masked for anonymity; the included mask is **not verified expert ground truth**. HD95 measures distances between voxel-center surface samples after the real world-affine transform, not subvoxel clinical contour distance. This is a controlled representation error on a *genuine image/mask pair*, not a learned-model performance gain or an upstream bug.


**New nonlinear runtime result (2026-10-08):** [PR #35](https://github.com/lindicaphxag-tech/kaggle/pull/35) extends the [pinned MONAI real forward](https://github.com/lindicaphxag-tech/kaggle/pull/32) from a linear patient-coordinate field to a quadratic field with an **analytically known physical Hessian bound**, and connects that envelope directly to per-channel runtime repair authorization. With 0.7×0.7×5 mm voxel spacing, the naive common-index readout MAE is **1.49385**, whereas physically repaired MAE is **0.19485**; repaired maximum absolute error **0.1948514** attains the sharp conditional **0.19485** interpolation bound within float32 tolerance. Remove a channel's Hessian evidence or tighten the error budget and the repair is refused. [Five-job green CI and JSON artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37711065982) · [direct script](research/medical_geometry/benchmark_monai_dints_quadratic_bound.py). **This is a controlled quadratic scalar field through a real MONAI module, not learned-feature curvature certification, downstream Dice, or clinical efficacy.**
**New actual-image evidence (2026-10-08):** [Real 3-D brain MRI witness](research/medical_geometry/REAL_MRI_WITNESS.md) from NiBabel's pinned `someones_anatomy.nii.gz` has now **run successfully** with [public real-image CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37709285219) and [actual Figure 1 / Figure 3 PNG + JSON artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37709285219/artifacts/11521335416). This is **actual scanned MRI, not an AI-generated image**. The ROI is explicitly a *controlled synthetic experimental marker*, not expert segmentation; the Conv3d/Pool3d figures are real operator outputs, not pretrained inference.

On the same acquired MRI, joint-array reindex voxel Dice stays `1.0`, stale-affine patient-grid Dice is `0.0`, certified repair restores `1.0`, and the marker centroid shifts `71.5 mm` before repair. Actual Conv/Pool operator centers differ by `7.14471 mm` in the example's world coordinates. **Independent NiBabel native `as_reoriented()` check agrees on the affine (error `0.0`) and reindexed voxels.** [Merged public PR #33](https://github.com/lindicaphxag-tech/kaggle/pull/33).

These measurements concern controlled representation mechanics on a real medical image. They do **not** show a lesion error, clinical outcome, trained-model benefit, third-party adoption or independent reproduction.


**New execution evidence (2026-10-08):** [ClinGeom PR #32](https://github.com/lindicaphxag-tech/kaggle/pull/32) runs the **actual pinned MONAI DiNTS `FactorizedReduceBlock`** and rebinds its two output channel fibers to one patient-space target before readout. On a controlled analytic linear field under 0.7×0.7×5 mm spacing, naive shared-index reading has mean absolute field error **6.400000**; typed per-fiber transport has MAE **9.18e-7**, maximum absolute error **3.81e-6**. The structural source phases are **5.0971 mm** apart. The runtime refuses non-point-valued/multi-support fibers and target regions requiring extrapolation. [All-five-jobs-green public CI and JSON artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37708936748) · [reproduce source](research/medical_geometry/benchmark_monai_dints_fiber_repair.py). **Fixed sampling weights, analytic signal, NOT trained segmentation accuracy or clinical performance.**
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
## Lossless medical label semantics (2026-10-08)

[ClinGeom PR #27](https://github.com/lindicaphxag-tech/kaggle/pull/27) fixes a real execution-semantics defect in the public repair operator: **segmentation/instance IDs were previously cast to float32 during nearest resampling**, and high int64 values could collide; continuous float64 features had a mismatched float32 sampling grid. The new integer/bool path gathers original values losslessly using float64 source *coordinates* and explicit nearest-even index ties. Float64 features use matching grid precision.

**Adversarial paired evidence:** On 343 synthetic target voxels with int64 IDs above `2**53`, the original float32 surrogate changed **343/343** label values, while the fixed path changed **0/343**. All **52** PyTorch geometry tests and the full Python-version/MONAI integration CI passed. [Full public CI and downloadable JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37707665910). [Runnable negative control](research/medical_geometry/benchmark_lossless_labels.py) · [typed data-value contract](research/medical_geometry/DTYPE_TRANSPORT_CONTRACT.md).

This demonstrates *categorical value preservation in a synthetic numerical stress test*, not clinical incidence, model accuracy improvement, or independent external adoption.
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
