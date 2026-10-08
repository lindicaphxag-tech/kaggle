# Public Research Artifacts

[![ClinGeom public core artifact](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml/badge.svg?branch=main)](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/clingeom-core-artifact.yml)

This repository is a public evidence and reproducibility surface for two research lines:

1. **ClinGeom** — typed patient-space semantics for medical AI pipelines.
2. **SemRepair** — authority-optimal identification of hidden embodied interface semantics, independently checkable frozen CPU artifact, real-stack evidence, and timestamped prospective transport.

They are deliberately kept separate in claims: public self-authored artifacts are **not** external adoption, and prospective cases are **not** counted before their frozen criteria are met.


**NEW — ClinGeom, actual pretrained MRI evidence:** [PR #68 (merged)](https://github.com/lindicaphxag-tech/kaggle/pull/68) follows genuine frozen T1 3-D U-Net predictions across three orientations into an unchanged downstream prediction fusion, enforcing the declared patient-grid inverse before mixing. [Real TensorFlow run](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37819406496) and [frozen numeric/provenance dossier](research/medical_geometry/upstream_repros/REAL_PRETRAINED_PLANE_FUSION_RESULT.md) show one-sample **wrong-index vs corrected-grid ensemble** weak-reference Dice **0.0458→0.1700** and HD95 **50.93→40.90 mm**, including six adversarial rejection tests. **One MRI, one checkpoint, nonexpert reference; not a clinical improvement, external adoption or a learned internal-feature fusion claim.** [Independent falsification requested](https://github.com/lindicaphxag-tech/kaggle/issues/9).


## SemRepair — start here

### Representative result: four frozen manipulation policies, stateful controller action semantics

The strongest completed **real closed-loop** evidence is
[**Stateful Action-ABI Transport for Frozen Robot Policies**](https://github.com/lindicaphxag-tech/ManiSkill/blob/main/research/frozen_policy_transfer/STATEFUL_ACTION_ABI_FLAGSHIP.md).
It uses four **different published PPO checkpoints** and official ManiSkill
PhysX CPU task success, not demonstration save rate. Each task has **32 paired
source states** (128 original task states; these are not 128 independent policy
families). Each target arm executes without model retraining.

| Native task | Frozen source | Naive direct copy | Stateful bounded adapter | Matched memory-blind |
|---|---:|---:|---:|---:|
| PickCube | 31/32 | 4/32 | **31/32** | 4/32 |
| PushCube | 29/32 | 21/32 | **29/32** | 20/32 |
| PullCube | 31/32 | 14/32 | **31/32** | 14/32 |
| StackCube | 28/32 | 0/32 | **30/32** | 0/32 |

The controller stores a **previous commanded target pose** that the source
policy does not encode in its action tensor. The adapter reconstructs the
physical source goal, reads the live target-controller memory, and computes
a bounded destination command. **Projection is explicitly `NOT_EXACT`**:
successful task execution is not a safety guarantee or exact policy equivalence.

The strongest *prospectively frozen* extension is PullCube and a new
StackCube seed cohort (64 original task states):
[original PhysX CI + full-source audit](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37815142810),
[per-state records and hash-bound verifier](research/semrepair/policy-transfer/REVIEWER_PACKET.md).
PickCube includes a separate frozen memory-ablation cohort; PushCube is
exploratory. All evidence remains contributor-operated, under **one Panda
controller-family mismatch**. Neither independent outside reproduction nor
another embodiment has been established.

**Why this is different from the method hypothesis:** These real results
support a *known* stateful controller chart translation, not yet automatic
**discovery of unknown** action semantics. The [separate source-bound
PegInsertionSide 2×2 contrast and exact-repair infeasibility evidence](research/semrepair/ROBOT_ACTION_CONTRACT_FLAGSHIP.md)
motivates the next **evidence-conditioned authorization** question. It is
not an additional claim of policy-success gain.

**Flagship research question:** When robot interfaces have hidden or mutually
compensating physical semantics, what is the **minimum admissible evidence**
needed to authorize one concrete implementation-bound repair?

- [**NEW: precommitted 64-state frozen-policy transfer across PullCube and a new StackCube cohort**](research/semrepair/policy-transfer/REVIEWER_PACKET.md):
  [real PhysX CPU run and complete original 64-state audit](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37815142810).
  **Stateful bounded adapter vs identical stateless ablation: PullCube 31/32 vs
  14/32; StackCube 30/32 vs 0/32**. Source-policy competence and the two
  predeclared memory-gain thresholds both pass. Every one of the
  **64 original state episodes × five arms** is archived
  [per-state](research/semrepair/policy-transfer/prospective_pull_stack_64.json)
  and can be re-counted by a [dependency-free auditor](research/semrepair/policy-transfer/audit_prospective_64.py).
  StackCube was previously studied on a different exploratory cohort;
  these are new seeds, **not** a previously unseen task family. All evidence
  remains author-operated, not an independently adopted upstream system.
- [**Flagship reviewer packet: Frozen Robot Policy Transfer — 96 real simulator episodes and a separate 32-demo causal-factor replication**](research/semrepair/policy-transfer/REVIEWER_PACKET.md).
  Directly inspect mechanism, predeclared versus exploratory data, exact
  source-seed-paired statistics, original CI links, and negative results.
  **Known controller semantics; no retraining; no third-party adoption claim.**
- [**Frozen learned-policy controller-ABI transport — 96 physical test episodes and independently re-countable evidence**](research/semrepair/policy-transfer/README.md):
  **PickCube predeclared holdout 32/32 vs naive 0/32**; StackCube
  exploratory **31/32 vs 2/32**; PushCube exploratory **30/32 vs
  20/32**. All use a published frozen PPO and controller-target-memory-aware
  action translation without retraining. The bounded projection is NOT an
  exact semantic equivalence or safety guarantee. One command audits every
  source-seed result with Python stdlib:
  `python research/semrepair/policy-transfer/audit.py`
- [**Flagship technical brief — abstract, actual results, limitations**](research/semrepair/FLAGSHIP_TECHNICAL_BRIEF.md)
- [**Flagship evidence — official 100-source robot replay + frozen PPO 32-seed live controller memory**](research/semrepair/ROBOT_ACTION_CONTRACT_FLAGSHIP.md): real failure, local-repair feasibility and explicit negative evidence
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
- [**Official ManiSkill 2×2 intervention**](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885):
  exact source-tree converter×controller replay **6/8, 6/8, 0/8, 6/8**,
  showing a harmful controller-only intervention and restoration when
  paired. This is **demo replay**, not learned-policy performance.
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

**New highest-rigor numerical artifact (2026-10-09):** [PR #64](https://github.com/lindicaphxag-tech/kaggle/pull/64) and [PR #65](https://github.com/lindicaphxag-tech/kaggle/pull/65) are merged. The [exact declared-chart witness](research/medical_geometry/EXACT_DECLARED_CHART_WITNESS.md) audits rational source indices built from **stored float64 geometry declarations**, and the [actual 3-D fusion witness](research/medical_geometry/EXACT_PREMIX_EXECUTION_WITNESS.md) independently replays **full PyTorch trilinear alignment + real 1×1 Conv3d weights/bias** against an exact rational computation from stored tensor values and parameters. This closes a previously acknowledged gap between ideal geometry proofs and **observed finite-precision execution** on bounded CPU test inputs. [Targeted dual-Python green CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37816724872) · [full core + MONAI green CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37816724817).

**Explicit scope:** one replayed CPU tensor/model snapshot with SHA-256 fingerprints and a finite work budget. No general error guarantee for unobserved patients, model inputs, altered weights, CUDA kernels or clinical performance is asserted. These are author-operated merged artifacts, **not external upstream adoption**.


**Execution-level numerical correctness (2026-10-09):** [merged PR #57](https://github.com/lindicaphxag-tech/kaggle/pull/57) adds an [exact-rational replay witness](research/medical_geometry/executed_interpolation_witness.py) to the existing physically typed feature-repair stack. [Dedicated Python 3.10/3.12 CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37810611794) (**6/6 tests each**) and [full core CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37810611915) both passed. A linear-field example has **real-arithmetic Hessian interpolation bound 0** but an actually executed **FP32 `grid_sample` error > 0**, so mathematical error alone cannot authorize a zero-error claim for finite-precision execution. [Reproducible JSON benchmark](research/medical_geometry/benchmark_executed_interpolation_witness.py) · [carefully bounded proof scope](research/medical_geometry/EXECUTED_NUMERICAL_WITNESS_V1.md). **The exact-rational witness covers stored source float values at computed double source coordinates; it does NOT yet certify rounding of the ideal patient-space coordinate map, underlying Hessian truth, CUDA kernels or clinical performance.** Floating-point certification is an established research area; this is a domain-specific execution-replay contribution, not a new theorem or upstream adoption.


**Single flagship research dossier (2026-10-09):** [evidence-audited scientific positioning, actual model results, precise novelty boundaries, missing clinical experiments and reviewer falsifiers](research/medical_geometry/FLAGSHIP_EVIDENCE_AND_DECISION_2026_10.md). The recommendation is to **stop adding scattered first-party PRs** and complete one proof-carrying patient-space dependency-and-pre-fusion-repair study. **New proof soundness closeout:** [merged PR #54](https://github.com/lindicaphxag-tech/kaggle/pull/54) fixes nonlocal normalization and residual-support provenance while preserving a strictly sample-conditioned MONAI curvature transport path; both [PyTorch nonlocal CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37807815760) and [full cross-version NumPy/PyTorch/MONAI core CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37807815719) passed on the exact merged head. **This does not supply the missing multi-patient expert-annotated task evidence or external maintainer adoption.**


**Highest-leverage learned-model interoperability evidence (2026-10-09):** [merged PR #55](https://github.com/lindicaphxag-tech/kaggle/pull/55) evaluates the **original pretrained 3-D T1 U-Net on one real MRI** under unchanged upstream TensorFlow patch extraction/reconstruction, then scores its original-vs-corrected NIfTI exports against the **upstream-provided nonexpert mask** on the same world-space affine. On the frozen 64³ crop, axial weak-reference Dice is unchanged (0.148993); coronal Dice changes **0.068705 → 0.082254** and HD95 **45.00 → 42.49 mm**; sagittal Dice changes **0.374345 → 0.363077** (a *negative* change), and HD95 **40.80 → 40.76 mm**. **All orientations and negative results are reported**; the mask is not an expert ground truth and no clinical benefit is claimed. Geometry preservation of the *same learned prediction* remains a separate exact contract. [Successful pinned pretrained CI + JSON and image](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37807593490) · [weak-reference HD95 source](research/medical_geometry/upstream_repros/brainmask_pretrained_weakmask_eval.py) · [maintainer-native upstream patch and tests](research/medical_geometry/upstream_repros/brainmask_export_with_native_tests.patch). **External upstream acceptance is still 0.**

**Proof-correctness red/green repair (2026-10-08):** In [merged PR #52](https://github.com/lindicaphxag-tech/kaggle/pull/52), ClinGeom's own same-center support deduplication was shown to **erase wider input receptive fields** depending on channel order. [Original failing CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37750826065) had **4 targeted failures and 71 previous NumPy passes**. [Corrected full green CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37751162239) passes NumPy Python 3.10/3.12/3.13, real PyTorch anisotropic-pooling-plus-dense-fusion regression and pinned MONAI DiNTS runtime. Source types now take a conservative componentwise **union of receptive supports** when centers coincide instead of arbitrarily keeping the first branch. [Proof and exact limitations](research/medical_geometry/SUPPORT_UNION_DEDUP_PROOF.md). This is a first-party methodological correctness fix, **not a third-party MONAI merge or a claim of clinical improvement**.


**Semantic-certificate provenance fix (2026-10-08):** [merged PR #49](https://github.com/lindicaphxag-tech/kaggle/pull/49) closes a case where **identical voxel tensors, conv weights and voxel-grid repair plans** had one old provenance digest despite different patient-mm charts. On the public executable witness, 1×1×1 mm versus 1×1×5 mm yields different authorized feature-value error bounds (`0.015` vs `0.135`, a **9× difference**), while legacy v2 still reports an identical digest; v3 binds physical affine, FrameOfReferenceUID, curvature premises and output bounds and distinguishes them. [Five-job green CI and before/after JSON](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37749517414) · [implementation contract](research/medical_geometry/SEMANTIC_CERTIFICATE_V3.md). **This is a first-party semantic fingerprint fix, not a SHA-256 break, authenticated patient identity, or external adoption.**
**New upstream-maintainer submission bundle (2026-10-08):** [merged public PR #50](https://github.com/lindicaphxag-tech/kaggle/pull/50) packages the real `brain_mask` spatial-export fix plus **native regression tests** into [one standard Git patch](research/medical_geometry/upstream_repros/brainmask_export_with_native_tests.patch). [Exact-original-RED / patched-GREEN public CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37749615927) succeeds on Python 3.10 and 3.12: **8 original nonaxial failures followed by 17 passing patched tests** per version. Tests are only NumPy/NiBabel/pytest, not ClinGeom or TensorFlow. [Maintainer-ready issue](research/medical_geometry/upstream_repros/BRAINMASK_UPSTREAM_ISSUE_READY.md) has the compact reproduction command and original full-TensorFlow / real-pretrained-model corroboration. **No official upstream acceptance has occurred; original project issue submission remains blocked by 403.**


**Physical-support certificate correction:** [merged ClinGeom #48](https://github.com/lindicaphxag-tech/kaggle/pull/48) fixes `physically_align_then_mix` erroneously copying a caller's nominal receptive field after trilinear interpolation. In the pinned real MONAI DiNTS → `Conv3d` forward, the caller nominal `(1,1,1)` input-voxel footprint underreports the conservative actual mixed support `(4,4,4)`. The result now carries the computed envelope in its spatial type **and** in the provenance digest; tests include native anisotropic pooling, fractional-neighbor support and invalid-metadata refusal. [All five public CI jobs green](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37746557233) · [derivation and claim limits](research/medical_geometry/RECEPTIVE_SUPPORT_ENVELOPE_V1.md). **Support enclosure is not equivalence of learned features or clinical effect.**

**External review, 3 reproducible experiments:** [ClinGeom maintainer/independent-replication one-pager](research/medical_geometry/EXTERNAL_REVIEW_PACKET.md) — real highdicom and pretrained MRI export witnesses, finite-precision pre-mix counterexample, full CI and falsification criteria. No external acceptance is claimed.

**Full original pretrained-model chain proves a natural nonaxial export defect (2026-10-08):** [merged public PR #46](https://github.com/lindicaphxag-tech/kaggle/pull/46) executes the original pretrained T1 3-D U-Net checkpoint (**340 weight variables changed**), actual native upstream loader/z-score/TF patch extraction/reconstruction and original vs fixed NIfTI export on a real pinned MRI. For the **same view's learned prediction**, Dice after mapping to its correct original patient grid is **ax 1.000→1.000, cor 0.004117→1.000, sag 0.363083→1.000** after the [one-file inverse-plane patch](research/medical_geometry/upstream_repros/brainmask_export_plane.patch). [Successful exact-head model CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979) · [successful five-job core CI](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977915) · [**actual six-panel acquired MRI + trained output image / JSON**](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718977979/artifacts/11524838113) · [maintainer-ready issue](research/medical_geometry/upstream_repros/BRAINMASK_UPSTREAM_ISSUE_READY.md).

**Scope:** genuine checkpoint and acquired MRI, natural nonaxial exporter code defect, not a naturally wrong prediction caused by the network or a clinical improvement. Changing model view changes the model's own prediction; comparisons use each view's own original prediction as the reference. This is author-operated evidence, with no upstream maintainer acceptance yet.


**Finite-precision exact fusion ambiguity (2026-10-08):** [merged PR #45](https://github.com/lindicaphxag-tech/kaggle/pull/45) strengthens [repair-before-fusion #44](https://github.com/lindicaphxag-tech/kaggle/pull/44) with a **bitwise IEEE-754 float32 collision** in the real pinned MONAI DiNTS + PyTorch `1×1×1` dense-fusion path. Two different controlled inputs produce **exactly identical post-fusion bit patterns** (maximum difference `0.0`) yet require co-located target outputs differing by `0.5`. Any deterministic post-fusion-only method has a worst-case error at least `0.25` on these worlds, while [`physically_align_then_mix`](research/medical_geometry/pre_mix_compiler.py) reconstructs the nonzero target to max absolute error `5.96e-8`. [Five-job green CI / exact JSON artifact](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37718214998) · [runnable witness](research/medical_geometry/benchmark_monai_bitwise_identical_fusion.py). **Constructed fields and controlled weights; no pretrained or clinical result.**
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
