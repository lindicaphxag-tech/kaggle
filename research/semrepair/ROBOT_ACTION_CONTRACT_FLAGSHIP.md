# Repair the interface, not just the tensor
## Two physical counterexamples to unqualified robot-action compatibility

**Status: public reviewer evidence note · 2026-10-09 · no claimed conference acceptance or upstream adoption**

**Research question.** A frozen robot policy can send a correctly shaped action to a controller while its physical meaning is wrong. When is a repair *actually authorized*? We argue that an adapter must be bound to the **producer chart, active controller chart, stateful controller memory, representability limits and source identity**—not inferred from tensor shape or final replay success alone.

This note deliberately separates **an observed engineering failure**, **a successful but non-exact physical adapter**, and **a proposed research method**. The first two are supported by public executable logs. The proposed active, evidence-conditioned repair-authority method still needs independent evaluation.

### 1. Real ManiSkill defect interaction: repairing a controller alone can make behavior worse

Interventions on ManiSkill PegInsertionSide official trajectory replay are a frozen converter patch [#1495](https://github.com/mani-skill/ManiSkill/pull/1495) and a separate controller sign change [#1472](https://github.com/mani-skill/ManiSkill/pull/1472). The original demonstrations were independently re-executed under **four different production source configurations**.

| Original demonstration cohort | Unmodified | Converter-only | Controller-only | Both changes |
|---|---:|---:|---:|---:|
| Development source indices 0–31 | 25/32 | 25/32 | **0/32** | 25/32 |
| Originally preregistered source indices 100–199 | 94/100 | 95/100 | **0/100** | 95/100 |

Primary evidence: [native 32-source run](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37806000218), [corrected native 100-source replay](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37752565886), and the [separate real-HDF5/source audit](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37807568126).

The 100–199 holdout uses disjoint **source episode indices**, but some original *seeds* reoccur across cohorts; it is not 100 statistically independent new seeds. The earlier purported 100–199 result that accidentally reused the first HDF5 groups was rejected and withdrawn. The independent audit above is still **contributor-operated**, not third-party execution.

**What is established:** changing only the controller is drastically incompatible with *these* preexisting delta-pose demonstrations, while including the converter's active-chart translation restores baseline-level replay success. The converter-only outcome differs from baseline by only **1/100** in the corrected 100-source holdout; this is **not** a strong performance improvement. A factorial interaction contrast of `(95−0)−(95−94)=94/100` is an observation about *conversion-survival dependency*, **not** a 94-point learned-policy success gain.

The controller-only arm's zero-survivor status means a **four-way common-success training set is empty**. Treating missing failed conversions as absent observations would introduce outcome-dependent selection bias.

### 2. A second, stateful semantic mismatch on a real frozen PPO policy

**Larger fixed 48-seed physical execution:** [public run #37809684134](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37809684134), with source episode IDs 20001–20048, recorded:

| Frozen policy/controller intervention | PickCube success / 48 |
|---|---:|
| Source checkpoint, original achieved-relative controller | **47** |
| Naive action-copy to previous-target-relative controller | **6** |
| Exact memory-aware inverse adapter, refuses out-of-range target-native command | **8** |
| Bounded approximate projection using real controller target memory | **48** |

**40/48** exact-adapter episodes refused because at least one requested native target action exceeded its representable domain. The projected version logged **43 non-exact intervention steps**. This is a separate physical-run observation, not a guarantee that projection maintains exact semantics, safety or zero controller deviation.

A published third-party PPO checkpoint was **SHA-256 verified, loaded and frozen**—no policy retraining. In actual ManiSkill PickCube simulation, the source controller interprets actions as **achieved-pose-relative delta**; the destination controller uses **previous-target-relative delta**, so conversion requires live controller target memory and a representability test.

The complementary [public 32-seed **memory-blind ablation**](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37812437614) uses source IDs 22001–22032 (disjoint from the 48-seed IDs) and reports:

| Controller path | Observed PickCube successes / 32 | Semantic interpretation |
|---|---:|---|
| Original frozen policy/controller | **31** | source reference |
| Naively copy policy action to target controller | **4** | wrong interface contract |
| Ignore previous-target memory but attempt translation | **4** | memory ablation |
| Memory-aware **exact** adapter, refuse nonrepresentable actions | **9** | 23/32 episodes encounter refusal |
| Memory-aware **bounded projection** | **31** | non-exact approximate action in multiple episodes |

The projection uses the live target memory and clips out-of-domain commands to admissible target-native bounds. It can preserve task success, **but it is not exact policy-equivalent transport**. The log records **27 approximate command steps** across the 32-seed bank; refusal occurs when the normalized target action requires amplitude above 1. The gate is a geometric representability condition, not a heuristic safety claim.

**Key distinction:** the exact adapter's 9/32 successes do not mean that preserving exact semantics is intrinsically worse. The other 23 episodes were **deliberately refused** under an impossible exact mapping. The projected controller's 31/32 successes cannot be relabeled as exact authority or safety-certified repairs.

### 3. One concrete research target: *evidence-conditioned repair authority*

A representative method would maintain a set of hidden controller-world hypotheses and authorize only when all admissible survivors lead to the *same exact repair object*:

```text
input: frozen policy identity, controller identity, candidate semantic charts
       physical observations, internal controller taps, probe/risk budget
loop:
    if the requested action is not representable in target chart:
        return REFUSE_EXACT   # or explicitly route to separately labeled APPROXIMATE
    if surviving hidden worlds authorize a different implementation-bound repair:
        choose a risk-admissible distinguishing probe/tap; if none, REFUSE_AMBIGUOUS
    else:
        bind source/controller/dependency evidence and apply that repair
        verify live effect against an independent physical observation
```

This is a **proposed integration**, not a demonstrated new learned recovery policy. Known coordinate transport, active diagnosis, robust constrained projection and system identification are established prior art. Our candidate novelty must be in **when hidden source-dependent physical contracts are identifiable enough to authorize a specific repair**, without false precision about the original policy or the permitted approximation.

An initial [bounded-noise active diagnosis correction](noisy_authority_game.py) removes false rejection caused by transitive noise-overlap grouping. A [native MuJoCo physical alias/tap experiment](mujoco_native_authority.py) is a controlled identifiability witness. Neither alone proves the complete algorithm on the frozen PPO bank.

### 4. Decisive blinded experiment required for a flagship paper

**Freeze before evaluation:** a second checkpoint/policy family; previously unseen chart/memory perturbations generated outside the diagnosis implementation; source/disjoint initial-state seeds; available physical probes; risk budget; and exact controller/calibration hashes.

Compare **shape-only**, **fixed sign/known transport**, **generic active diagnosis**, **exact-only with refusal**, **approximate projection with a distinct authorization flag**, and **evidence-conditioned adaptive repair**. Measure separately:

1. wrong repair authorization / refusal correctness;
2. probe count and physical intervention cost;
3. actual frozen-policy task success and completed episodes;
4. exact semantic fidelity vs approximate command deviation;
5. OOD/controller-nuisance/generalization failures.

**Kill criteria:** if generic active diagnosis or known chart translation attains the same wrong-authorization rate and intervention cost; if the proposed algorithm depends on hidden source metadata unavailable at runtime; if exact repair is almost always infeasible; or if another pretrained policy does not replicate the mechanism, do not claim a top-tier general solution.

### How an outside reviewer can verify or falsify this evidence

The [short maintainer decision packet](https://github.com/lindicaphxag-tech/ManiSkill/blob/handoff/pr1495-maintainer-decision/validation/PR1495_MAINTAINER_DECISION.md) lists exact **one-commit/two-file** upstream #1495, compatibility tests **13/18 passing**, official source-HDF5 auditors and corrected holdout provenance. The [PPO execution artifact](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37812437614/artifacts/11564969770) includes per-episode refusal/approximation records. An independent counterexample proving a **wrong concrete repair authorization** or showing a falsely claimed exact adapter is executable is especially valuable.

**Strict credit:** both upstream PRs are still open as last checked; these experiments are contributor-operated; there is **zero confirmed independent SemRepair replication, zero qualified prospective I2 and zero maintained upstream adoption of the research abstraction**. An author-created CI artifact is not a third-party endorsement.

**Suggested research title:** *When Can a Robot Repair an Interface? Evidence-Conditioned Authorization Under Compensating and Stateful Action Semantics.*
