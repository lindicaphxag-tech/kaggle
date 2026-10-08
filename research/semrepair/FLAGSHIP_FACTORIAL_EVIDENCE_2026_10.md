# When One Robot Repair Breaks Another: A Frozen 2×2 Causal Compatibility Witness

**Reviewer brief · 9 October 2026 · CPU reproducible · status: author-operated public evidence, not upstream adoption**

> **Main result.** In an official ManiSkill `PegInsertionSide-v1` action-conversion
> replay, introducing a controller-contract repair *in isolation* nearly
> abolishes recoverable demonstrations. Applying the converter-contract repair
> as well restores replay convertibility. This is evidence of a concrete
> **cross-module semantic compatibility dependency**, **not** an increase in
> learned-policy task success.

## Question and intervention

Physical controllers, trajectory converters, and data producers can implement
individually plausible repairs that are not safely compositional with the
existing stack. How should a robot system **identify the *joint* authority of
an executable repair** rather than authorize components one at a time?

We ran four exact software interventions, each independently replaying the
**same original public motion-planning demonstrations** using the official
ManiSkill environment and `pd_ee_delta_pose` controller:

| Converter | Controller | Group | Abbreviation |
|---|---|---|---|
| original | original | upstream baseline | B |
| #1495 repair | original | converter-only | C |
| original | #1472 repair | controller-only | K |
| #1495 repair | #1472 repair | composed repair | CK |

The recorded outcome is **binary trajectory replay/save success per original
source episode**, not the success of a learned policy. Each group's original
source episode is counted whether or not the converted trajectory survives.
There is no post-treatment filtering in these replay denominators.

## Frozen exact-source result

<details>
<summary>Evidence and immutable source coordinates</summary>

- Official dataset: `haosulab/ManiSkill_Demonstrations`, revision
  `d674485bbffdd533914e52d272fdda34c0515608`,
  `demos/PegInsertionSide-v1.zip` with SHA-256
  `7d61e4319a0395b220574f1e26ea65bd4ad1406387fb3debfbea96a2ddbb6a9c`.
- Baseline: `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`.
- Current #1495: `69facfaafaa0ef233d36ef19e6cd9a0f03532ee0`.
- Exact #1472 overlay: `eed9be164797d41540421bda8adb3840377d7087`.
- Native solver: official `PegInsertionSide-v1`, `physx_cpu`, state-only,
  Mesa software Vulkan for initialization (disabled image rendering), fixed
  original demo ZIP, exact arm-specific source checkouts and source-seed ledger.
- These are *two separate original-episode index cohorts* from one official
  archive. The cohort of indices 100–199 was locked before the enlarged
  0–99 result was inspected. The initial 0–7 discovery overlaps the 0–99
  exploratory cohort and cannot be counted as a third holdout.

</details>

| Original source episode indices | B | C | K | CK | Interpretation |
|---|---:|---:|---:|---:|---|
| 0–7 (exploratory discovery) | 6/8 | 6/8 | 0/8 | 6/8 | motivating observation |
| 0–99 (enlarged exploratory cohort) | **90/100** | **91/100** | **1/100** | **91/100** | independently ledger-audited |
| 100–199 (corrected, HDF5-bound holdout) | **94/100** | **95/100** | **0/100** | **95/100** | separate original episodes; audited source provenance |
| Holdout rows whose episode seed does not occur in 0–99 | **71/76** | **71/76** | **0/76** | **71/76** | *post-hoc* seed-disjoint sensitivity |

**Do not add the 8-case discovery to the 0–99 results.** More importantly,
**do not label 0–199 as 200 independent random seeds**: the independently
verified cross-cohort audit found **24 duplicated `episode_seed` identifiers**
between the two 100-episode cohorts (176 unique seed IDs across 200 episode
records). Distinct original episode *indices* do not guarantee independent
stochastic trials.

### Actual effect sizes on the original-source denominator

For indices 0–99, the observed replay interaction
`CK - C - K + B` is
`(91 - 91 - 1 + 90)/100 = +0.89` (89 percentage points).

For corrected indices 100–199, it is
`(95 - 95 - 0 + 94)/100 = +0.94` (94 percentage points).
This latter cohort meets the **frozen >20-point thresholds** for controller-only
penalty and composed-vs-controller-only recovery.

For the 76-source seed-disjoint holdout sensitivity analysis, the observed
pattern is `B=C=CK=71/76`, `K=0/76`.
That subset was selected by **seed identity only, not outcome**, but the
sensitivity analysis was devised *after inspecting* the cross-cohort seed
overlap. No preregistered significance claim is made.

These are finite-cohort **software-intervention contrasts**, not estimates of
out-of-distribution robot task-success improvement or a universal
noncompositionality theorem.

## Why the negative cell matters

An **individually correct-looking controller change** can fail when paired
with the *unchanged* converter's coordinate/action semantics. The restored
`CK` replay then indicates that the converter/controller interface is an
inseparable **repair-authority bundle** for this source/task/control mode.

It does **not** follow that the converter modification by itself improves
learned-policy performance: `C` is nearly identical to baseline in conversion
success, and no properly trained four-cell policy comparison has passed.

The all-four-cell successful-source intersection is **0** in the two
100-episode results. A purported four-arm policy experiment that silently
filters to that common intersection cannot be executed. Dropping the failed
controller-only cell, copying another cell's replayed dataset, or assigning
its missing model a zero task-success rate would each change the estimand or
fabricate a learned-policy result.

## Reproduction and third-party falsification

The following are **successful public CI runs**, not outside endorsements.

| Evidence | Public run |
|---|---|
| Official 8-source four-arm first observation | [#37715587885](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885) |
| Frozen 0–99 replay | [#37719545972](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37719545972) |
| Independent-implementation 0–99 per-seed/sha audit | [#37751515316](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37751515316) |
| Source-index-locked 100–199 original run (historical, superseded by more exact HDF5-source audit) | [#37745942944](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37745942944) |
| **Corrected 100–199 source-bound replay** | [#37752565886](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37752565886) |
| **Physical HDF5 source, duplicated-seed and corrected holdout independent-source-code audit** | [#37807994333](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37807994333) |
| Independent 32-source follow-up (first eight vs next 24) | [#37806000218](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37806000218) |
| Separate two-arm end-to-end CPU Diffusion Policy *training smoke* | [#37713921020](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37713921020) |

The source-bound audit rejects mismatched source-seed matrix, changed
intervention SHA, altered outcome, wrong original source HDF5 ancestry and
mismatched original source archive.

**Specific outside falsification:** reproduce the four-arm 2×2 on a distinct
controller mode, ManiSkill version, or robot task, retaining all source
episodes even when replay conversion fails. Identify a counterexample where
the controller-alone collapse does not occur, or where `CK` fails to restore
replay. File a concrete reproduction with source HEADs, original trajectory
indices and raw logs at the
[public reproducibility issue](https://github.com/lindicaphxag-tech/kaggle/issues/23).

## How this belongs in the original SemRepair research

The [SemRepair method packet](README.md) poses an independent question:
**How much admissible evidence is needed to authorize one exact repair
bundle in hidden semantic interface worlds?**

This factorial observation demonstrates *why* the authority must cover the
specific **converter + controller pair**, instead of individual repairs.
The [finite exact-rational oracle](independent_oracle.py), [bounded-error
active diagnosis correction](noisy_authority_game.py), and [native MuJoCo
structural alias experiment](mujoco_native_authority.py) are different
method-level demonstrations. **They have not yet been shown to predict or
prevent the ManiSkill four-cell failure prospectively**; doing that under a
frozen, blinded intervention protocol is still the central missing scientific
experiment.

## Contribution and adoption boundary

- This is a **strong, reproducible causal engineering witness** about one
  source/target controller conversion, not a new universal robotics law.
- Active diagnosis, representation/gauge transport and software action
  conventions have established prior art. See
  [the fatal-neighbor audit](GAUGEBENCH_NOVELTY_AUDIT_2026_10.md).
- Neither #1495 nor #786 has been shown in this packet to be merged upstream;
  upstream maintainer retention remains unestablished.
- The separate paired **two-update** Diffusion Policy smoke yielded **0%**
  tiny-horizon success in both B and CK. It validates the runnable
  CPU pipeline, **not task-level improvement**.
- A stronger publication claim requires a held-out hidden-contract
  **prediction of the repair-authority bundle**, prospective risk-aware
  probe/tap choice, and a demonstration that this avoids wrong repairs
  **against appropriate baselines in previously unseen tasks**.

**Research-facing title suggestion:** *When Local Robot Repairs Are Not
Composable: Identifying the Evidence Needed to Authorize Semantic Patches.*
This is a working title, not a submitted or accepted paper.
