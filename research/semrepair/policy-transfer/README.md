# Stateful Controller-ABI Transport of Frozen Robot Policies

**A concrete embodied-AI research representative** · fixed pretrained PPO · native ManiSkill/PhysX CPU · no policy retraining

> Can a pretrained robot policy remain effective after a controller changes
> from *achieved-end-effector-relative* actions to *previous-target-relative*
> actions, even though the normalized numeric action has the same shape?

## Start with the reviewer packet and paired statistics

- [**Single reviewer-facing evidence / novelty / limitations packet**](REVIEWER_PACKET.md)
- [**Exact paired statistics, Wilson armwise intervals and source hashes**](paired_stats.py)
- Reproduce with `python research/semrepair/policy-transfer/audit.py` and
  `python research/semrepair/policy-transfer/paired_stats.py`.
- **Predeclared PickCube holdout:** projected controller-ABI transport
  **32/32** vs naive action copying **0/32**; **32 favorable and 0 unfavorable**
  source-seed-discordant episodes. Exact two-sided paired McNemar
  `p=4.656612873077393e-10`. This is deterministic, fixed-checkpoint,
  *author-operated* task evidence, not a measure of across-robot variability.
- **Exploratory cross-task observations:** StackCube 31/32 vs 2/32
  (29 favorable, 0 unfavorable); PushCube 30/32 vs 20/32 (11 favorable,
  1 unfavorable). Neither was prospectively predeclared as confirmatory.

The 2×2 [official delta-pose demonstration replay](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885)
is a **different** experiment: baseline 6/8, converter-only 6/8,
controller-only 0/8, combined 6/8. It demonstrates a non-monotonic
controller/converter interaction in *demo replay*, **not** a learned-policy
success gain. No figures from this 2×2 are pooled with frozen PPO evaluation.

## Research question and mechanism

A 6D normalized delta action is **not an executable semantic contract**.
In ManiSkill's `pd_ee_delta_pose` controller, an action is relative to the
current achieved end-effector pose. In `pd_ee_target_delta_pose`, the analogous
action is relative to the *previous commanded controller target*, a persistent
internal state. Passing the same numbers to the new controller can therefore
change the physical command even when tensor shape/range still look valid.

The examined implementation uses the target controller's live previous-target
pose and the source controller's actual production action decoding to
translate the source policy's requested absolute end-effector goal into the
target-relative native action chart. It also projects the target
controller's extra **7D target-memory observation** away while independently
verifying that those seven values match live controller state, preserving all
task observation features.

A distinct **exact-only** branch refuses action authorization when the
required native target command is outside its bounded position/rotation
domain. A third **projected** branch uses bounded action projection
(box projection for position, unit-ball projection for rotation), records
every non-exact command and tests the resulting *task behavior*.
Successful bounded projection is **not** proof of exact action equivalence.

The negative control copies the source action numbers unchanged to the target
controller. All arms use the **same frozen pretrained policy**, identical
environment seed, genuinely initialized source and target controllers and
50-step official task success definitions.

The interface translation was designed using known source and target
controller implementations. This study does **not** establish active
discovery of previously unknown controller semantics.

## Original physical-simulation results

| Task | Cohort role | Source policy | Exact-only memory-aware | Bounded projected memory-aware | Naive action copy |
|---|---|---:|---:|---:|---:|
| **PickCube-v1** | **Predeclared 21001–21032 holdout** | 31/32 | 3/32 | **32/32** | 0/32 |
| **StackCube-v1** | Exploratory 50001–50032 | 30/32 | 14/32 | **31/32** | 2/32 |
| **PushCube-v1** | Exploratory 40001–40032 | 30/32 | 28/32 | **30/32** | 20/32 |

All values are **episode-level closed-loop task successes** from original
native simulator runs, not demonstration replay success, offline action-chart
error or training-set performance.

For PushCube, an additional **stateless controller-translation ablation**
successfully completes **20/32**, compared with **30/32** when the live
previous-target state is used. This supports the importance of controller
memory in that particular system, without establishing general necessity in
other robots.

The PickCube 21001–21032 cohort was [frozen before execution](https://github.com/lindicaphxag-tech/ManiSkill/blob/validation/heldout-ppo-memory-projected-21001-21032/research/FROZEN_PPO_MEMORY_HOLDOUT_21001_21032.md);
the development 20001–20032 results are not counted as confirmation.
The StackCube/PushCube runs are **exploratory cross-task reproductions**, not
prospectively preregistered confirmatory tests.

## Independently recompute every original episode

```bash
git clone https://github.com/lindicaphxag-tech/kaggle.git
cd kaggle
python research/semrepair/policy-transfer/audit.py
```

The **standard-library-only** auditor reads 96 records extracted from the
original three public GitHub Actions job logs, never just the asserted summary.
It checks:

- the exact task, cohort status, source commit and frozen-policy SHA-256
  identifiers used by the original implementations;
- the full, contiguous, nonduplicated original 32 seeds per task;
- success counts calculated afresh from `success_once` for **every arm**;
- paired `projected` vs `naive`, `exact-only`, and `source` discordances;
- refusals, per-step explicitly non-exact bounded projections, and horizon;
- initial source-vs-target physical/task observation agreement;
- the episode-by-episode records agreeing with the original console summary;
- six adversarial mutations **per task** (18 tampered fixtures) must fail.

[Original PickCube holdout run #37809892720](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37809892720) ·
[Original StackCube run #37747596201](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37747596201) ·
[Original PushCube run #37810291098](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37810291098)

Each compact JSON file in [`raw/`](raw/) includes the source-run and
job identifiers, exact source commit / original implementation blob, frozen
checkpoint identity, per-seed refusals, projection events, physical
initial-state checks and success flags. **No checkpoint weights or personal
data are redistributed.** Reviewers can compare against the original
public GitHub Actions logs.

This auditor was independently implemented **relative to the original runner**
but is still **contributor-operated**, not a third-party replication.

## Scientific distinction

The **stateful controller target memory** is a separate part of the policy
execution contract. A policy that sees only the achieved physical pose can
be paired with an actuator/controller that still holds a different internal
commanded target. A conversion based only on the current achieved pose or
on the action tensor cannot in general reconstruct the next command exactly.

An implementation-bound translation must track both the target state's
previous command and whether the desired source goal lies in the target
controller's admissible action set. This exposes a useful separation:

- **semantic transport:** reproduce the requested physical goal *when
  representable*;
- **bounded projection:** deliberately choose a feasible but *not exactly
  equivalent* target action and report loss of exactness;
- **repair authorization:** refuse rather than certify equivalence when
  necessary evidence or bounded representability is absent.

The task-success comparisons show why these distinctions matter for an
actual learned policy. They **do not** establish universal safe repair or a
general optimal controller for nonrepresentable target actions.

## Fatal neighbors, limitations and next scientific gate

Coordinate/sign/action semantic mismatch and known-frame exact transport
are established prior art (including
[GaugeBench](https://arxiv.org/abs/2610.07597)); active system identification
and decision-class active diagnosis are also established.
The contribution hypothesis here is narrower: **controller-memory-bound
compilation of a frozen learned policy, with explicit
representability/approximation evidence and fail-closed authority**, not the
invention of action coordinates or gauge invariance.

Limitations that remain:

1. These source/target controllers were known from production code; there is
   **no blinded hidden-contract discovery** yet.
2. Bounded projection changes the intended action. A 32/32 task outcome is
   **not** a physical safety certificate or guarantee of equivalence.
3. There is one publicly published PPO checkpoint per task, one target
   controller family, short 50-step horizons and no physical hardware.
4. StackCube and PushCube are exploratory. They do not substitute for
   independently precommitted new task/checkpoint holdouts.
5. The archived author-operated source-run logs and separately authored audit
   are **not independent external adoption**. NVIDIA/ManiSkill maintainers
   have not accepted this abstract repair framework.

**Most consequential next step:** a blinded, source-frozen holdout across a
*new controller semantic identity*, with a third-party runner or maintainer
reproducing the exact checkpoint-and-episode protocol, and negative results
preserved. This would test genuine research transfer beyond a known pair.

Please report reproduced results **or counterexamples** at the
[public SemRepair falsification issue](https://github.com/lindicaphxag-tech/kaggle/issues/23).
A single authentic independent replication or maintenance adoption is
more valuable than additional author-generated CI counters.
