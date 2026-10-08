# Flagship research candidate — technical brief

**Working title (not a publication claim):**
**Which Interface May Be Repaired? Evidence-Constrained Semantic Diagnosis under Compensating Robot Control Faults**

**Central falsifiable question.** Given an executing robot whose numerically valid
action interfaces may conceal multiple, mutually compensating semantic defects,
which probe or internal boundary observation is *actually necessary* before
authorizing a specific local code repair? When is the correct answer **refuse**?

## Abstract — evidence-aligned draft (not a submitted paper)

An end-to-end robot trajectory may appear correct even when its producer and
controller disagree about the meaning of an action. Correcting one interface
in isolation can then degrade or disable execution. We study repair
authorization when the underlying interface semantics are only partially
observable. Rather than demand identification of every hidden implementation
detail, we seek an admissible sequence of physical probes and internal taps
that leaves one implementation- and evidence-bound repair authority.
Conflicting surviving authorities require refusal. For deterministic finite
observations we solve the resulting adaptive decision problem exactly; for
bounded adversarial measurement error we retain overlapping posterior belief
sets rather than merging transitive interval overlaps. A native MuJoCo
construction exhibits externally identical trajectories under different
producer, actuator and inertia parameters while requiring different local
repairs; access to internal force evidence resolves the ambiguity.
In a pinned ManiSkill PegInsertionSide four-cell experiment, independently
replaying eight official demonstrations yields conversion-success counts
**6, 6, 0 and 6** for baseline, converter-only, controller-only and composed
repairs. This illustrates how an isolated patch may destroy nominal
compatibility, but **does not show an improvement in learned-policy task
success**. Further source-episode replication and independent validation
are required before broader reliability claims.

## What the method contributes

1. **Decision-sufficient authority target.**
   A world is not necessarily fully identified; evidence is sufficient only
   when every possible world licenses the *same concrete patch / implementation
   / evidence digest / dependency digest*. This is adapted from active
   diagnostic decision-region concepts, not presented as invention of optimal
   decision trees.
2. **Evidence-limited physical diagnosis.**
   External trajectory data, internal controller-input signals and actuator
   force are distinct potential experiments with explicit costs and risk
   limits. If the admissible evidence cannot disambiguate conflicting local
   repairs, the solver refuses.
3. **Actual robotics-stack intervention.**
   The controlled native MuJoCo example demonstrates mechanism; the real
   ManiSkill converter×controller source permutations demonstrate an
   observable non-monotone integration consequence. The method is **not yet
   an accepted ManiSkill upstream feature**.

## Evidence and limitations, pinned

| Evidence | Measured result | Exact public source | Claim boundary |
|---|---|---|---|
| Finite authority vs full identity | Expected cost 1 vs 3.5 on the frozen table | [Semantic ABI v0.1](https://github.com/lindicaphxag-tech/Isaac-GR00T/commit/f69254a1e3a621a10236902ac77e9c1b1c7de668) | Designed finite fixture only |
| Robust bounded-observation game | Connected overlap quotients falsely reject; exact two-probe minimax cost 2 | [PR #40](https://github.com/lindicaphxag-tech/kaggle/pull/40) | Known mathematical family; finite adversarial scalar model |
| MuJoCo physical non-identifiability | Externally identical yet different local repairs; internal evidence resolves | [PR #31](https://github.com/lindicaphxag-tech/kaggle/pull/31) | Controlled one-DOF plant, not broad robot performance |
| Official ManiSkill 2×2 conversion replay | 6/8, 6/8, 0/8, 6/8 | [Native CI #37715587885](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885) | Eight development-exposed demos; no policy gain |
| Actual official diffusion-policy smoke | Six paired demonstrations, 893 transitions/arm, two optimizer steps/arm | [Paired CI #37713921020](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37713921020) | Both arms 0% in tiny short-horizon evaluation |
| Initial four-cell replication | A predeclared first-32 source-episode 2×2 run and exact seed audit | [Frozen branch](https://github.com/lindicaphxag-tech/ManiSkill/tree/validation/dp-peg-factorial-32-holdout) | Outcome not asserted until CI complete |

## Three honest scientific concerns to resolve

**Source selection / generalization.** The initial eight demonstrations were
used during debugging. Source episodes 8–31 have not been used in the initial
four-cell experiment, but the first 100 were previously processed in a
two-cell Kaggle study. Therefore episodes 8–31 are a **new four-cell
replication slice**, *not* completely untouched task data or independent
robots. This limitation must accompany any headline result.

**Non-trainable factorial.** If the controller-only cell produces zero
converted trajectories, a matched four-way learned-policy comparison is
undefined. A result from training only surviving cells must never be
relabeled a four-way task-success comparison. The representation-level
interaction can still be a valid and informative system-reliability
finding, provided every denominator is preserved.

**Novelty and external recognition.** [GaugeBench](https://arxiv.org/abs/2610.07597)
already studies physically equivalent re-description and transport between
known conventions. Active diagnosis and noisy decision-region stopping
are prior art. The narrow proposition here is *repair-authority sufficiency
under hidden semantics, evidence provenance and intervention admissibility*.
There is **no confirmed SemRepair maintained upstream adoption, independent
third-party reproduction, accepted top-tier paper or qualified prospective
I2** in the present evidence package.

## Immediate manuscript go/no-go gates

- Freeze and audit 32-case four-cell source-seed matrix with development vs
  new-four-cell-replication strata and exact paired discordance counts.
- At minimum, independently replay the decisive failed local-repair case on
  a second runtime/official task and report *all* negative conversions.
- Evaluate the decision solver on hidden code/specification worlds, with
  the semantic source withheld from the acting diagnostician.
- Separate *demonstration compatibility*, *diagnostic authority correctness*,
  *repair execution closure*, and *trained policy task success* as four
  distinct outcomes. Report meaningful effect sizes and uncertainty; do not
  confuse them.
- Obtain actual third-party reproduction or an upstream maintainer retaining
  the semantics/repair contract. Internal green CI is not that evidence.

**Conclusion:** This is the highest-leverage *candidate flagship* in the
current public robotics evidence. It is a mechanism-plus-real-stack story
with falsifiers, not yet a publication-ready general-purpose self-healing robot.
