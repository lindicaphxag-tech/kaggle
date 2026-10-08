# Controller State Is Part of a Robot Policy
## Training-free, stateful semantic transport of frozen manipulation policies

**Reviewer evidence packet — October 2026**
**Status:** Public software research; not accepted or peer reviewed.
**Primary (predeclared) evidence:** PickCube-v1, seeds 21001–21032.
**Secondary (exploratory) evidence:** StackCube-v1 and PushCube-v1, 32 paired episodes each.

### NEW: prospective independent-checkpoint/seed test — complete and audited

The strongest new source-frozen evidence is
[**ManiSkill Actions #37815142810 (SUCCESS)**](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37815142810).
After an execution-only GitHub matrix-expression correction
(commit `9ffa86c6`), **all eight** real PhysX CPU replay chunks plus the
full-denominator audit completed. The study fixed the untrained PPO models,
action translation, null controls, 50-step horizon, 64 source state seeds,
and report/stop gates **before** examining these task outcomes.

[Original preregistration](https://github.com/lindicaphxag-tech/ManiSkill/blob/research/action-abi-independent-pull-stack-replication-v1/research/ACTION_ABI_PULL_STACK_PREREGISTERED_V1.json)
(Git blob `721509118c490f2e9a5a87a6570922b7106bfbea`).
[Original 64-state aggregated ZIP](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37815142810/artifacts/11567460897)
(SHA-256 `2a4701dcdc77f9f4f7cf72137e55172f8a6882001be5019d92b7c5af0c50225c`).
[Permanent 64 individual task-state records](prospective_pull_stack_64.json) |
[Standard-library third-party-friendly independent verifier](audit_prospective_64.py).

| New frozen cohort (32 physical states) | Source PPO | Direct action copy | Strict exact/refuse | Bounded stateless | **Bounded stateful** |
|---|---:|---:|---:|---:|---:|
| PullCube-v1, seeds 51001–51032 | 31 | 14 | 17 | 14 | **31** |
| StackCube-v1, seeds 61001–61032 | 28 | 0 | 11 | 0 | **30** |

- Same frozen source policy/checkpoint and identical seed within a task,
  without policy training or selection after outcome. Different tasks use
  their own **independently published pretrained PPO weights**.
- **Direct mechanism ablation:** comparing the two *bounded-projection*
  arms changes only whether the previous **commanded target pose** is
  correctly carried through the controller ABI.
  On the 32 original PullCube states: **17 stateful-only successes,
  zero stateless-only**; on StackCube: **30 stateful-only, zero stateless-only**.
- **Both preregistered gates passed:** source-policy competence
  31/32 and 28/32 (minimum 24/32), and net memory contrasts
  +17 and +30 (minimum +6), respectively. Strict exact-only
  refuses nonrepresentable commands and achieves 17/32 and 11/32;
  projected commands that leave the exact action domain are labeled
  **NOT_EXACT**.
- All five arms ran for all original states: **64 task-state episodes ×
  five controller arms = 320 controller-arm trials (including exact-mode early refusals)**. No successful
  state was selected in place of failed controller states, and no
  four-way intersection conditioning was applied.
- **Scope correction:** PullCube is a new task relative to earlier
  PickCube/PushCube/StackCube source tests. StackCube is a *new precommitted
  seed cohort*, but StackCube had already been studied exploratorily;
  it is NOT a wholly previously unseen task family. The frozen protocol
  itself is preserved rather than rewritten after this observation.
- This is still **contributor-operated simulation**, not independent
  scientist replication, not a new embodiment family, not unknown
  interface identification, and **not upstream SemRepair adoption**.
  The bounded stateful arm does not prove safety or exactness under clipping.

**Fast review:**
`python research/semrepair/policy-transfer/audit_prospective_64.py --self-test`.
The verifier reads all 64 original per-state records, recomputes successes,
discordant outcomes, predeclared gates, refusals and nonexact projection
provenance, and rejects deliberately altered records without loading any
policy or simulator.

### Research question

Can a *frozen* source manipulation policy survive a change in the target
controller's semantics without retraining, without assuming that a normalized
action tensor is a complete specification of what physical command will be
executed?

The source ManiSkill controller interprets delta end-effector actions
relative to the *achieved* pose. The target controller interprets analogous
actions relative to its *previous commanded target*, an internal state not
encoded in the source policy's action. Directly copying normalized numbers
can therefore apply a different physical command.

### Mechanism

The proposed **stateful controller-ABI transport** (1) verifies and projects
the target observation to the frozen policy's original observation ABI,
including a consistency check against the live 7D target-memory state;
(2) decodes the source policy's normalized action using its production action
chart to obtain a physical end-effector goal; (3) uses the target controller's
*current commanded-target memory* to compute the equivalent desired
target-relative action; (4) checks whether the desired action lies within the
target's bounded native action domain.

**The important decision boundary:** If the action is representable, exact
target transport is allowed. If not, the strict mode **refuses** rather than
mislabeling a clipped command as exact. A separate *projected* mode explicitly
solves a feasible box/unit-ball projection, records every nonexact step and
evaluates downstream behavior. A successful projected action is **not**
semantically identical or necessarily physically safe.

All experimental policy weights remain frozen; no policy fine-tuning or
checkpoint selection based on holdout results is performed.

### Original physical simulator results

Each task uses 32 source-seed-paired episodes (50-step horizon) with identical
frozen pretrained PPO checkpoint across comparison arms. Success is ManiSkill's
actual per-step task success, not converted-demo retention or tensor closeness.

| Task / evidence role | Source | Exact-only transport | Bounded projected transport | Direct action copy | Stateless ablation |
|---|---:|---:|---:|---:|---:|
| PickCube — **predeclared holdout** | 31/32 | 3/32 | **32/32** | 0/32 | — |
| StackCube — exploratory task | 30/32 | 14/32 | **31/32** | 2/32 | — |
| PushCube — exploratory task | 30/32 | 28/32 | **30/32** | 20/32 | 20/32 |

**Paired evidence (projected vs direct copy):**

- PickCube: **32 wins / 0 reverse wins**, exact two-sided paired McNemar
  `p = 4.656612873077393e-10`; absolute sample difference **+100 percentage
  points**. The primary contrast and fixed seeds were declared before this run.
- StackCube: **29 / 0**, `p = 3.725290298461914e-9`, **+90.625 percentage points**.
  Exploratory, not a second predeclared confirmation.
- PushCube: **11 / 1**, `p = 0.00634765625`, **+31.25 percentage points**.
  Exploratory; the difference is not a guarantee for new tasks or robots.

**Mechanism ablations:** In PickCube, exact-only transfer refuses 29 of 32
episodes (3/32 success), while the bounded *nonexact* projected branch achieves
32/32 and reports 34 nonexact action steps. In StackCube, exact-only obtains
14/32 vs projected 31/32, with 21 nonexact steps. In PushCube, live
target-memory transport reaches 30/32 vs 20/32 for a memory-agnostic controller
translation; this contrasts two controllers/algorithms in one exploratory
task only. The strict/approximate split is a tradeoff between representational
fidelity and task completion, **not** a theorem that nonexact projection is safe.

The three rows are not pooled into a new single inferential claim: different
tasks were not all preregistered and share aspects of the same controller
family. Episode-seed paired McNemar tests use the exact discordance counts;
they do not address uncertainty over checkpoint choice, embodiments or
physical hardware. Reported marginal Wilson intervals are descriptive
for each arm, **not** CIs on paired treatment gain.

### Independent check: 32-demo converter × controller interaction

A separate [precommitted 32-source-episode four-cell protocol](https://github.com/lindicaphxag-tech/ManiSkill/blob/validation/dp-peg-factorial-32-holdout/research/kaggle_diffusion_policy_peg/FACTORIAL_32_PROTOCOL.md)
was executed under the official PegInsertionSide-v1 demonstration replay
path: [public successful Actions run #37806000218](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37806000218).
This is *not* frozen-PPO task reward; it measures whether recorded official
demonstrations can be converted and replayed successfully.

| Slice | Baseline | Converter only | Controller only | Both fixes |
|---|---:|---:|---:|---:|
| Development-exposed first 8 | 6/8 | 6/8 | 0/8 | 6/8 |
| Frozen new-four-cell-replay next 24 | **19/24** | **19/24** | **0/24** | **19/24** |
| Overall 32 | **25/32** | **25/32** | **0/32** | **25/32** |

In the second stratum, comparing controller-only to baseline gives 19
discordant source episodes in the baseline's favor, zero in reverse;
the exact two-sided sign-test `p=3.814697265625e-06` is a **descriptive
paired statistic for demonstration conversion**, not robot policy success.
The four-cell interaction in conversion-success fraction is `19/24`.
Converter-only and both-fixes converted precisely the same 19 episodes,
so this sample does **not** establish a separate benefit from applying the
controller patch on top of the converter patch.

Important qualification: episodes 8–31 were unseen in the earlier *four-cell*
eight-demo experiment, but parts of this source dataset have been used in
earlier two-cell studies. They are **not untouched across all prior work**.
The empty four-way success intersection makes a matched four-cell
learned-policy trial from these converted demos **not trainable without
changing the protocol**; it must not be replaced by a favorable
three-cell learner comparison.

[Independent factorial seed-matrix auditor](https://github.com/lindicaphxag-tech/ManiSkill/blob/validation/dp-peg-factorial-32-holdout/research/kaggle_diffusion_policy_peg/audit_factorial_32.py).
All runs are first-party, not maintainer-accepted upstream validation.

### Reproduce and independently falsify

Original first-party native PhysX CPU runs:

- [PickCube frozen holdout — run 37809892720](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37809892720)
- [StackCube exploratory — run 37747596201](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37747596201)
- [PushCube exploratory — run 37810291098](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37810291098)

A standard-library-only audit reads all **96 raw episode outcomes** copied
from original first-party run logs. It checks original commit identity,
checkpoint digest, complete seed order, initial-state matching, strict-refusal
events, nonexact projection events, episode-level outcomes and source summary.
It includes six tampering tests per task (18 total). A separate exact paired
statistics script recomputes treatment/control discordance and its conditional
binomial p-value without SciPy.

```bash
git clone https://github.com/lindicaphxag-tech/kaggle.git
cd kaggle
python research/semrepair/policy-transfer/audit.py
python research/semrepair/policy-transfer/paired_stats.py
```

[Raw public per-episode records](raw/) ·
[one-command audit](audit.py) ·
[exact paired statistics](paired_stats.py)

**High-value third-party falsification:** reproduce even one of the
predeclared PickCube outcomes with independently downloaded frozen weights,
a separately set up ManiSkill environment, and a different machine; or show
that the 7D controller memory is misread, that a transport labeled exact is not
representable, or that source/target initial physical states differ.

### Scientific positioning and competing prior art

Robot coordinate/chart transport, policy representation equivariance,
active system identification and executable policy metadata are already
studied. [GaugeBench (2026)](https://arxiv.org/abs/2610.07597) explicitly
covers changes of physically equivalent robot descriptions and known-gauge
transport. [ExecSpec (2026)](https://arxiv.org/abs/2606.03724) addresses
action-normalization/controller specification semantics.

The *narrower* hypothesis tested here is that **controller commanded-target
memory must be treated as state of the executable policy ABI**, and that
cross-controller transport requires representability tracking and a
distinction between exact authorization, explicit approximation and refusal.
The present implementation reads known source/target controller semantics.
It does **not** discover hidden ABI conventions, prove general safety, or
establish a new abstract optimal diagnosis law.

The separate [two-by-two delta-pose replay check](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885)
also highlights partial-repair non-monotonicity: baseline **6/8** saved demos,
converter-only **6/8**, controller-only **0/8**, combined **6/8**. That result
concerns **demonstration conversion**, not learned-policy success, and belongs
to a different experiment. Do **not** pool its counts with the three frozen
PPO task evaluations above.

### Falsification criteria, gaps and use boundary

Do not promote this artifact as L8/L9 external adoption until there is
authentic outside reproducer feedback or retained upstream use. Known gaps:

1. One public PPO checkpoint per task and one source→target controller pair;
   the three tasks are not three independent embodiments.
2. Only PickCube was independently precommitted as a confirmation cohort;
   the other two tasks were exploratory.
3. The projected solution sacrifices exact equivalence when clipping. It
   must not be deployed as an unconditional safe control adapter.
4. The controller descriptions were known, and this does not demonstrate
   **blinded inference** of unknown interface semantics.
5. All original rollouts and the audit are author-operated. Neither NVIDIA
   nor ManiSkill has accepted SemRepair as a framework, and there is no
   independent third-party reproduction claim.
6. Public repos have no confirmed repository-wide reuse license. Public
   read access does not imply permission to redistribute source.

**Most important next independent check:** a fresh checkpoint, a new
controller-semantics pair and a predeclared unseen cohort, with an outside
runner independently checking source fidelity and both exact-authorization
and approximation/refusal behavior.

For third-party counterexamples and reproducibility reports, use the
[SemRepair public issue #23](https://github.com/lindicaphxag-tech/kaggle/issues/23).
