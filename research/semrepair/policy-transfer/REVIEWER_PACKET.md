# Controller State Is Part of a Robot Policy
## Training-free, stateful semantic transport of frozen manipulation policies

**Reviewer evidence packet — October 2026**
**Status:** Public software research; not accepted or peer reviewed.
**Primary (predeclared) evidence:** PickCube-v1, seeds 21001–21032.
**Secondary (exploratory) evidence:** StackCube-v1 and PushCube-v1, 32 paired episodes each.

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
