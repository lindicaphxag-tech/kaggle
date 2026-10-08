# SemRepair — Independent Reviewer Entry

**Research question:** How much admissible evidence must a robot collect before it may authorize a concrete repair when its tensor interfaces are numerically valid but physical semantics are hidden or mutually compensating?

**Primary method:** *Authority-Optimal Identification of Hidden Semantic Interfaces.*

This is a **public, first-party reproducibility artifact**, not a paper acceptance, independent third-party reproduction, or maintained upstream adoption.

## Reproduce the decisive finite counterexample

On this public repository, run:

```bash
python research/semrepair/reproduce_crossrepo.py
```

Dependencies: Python 3.10+ and Git. No PyTorch, checkpoints, CUDA, simulator, robot, or private data.

### Independent exact mathematical oracle

In addition to checking the upstream program's reported JSON, the public repo
now contains [`independent_oracle.py`](independent_oracle.py): a **separately
implemented exhaustive adaptive-partition dynamic program using exact rational
arithmetic** (`fractions.Fraction`), with no imported SemRepair algorithms.

```bash
python research/semrepair/independent_oracle.py
```

It independently derives `7/2` vs `1` expected diagnostic cost from the
frozen experiment table, proves that external-only compensated faults are
unidentifiable in this finite experiment library, and computes the cost `2`
of admitting the internal producer tap. A risk budget excluding that tap
restores the unidentifiable state. A constant repair-authority map stops at
exactly zero probes.

The cross-repository verifier now **requires its output to agree with the
independent exact solver**. This validates a finite, deterministic,
uniform-prior witness, **not** general stochastic identifiability or safety.

The runner checks out the exact immutable commit
[`f69254a1`](https://github.com/lindicaphxag-tech/Isaac-GR00T/commit/f69254a1e3a621a10236902ac77e9c1b1c7de668)
from the publicly frozen `semantic-abi-v0.1` branch of
[`lindicaphxag-tech/Isaac-GR00T`](https://github.com/lindicaphxag-tech/Isaac-GR00T).
It checks the source SHA, executes the artifact's own CPU experiment, and **independently checks the reported JSON** with a distinct verifier in this repository.

```text
full hidden-identity expected cost        3.5
repair-authority expected cost           1.0
authority evidence saving                2.5
externally aliased conflicting repairs   unsafe_unidentifiable
same case with internal producer tap     optimal (cost 2.0)
implementation/dependency drift          splits authority
identical authority among worlds         0 probes required
```

To test that the cross-repository checker refuses corrupted evidence:

```bash
python research/semrepair/reproduce_crossrepo.py --self-test
```

The self-test deliberately tampers with numeric cost, refusal status, implementation/dependency authority, verifier result, and floating-point finiteness.

[Cross-repository CI](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/semrepair-crossrepo-reproduction.yml) repeats this on Python 3.10 / 3.12 / 3.13, separate from the [origin artifact CI run 37471167749](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37471167749). Both are still author-operated CI; this is **not** a new independent third-party replication count.

## October 2026 novelty audit

The closely related [GaugeBench paper (2026-10-06)](https://arxiv.org/abs/2610.07597)
already studies physically equivalent robot-description changes and exact
two-description policy transport. [ExecSpec (2026-06)](https://arxiv.org/abs/2606.03724)
already treats action unnormalization and controller convention as an
executable-policy specification concern. Those results are **prior art, not
SemRepair inventions**.

The narrower research hypothesis is **active identification of unknown
semantic contracts up to a concrete repair-authority class**, with
risk-bounded probe/tap evidence and mandatory refusal under
unidentifiability. See the
[claim restrictions and blinded-experiment kill criteria](GAUGEBENCH_NOVELTY_AUDIT_2026_10.md).
The existing native MuJoCo case is a controlled structural witness, **not yet
the blinded cross-stack test required to establish a new general method**.

## Why it matters

A robot pipeline can contain two locally incorrect transformations whose errors cancel at the end. An output-only evaluator may see the same physical result before and after both defects, even though repairing just one component makes execution worse.

SemRepair formalizes the distinction between:

- identifying every hidden semantic cause; and
- collecting enough admissible probe/tap evidence to identify **one exact authorized repair bundle**.

The stopping condition is:

[
\forall h_i,h_j \in L,\quad A(h_i)=A(h_j).
]

Here `A` must identify **the concrete repair bundle, implementation identity, evidence digest, and dependency digest**. Equal net end-to-end behavior alone does not imply equal repair authority.

When two surviving hypotheses require different repairs and cannot be separated by admissible probes or internal taps, the correct output is **refusal**, not a speculative repair.

This is a narrow embodied-systems contribution. We do **not** claim invention of active diagnosis, equivalence-class determination, value of information, fault masking, general proof-carrying actions, or executable policy metadata.

## Bounded-error counterexample and a corrected active diagnosis game

The pinned Semantic ABI v0.1 solver's nonzero-`observation_atol`
branch conservatively *coalesces the connected components* of overlapping
observation balls. This does **not** introduce false authorization, but it
can create **false unidentifiability** and therefore does not solve the
original worst-case observation game optimally.

[Exact-rational counterexample and minimax correction](noisy_authority_game.py):

```bash
python research/semrepair/noisy_authority_game.py
```

Four hidden worlds with distinct concrete repair authorities each produce
a scalar observation interval of radius `1` per admissible probe. Two
different probes have overlap graphs that are **each connected**, so naive
transitive merging collapses each to one non-informative outcome. Yet for
any actual observation of probe A, the surviving worlds form at most a
two-world candidate set; the second probe separates every such candidate
pair. An exact closed-interval cell decomposition and worst-case
belief-set dynamic program verifies **minimum worst-case diagnostic cost 2**,
with no unsafe authority leaves.

Unlike a deterministic outcome *partition*, set-valued noisy observations
produce **overlapping successor belief sets**. The new solver keeps these
successors distinct and minimizes cost against every admissible adversarial
observation. It preserves closed-interval touching endpoints; it refuses
diagnosis if risk rules prohibit the second probe or observation radius
increases to `3/2`.

**Claim boundary:** the original v0.1 deterministic examples (error radius
zero) are not retroactively changed. The correction applies to finite
scalar bounded-adversarial-error models and the *worst-case*, not the
uniform-expectation stochastic objective (which would require a likelihood).
Adaptive set-valued diagnosis and exact minimax decision games are
established algorithmic ideas; the embodied contribution hypothesis
remains the mapping to concrete repair authorities, admissible robot
probe/tap signals, risk budgets and refusal. No arbitrary robot-safety
theorem or external adoption is claimed.

### Related methods and novelty restriction

This bounded-noise correction is **not** a newly invented optimal-decision-tree
algorithm. Prior work already covers:

- [Cicalese, Laber & Saettler, ICML 2014](https://proceedings.mlr.press/v32/cicalese14.html):
  adaptive diagnosis decision trees with worst-case and expected testing costs;
- [Golovin, Krause & Ray, NeurIPS 2010](https://arxiv.org/abs/1010.3091):
  noisy equivalence-class determination and edge-cutting active learning;
- [Yong, Gao & Ozay, 2017](https://arxiv.org/abs/1701.06731):
  group-based active diagnosis under partial/noisy sensing;
- [Robust Diagnosis Using State-Set Observation, IFAC 2006](https://doi.org/10.3182/20060829-4-CN-2909.00231):
  bounded-error set-valued fault diagnosis.

The research contribution under investigation is **not interval partition
enumeration**. It is the robot-system-specific assignment of admissible
intervention/tap evidence to **concrete implementation/dependency/evidence-bound
repair authorities**, where a wrong repair to one member of a compensating
fault pair can make the end-to-end behavior worse. Realized improvements over
these active-diagnosis baselines require blinded, real-stack experiments and
external independent replication; they are not assumed by the finite witness.

## Native MuJoCo: hidden semantics under controller nuisance

The finite four-world example above uses a frozen experiment table. The more
demanding [native MuJoCo authority experiment](mujoco_native_authority.py)
instead builds the observation table **by stepping the official MuJoCo 3.3.7
CPU simulator**. No renderer, GPU, VLA checkpoint, or training is needed.

```bash
python -m pip install mujoco==3.3.7
python research/semrepair/mujoco_native_authority.py
```

The test constructs four concrete worlds in a one-dimensional articulated
control system, with different software-scale and actuator/plant parameters:

| Hidden world | Producer action scale | Actuator gear | Mass / damping | Correct concrete action |
|---|---:|---:|---|---|
| gain fault | 1 | 2 | 1 / 0.2 | replace gear 2 with 1 |
| producer fault | 2 | 1 | 1 / 0.2 | replace producer scale 2 with 1 |
| correct | 1 | 1 | 1 / 0.2 | no change |
| nuisance alias | 1 | 4 | 2 / 0.4 | replace gear 4 with 2 |

The controlled slider obeys the physical equation

```text
mass * q_ddot + damping * q_dot = actuator_gear * producer_scale * u(t)
```

The external motion therefore depends only on the two parameter ratios
`(actuator_gear * producer_scale / mass, damping / mass)`, given the same
initial state. Any two parameter tuples with those same ratios are
**structurally input/output-indistinguishable for every admissible input
history** in this specific ideal plant, not merely close for one sampled
trajectory. The three faulty worlds share both ratios, while they require
different *factorized local repairs*.

In contrast, a controller-input tap exposes `producer_scale * u(t)` and a
generalized actuator-force tap exposes `actuator_gear * producer_scale * u(t)`.
The pair of taps breaks the remaining equivalence under a nonzero command;
a zero-action probe necessarily carries no discriminating information.

This physical scaling symmetry is standard system-identification reasoning,
**not** a newly claimed law or theorem. The narrower novel question under
test is whether an evidence-constrained, risk-aware diagnosis policy can stop
exactly when a concrete implementation/evidence-bound repair becomes
authorized, and reject when the necessary tap is unavailable.

The program selects actual native observations:

1. **External joint state**: distinguishes the reference from the faulty
   external-response class but not the three confounded faults.
2. **Internal command / pre-transmission tap**: separates producer scaling but
   cannot distinguish gain-2 from gain-4 with compensated mass/damping.
3. **Internal actuator-force tap**: exposes the remaining confounding.
4. **Zero input negative control**: none of those taps provides information.

The independent `Fraction`-valued solver computes the least expected
experiment cost from the resulting MuJoCo measurements. It returns an
authority-optimal cost **4** under the frozen uniform-prior and declared costs.

**Essential negative controls:**

- external-only experiments: `unsafe_unidentifiable`;
- external + controller-command tap: `unsafe_unidentifiable`;
- force tap disallowed by the frozen risk budget: `unsafe_unidentifiable`;
- after executing each **correct local repair**, the reference trajectory is
  recovered in the controlled simulator.

Three separate forcing sequences and a conservative additive measurement
half-width check make accidental distinguishability by rounding less likely.
These checks are specific to the frozen deterministic plant, observation
library and perturbation model, not a general robustness theorem.

This is **not** a discovered MuJoCo defect, not a robot-task success result,
not a maintainer-approved library fix, and not independent third-party
reproduction. It is an executable causal/identifiability witness produced from
real simulation dynamics.

[Public native MuJoCo CI](https://github.com/lindicaphxag-tech/kaggle/actions/workflows/semrepair-native-mujoco.yml)

## Native ManiSkill 2×2 causal intervention: first official task-level result

The [public native PegInsertionSide factorial replay run #37715587885](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37715587885)
runs the same eight officially supplied motion-planning demonstrations
independently through the *actual upstream action-conversion/controller
implementation* in four source configurations:

| converter #1495 | controller #1472 | Official demos successfully replayed |
|---|---|---|
| Off | Off | **6/8** |
| On | Off | **6/8** |
| Off | On | **0/8** |
| On | On | **6/8** |

All four source trees are fixed to exact Git commits; the #1472 controller/test
blob is independently hash-checked. Failed conversions are retained as
failures, **not** silently removed. Zero success in the controller-only cell
makes a conventional four-way matched learned-policy comparison **impossible
on the eight development-exposed demonstrations**, which is itself a concrete
and important limit on what the data can establish.

The observed four-cell interaction in conversion success is descriptive:
`6/8 - 0/8 - 6/8 + 6/8 = +0.75` on this small development set.
It is neither statistical evidence of general task-level learned-policy
improvement nor evidence of a maintainer-confirmed upstream defect.

**The most revealing practical behavior:** fixing the controller sign alone
can destroy demonstrated behavior even though the original end-to-end pipeline
replays correctly. Correct producer/controller interaction matters; using
a *different* local repair is not automatically licensed by the same
end-effector output.

This is a test of a real existing robotics stack, not only a toy MuJoCo
witness, but **no maintainer has yet adopted the SemRepair authority framework**.
The two distinct upstream PRs (#1495 and #1472) remain separate fixes.

The associated [CPU-native two-arm Diffusion Policy run #37713921020](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37713921020)
successfully trained both baseline and combined cells for **two optimizer
steps each**, using **six paired source-seed demos/893 transitions** in each.
Both showed **0% task success** on tiny 20-step evaluation horizons.
That demonstrates reproducible *pipeline execution*, not an improvement
from one repair.

**Replication policy:** the first eight are development-exposed, and a
predeclared four-cell 32-source-episode extension is being run with the next
24 forming a **four-cell-analysis replication slice**.
These 24 are **not a completely untouched dataset**: a previous two-arm
Kaggle study replayed the first 100 source demonstrations and published
aggregate baseline+combined results. The new replication can test
converter/controller interaction on additional source episodes, but cannot
be described as a fully independent data holdout.

## Formal and real-stack evidence

| Evidence | Exact public reference | What it demonstrates | What it does **not** demonstrate |
|---|---|---|---|
| Finite active diagnosis | [Semantic ABI v0.1 artifact #20](https://github.com/lindicaphxag-tech/Isaac-GR00T/pull/20) | Optimized probe/tap decision and repair-authority stopping | Universal optimality for arbitrary nonlinear dynamics |
| Lean structural core | [Pinned Lean source](https://github.com/lindicaphxag-tech/Isaac-GR00T/tree/f69254a1e3a621a10236902ac77e9c1b1c7de668/semrepair_validation/formal) | Full identity implies authority agreement; injective authorities recover full identity; conflicting authorities forbid stopping | Formal proof of task safety or of the entire solver |
| ManiSkill rotations | [Upstream #1495](https://github.com/mani-skill/ManiSkill/pull/1495), [current-head 13/18-test CI](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37687749068) | Converter/controller contract interaction, head-exact regression | Maintainer acceptance or broad learned-policy success |
| Native physical rollout | [PickCube Kaggle assay](https://github.com/lindicaphxag-tech/ManiSkill/blob/research/native-delta-pose-assay/research/kaggle_native_assay/README.md) | Unsaturated orientation-target improvement; a **mixed** 16-step saturated outcome | General performance improvement |
| GR00T temporal boundaries | [Upstream #786](https://github.com/NVIDIA/Isaac-GR00T/pull/786), [exact-head validation](https://github.com/lindicaphxag-tech/Isaac-GR00T/actions/runs/37380686525) | Terminated vs truncated episode semantics and boundary bookkeeping | NVIDIA maintainer adoption |
| Prospective watchset | [Public freeze and transport](../../transport/SAMPLING_FREEZE_89.md) | Immutable first-seen / prediction deadlines and fail-closed expired cases | Successful I2 discovery |

The **real task success** question is intentionally kept separate from semantic agreement. In controller-aware replay, repaired semantics did not establish broad task-level superiority; negative and mixed results remain visible.

## New: official ManiSkill Diffusion Policy training pipeline (2026-10-08)

The prior native PickCube assay tested a **controller target**, not a learned
robot policy. A separate, public GitHub Actions experiment has now completed
the **official PegInsertionSide-v1 Diffusion Policy CPU pipeline** through
independent per-arm demo replay, source-seed pairing, gradient updates, real
short-horizon evaluation, TensorBoard export and clean process shutdown:

- [Full successful public run 37713921020](https://github.com/lindicaphxag-tech/ManiSkill/actions/runs/37713921020)
- [Frozen result summary and data/provenance digests](https://github.com/lindicaphxag-tech/ManiSkill/blob/research/native-delta-pose-assay/research/kaggle_diffusion_policy_peg/README.md)
- Baseline replay `6/8`; composed current #1495 + exact #1472 `6/8`.
  The six paired source seeds, **893 transitions per arm**, and different
  converted dataset SHA-256 digests are retained as evidence.
- Both ~4.40M-parameter models executed **two optimizer updates** and
  short evaluation loops. **Both achieved 0% success** over tiny two-episode
  20-step-horizon evaluations. These are entirely insufficient for a
  performance-superiority or L8 learned-policy effect claim.

This result certifies that the previously blocked **real training pipeline
is runnable CPU-only**, not that SemRepair increases policy success.
The experiment uses isolated headless-render/Gymnasium compatibility shims
and is independent of the upstream #1495 two-file patch.

A separate four-cell converter×controller replay study is underway;
until it reports real data, converter-only and controller-only effects are
not attributed or inferred from this two-arm result.

## Independent falsification requests

The most useful outside contribution is an **independent counterexample**, not a star or a restatement of the author's test log.

Please try to show one of the following:

1. The finite program stops with multiple hypotheses that authorize **different concrete repairs**.
2. A supposedly discriminating physical probe remains ambiguous under bounded uncertainty or nuisance variation.
3. A claimed source/implementation/evidence digest changes without revoking authority.
4. The same pinned source produces different results on a clean runtime.
5. A real ManiSkill/GR00T case contradicts the proposed semantic mechanism.

Open a **new issue in this public repository** with the label or title prefix `[SemRepair reproduction]`. Include the commit SHA, environment, exact commands, full output, and whether it reproduced or falsified the claim.

See also the existing [independent ManiSkill replication request, issue #1](https://github.com/lindicaphxag-tech/kaggle/issues/1). Author-generated CI and this solicitation are **not external responses**.

## Honest promotion criteria

SemRepair maintained external adoption: **not established**.

Resolved primary prospective I2: **0 at the latest confirmed public-ledger audit**.

Independent third-party SemRepair reproduction: **not yet established**.

[Braindecode NeuroRVQ #1218](https://github.com/braindecode/braindecode/pull/1218) is a **separate** externally maintained and independently validated upstream contribution. It should not be presented as SemRepair adoption.

## Reuse permissions

The public repositories currently have **no confirmed repository-wide open-source license**. Public access permits inspection and running the reproduction protocol, but does not by itself establish a general right to redistribute or incorporate source code. License selection remains an explicit repository-owner decision.

## Citation

See [Semantic ABI v0.1 `CITATION.cff`](https://github.com/lindicaphxag-tech/Isaac-GR00T/blob/f69254a1e3a621a10236902ac77e9c1b1c7de668/CITATION.cff) and cite the **exact pinned commit**. This is a software artifact, not an accepted peer-reviewed paper.
