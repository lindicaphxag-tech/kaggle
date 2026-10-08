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

Because the physical dynamics depend on the ratios of effective force to mass
and damping to mass, three **different repair authorities** can have exactly
the same observable state trajectories.

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
