# Fatal-neighbor audit: GaugeBench and executable-policy semantics (2026-10-08)

**Status:** Novelty claim restriction; not a new experimental result and not a claim of publication priority.

## Closest recent research

1. Malladi, Sangwan, Gupta and Rahman, [*The Robot Is Not Its Description: GaugeBench for Representation Robustness in Morphology-Aware Policies*](https://arxiv.org/abs/2610.07597), posted 2026-10-06. This work reports that physically equivalent rewritings of joint-axis direction, zero offset, naming, and ordering can change morphology-aware policy behavior dramatically. It also reports **exact two-description transport** restoring controller behavior.
2. Tai, [*Same Weights, Different Robot: A Deployment Safety View of VLA Policies*](https://arxiv.org/abs/2606.03724), posted 2026-06-02. This paper treats action normalization metadata and controller conventions as part of an executable policy specification and presents metadata-mismatch auditing.
3. Sobanbabu et al., [*Sampling-based System Identification with Active Exploration for Legged Sim2Real Learning*](https://proceedings.mlr.press/v305/sobanbabu25a.html), CoRL 2025. Active physical parameter identification is established prior art.
4. Equivalence Class Determination / Decision Region Determination and optimal adaptive decision trees are established algorithmic concepts. They are NOT invented by SemRepair.

## Claims now explicitly removed

SemRepair must not claim novelty merely for:

- physically equivalent robot descriptions being represented by different coordinates;
- controller breakage under sign reversal or joint reorderings;
- exact transport between **known** representation descriptions;
- a deployment containing more than its model checkpoint;
- active system identification, information gain, or optimal equivalence-class decision trees in isolation;
- proving that multiplicative control gain and inertial mass may be confounded.

These are background or prior art, including the recently published GaugeBench.

## Narrow claim requiring experimental proof

**Hidden, evidence-constrained, repair-authority identification**:

When the source and destination semantic ABI are *not* reliably declared,
the robot must decide **which risk-bounded physical probe and which
internal observation tap** to expose, until all surviving semantic
worlds bind the **same exact implementation/evidence/dependency-bound
repair authority**.

If two worlds remain observationally equivalent under the admitted
observations and demand different concrete repairs, **do not guess**:
return an unidentifiability / insufficient-authority result.

Difference from GaugeBench: GaugeBench constructs a *known*
physical equivalence and tests policies under known rewriting.
SemRepair targets **recovering the repair-decision-relevant unknown
interface semantics using active evidence** and refusing a repair when
the corresponding authority is not identifiable.

Difference from ExecSpec: ExecSpec checks / identifies a
**specified executable object** and its metadata mismatch.
SemRepair requires *identification under untrusted or incomplete specs*,
including compensating producer/controller defects and physical nuisance.

Difference from active system identification: the terminal target is
not an estimated physical parameter vector but a **concrete,
implementation-bound authorized software repair**; a complete
parameter estimate may be unnecessary or impossible.

## Falsifiable tests required before a strong paper submission

1. **Hidden-gauge blind test.** Use third-party-generated physical
   convention transforms (e.g. sign/ordering offsets) and hide their
   transform parameters from the diagnosis method. The operator must
   identify the repair authority by probe/tap evidence, not by reading
   the known transformation specification.
2. **Compensating aliases.** Present two worlds with identical
   end-to-end actions and trajectories but different *local correct*
   repairs; require refusal until an internal tap genuinely separates them.
3. **Confounding nuisance.** Vary controller gain, mass and damping
   between episodes; ensure the procedure does not infer one semantic
   convention solely from correlated physical nuisance.
4. **Authority-optimal vs full identification.** On task-realistic
   libraries compare probe/risk/tap cost and *wrong-authority rate*,
   including cases where extra identification is provably unnecessary.
5. **Effect-level falsification.** After authorizing a specific local
   repair, run an independent target environment and report end-effect
   errors and task success, including negative or saturated-controller
   cases. A finite synthetic table alone is insufficient.

No reading of a test world's hidden source spec may be used as
the *method's* semantic oracle; offline authors may inspect it
only when scoring the independently frozen result.

## Evidence currently available vs missing

**Available:** native MuJoCo controlled alias/tap/repair closure
([public PR #31](https://github.com/lindicaphxag-tech/kaggle/pull/31));
independent exact-rational finite oracle
([PR #26](https://github.com/lindicaphxag-tech/kaggle/pull/26));
[ManiSkill #1495](https://github.com/mani-skill/ManiSkill/pull/1495)
current-head 13/18 focused-test compatibility and one deterministic native
controller assay.

**Missing:** blinded third-party generated gauge transforms, independent
hardware/learned-policy benchmark evidence, a maintained upstream SemRepair
adoption, and resolved frozen prospective I2 credit.

**Promotion rule:** The flagship will not claim a novel universal
representation gauge-invariance repair, a verified learned-policy
improvement, or L8/L9 external adoption without those missing gates.
