# SemRepair — submission-oriented method, falsifiers and results (2026-10-10)

**Working paper:** *Which Interface May Be Repaired? Provenance-Constrained Authorization under Compensating Robot Faults*.

This is a **draft method-and-results core**, not an accepted paper, a novel active-learning theorem, a trusted-sensor implementation, or hardware safety evidence.

## Abstract (research draft)

Nominal robot trajectories can conceal different producer, controller and actuation semantics. Identical motion does not identify which software or mechanical interface should be repaired. We examine evidence-limited *repair authorization* rather than unrestricted hidden-parameter diagnosis. An authorized intervention requires a common concrete patch identity across every world still compatible with admissible observations, verified source constraints and current repair-artifact bytes. When critical observations are unavailable or untrusted, the method declines intervention. We formalize the resulting indistinguishability obstruction and build finite exact policies and a MuJoCo dynamics witness. On four predefined physical worlds with three held-excitation sequences, a full observation policy authorizes 12/12 conditions correctly, whereas external-only evidence authorizes 3/12. However, its mean query cost equals a strong exact active-diagnosis oracle (4 vs 4), so no query-efficiency improvement is claimed. In six matched physical conditions, replacing only a deep actuator-force observation causes 6/6 wrong-interface decisions by an unverified actor; suppressing that untrusted probe yields 6/6 conservative refusals. A second native experiment finds 72 compensating wrong-interface patches whose output trajectories remain identical to correctly repaired trajectories. These controlled counterexamples motivate treating provenance and authorization as separate from trajectory performance. They do not establish transfer to unseen controllers, real hardware, actual sensor attestation, or learned-policy task recovery.

## Setup

Let `W` be a finite set of possible robot implementations and `A: W -> H` map each world to a **concrete** repair hash that binds patch, implementation, evidence and dependency. Probes `j` have an observation function `O_j(w)`, cost `c_j`, risk `r_j`, and an externally supplied trust/admissibility flag. Only predeclared trusted probes whose risk and cost fit the budget are selectable. From a transcript `t` form the surviving set

`W(t) = { w in W : O_j(w) agrees with every observation in t within the specified error model }`.

Issue `A(w)` only if `{A(w): w in W(t)}` is a singleton and actual patch and dependency files match the bound SHA-256 digests. Otherwise return **REFUSE**. File integrity is checked through no-symlink regular file descriptors. This *does not authenticate sensor hardware* or rule out changes after file verification.

### Proposition 1: trusted-observation impossibility

If `w_a` and `w_b` have identical transcripts under every admissible trusted probe yet require `A(w_a) != A(w_b)`, any deterministic policy based solely on trusted probes must either refuse on their common transcript or authorize incorrectly for at least one world.

**Proof.** The policy sees the same transcript in each world and therefore must output the same patch-or-refuse decision. If it outputs a patch, that patch cannot equal both distinct authorities. Thus universal correct authorization requires abstention. The argument is a direct specialization of classical observational equivalence and decision-region identification, **not a newly invented optimality theorem**.

### Proposition 2: untrusted-tap substitution

Suppose two worlds are identical on trusted channels `T` but differ at an untrusted actuator-force tap `F`, and `A(w_a) != A(w_b)`. A substitute observation `F(w_b)` attached to `T(w_a)` creates exactly the same model-predicted transcript as world `w_b`. Any policy that presumes `F` authentic can thus authorize `A(w_b)` while operating on `w_a`. A policy that declines untrusted `F` must refuse if `T` leaves their repairs conflicting. The experiment below **constructs** this witness in native MuJoCo. This is an information-level limitation, not a model of a deployed cyberattack.

## Experiments and raw provenance

| Check | Real measured observation | Source and limitations |
|---|---|---|
| Finite native physical recovery | 12/12 correct repairs when full controlled tap set admissible; only 3/12 external-only, 6/12 when force unavailable | [CI #37975041561](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37975041561); 4 fixed known worlds x 3 excitation waves, NOT 12 independent robots |
| Strong active diagnosis | Matched **independent optimal Bellman** uniform-prior expected probe cost 4; staged approach also 4 | Same CI; **no algorithmic cost advantage**, baseline is matched to admissible sensor set |
| Output-semantic counterexample | 24 command sequences x 3 fault types = 72 dependent cells: wrong-interface compensation and source-authorized repair both recover nominal trajectory while touching different components | [CI #37973635217](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37973635217); one constructed plant family |
| Malicious evidence-source counterexample | Actual native MuJoCo dynamics, 2 aliased world identities x 3 held excitations; **6/6 wrong patch decisions when deep force readout is substituted**, **6/6 refusals if deep tap is declared untrusted**, 6/6 correct if force reading genuine | [three-version CI #37984576868](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37984576868); adversarial source substitution is *synthetic*, no sensor authentication |
| Executable combined gate | Tested links among candidate authority identity, allowed observation set and four actual artifact hashes. Correctly refuses an untrusted probe or tampered patch | [CI #37985266948](https://github.com/lindicaphxag-tech/kaggle/actions/runs/37985266948), pure finite-module tests; model isolation and trusted-source identity are externally stipulated, not certified |

**Sample accounting:** A physically stepped MuJoCo model creates real simulation traces, but the 12 and 6 experimental cells share the same four or two known latent physical world types. The 72 wrong-patch comparisons share a single physical construction and are not 72 independent robot embodiments. None of these has a patient-safety, statistical fault-prevalence, clinical or independent hardware interpretation.

## How the contribution differs from prior art

The target “identify a decision region rather than the whole hidden hypothesis” is **established prior art**, including Javdani et al., *Near Optimal Bayesian Active Learning for Decision Making*, AISTATS 2014 (https://proceedings.mlr.press/v33/javdani14.html). Finite Bellman decision-tree optimality and stopping on a singleton repair equivalence class are likewise not presented as new mathematics.

The narrower possible original contribution is an **executable authority contract across compensated physical actions, evidence-source admissibility and actual implementation-bound patch bytes**, together with falsifiers showing why a correct motion trajectory is not sufficient to license a code change. An active-diagnosis system augmented with identical evidence provenance and patch contracts is an obligatory baseline. Until the external/unseen-controller experiments establish an improvement under equal information access, **the method remains a candidate paper, not a T-RO-level advance**.

A neighboring robotic policy-deployment safety view is *Same Weights, Different Robot* (arXiv:2606.03724); source-level semantic contracts must not be misrepresented as wholly new.

## Hard failure and success conditions before submission

1. **Novel task:** genuinely unknown controller and actuator implementation families, with source code and true patch location withheld from the acting method. A fresh Python restatement of these same worlds does not satisfy this condition.
2. **Truly matched competitors:** exhaustive cost-optimal authority decision region, full-identification controller diagnosis, best decision-region active learner, no-intervention, downstream task-success-only repair, and a provenance-aware variant of the strongest competitor. Match probe costs and physical risk.
3. **Untrusted inputs:** implement and measure actual source attestation, independent measurement corroboration or an expressly scoped adversarial threat model. A caller-provided `trusted=True` Boolean is insufficient.
4. **Outcome separation:** report (i) implementation repair identity, (ii) abstention/missed repairs, (iii) actual post-patch physics, (iv) learned-policy downstream task success. Do not conflate one with another.
5. **Statistical independence:** each new controller family and test episode is a different unit; paired tests under the same seed/initial configuration; do not promote 6/6 or 12/12 to population safety probabilities.
6. **External recognition:** signed independent run/reproduction or sustained upstream adoption. Author-controlled GitHub CI is valuable evidence but not third-party adoption.

**Editorial stance:** target T-RO only if the method beats strong diagnosis and recovers tasks on genuinely new robot controllers. T-RO initial Regular paper limit is 18 pages as of Oct 2026 (https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/). Otherwise choose a more narrowly scoped software correctness/robot semantics venue.
