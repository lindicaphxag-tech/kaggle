# SemRepair: no-hardware journal fit, exact 2025–2026 accepted comparators and editorial pivot

**Date:** 2026-10-10. **Status:** research strategy and actual first-party artifact plan, NOT an editorial endorsement, submission, acceptance or verified CAS partition. When this was authored, the latest 7-channel minimal repair CI and actual Pusher clip-only field activation were newly submitted; do not claim either passed until its GitHub Action concludes.

## 1. The biggest repositioning decision

**Do not market this paper as a general VLA failure-recovery method.** That title creates a severe architecture mismatch: our policies are published state-based SAC, not video-language-action agents. The achieved contribution is narrower and potentially deeper as a software-and-control systems problem:

> A control policy may be correct, but its physical action contract is silently compromised by software middleware. Two erroneous interface maps can compensate during mild probes, yet an intermediate saturation makes some commands *unreachable* to ANY upstream-only compensator. Under authorized source changes, what is the minimal, mathematically certified, provenance-bound source patch that restores the declared action box? When must the repair be refused rather than activated?

Candidate paper title (journal software focus):
**Authority-Constrained Repair of Robot Action Middleware: Signed Source Contracts, Robust Patch Synthesis, and Executable Validation**

More robotic alternative for RAS:
**When Action Compensation Cannot Restore Control: Reachability-Constrained Source Repair Across Frozen Robot Policies**

Avoid "provably safe robotic self-healing," "trustworthy VLA," or "100% success" because none is supported.

## 2. JOURNAL OF SYSTEMS AND SOFTWARE (JSS) — FIRST CHOICE **if** code-level generality is achieved

Journal home / aims: https://shop.elsevier.com/journals/journal-of-systems-and-software/0164-1212 .
JSS explicitly accepts simulation, empirical studies, formal proofs and other validation, and demands software engineering contributions *independent of one application domain*. It explicitly accommodates negative outcomes and provides an Open Science Initiative for code/scripts/data. Do not submit this as JSS "In Practice": that section targets industry applications, which we do not have. Use the regular research-article pathway.

Verified directly comparable PAST ACCEPTED JSS contributions:

**JSS 2025, volume 230, article 112499, Nenchev et al., "Compositional code-level safety verification for automated driving controllers."**
DOI https://doi.org/10.1016/j.jss.2025.112499 .
This paper's CoCoSaFe verifies safety at actual controller CODE level with compositional contracts and bounded model checking, including traditional and neural-network controllers. The main value is a REUSABLE code-verification framework, rather than requiring authors to own a physical car. We match in ambition only if our patch checker generalizes beyond the author-built 2x2 shear and is applied to independently obtained actual software versions. A signed author-generated config is NOT independently corroborated vendor code.

**JSS 2025, volume 222, article 112324, Päßler et al., "Analysing Self-Adaptive Systems as Software Product Lines."**
DOI https://doi.org/10.1016/j.jss.2024.112324 .
Introduces a reusable dynamic software product line and probabilistic model-checking approach, demonstrated on a SMALL underwater-robot case. This establishes that NO HARDWARE may be compatible with a serious JSS formal software-method paper; it does NOT establish that a single small self-generated test harness suffices. They analyze a FAMILY of configurations, whereas the original SemRepair always replaced both source modules.

**Our stronger new algorithmic response:** generalize source policy to n=2..32 actuators and exhaustively minimize user/issuer DECLARED source-edit cost over a finite, externally authorized catalog, subject simultaneously to whole action-box semantics, independent elementwise calibration error, per-channel input/output tolerances, source edit permissions and trusted *hard* clip caps. Show a concrete 7D counterexample where replacing ONLY the intermediate source clip is cheaper than rewriting both compensated erroneous matrices, but is correctly refused when that clip exceeds an exogenously specified hard cap. Cost optimality is exact ONLY over the enumerated signed candidate catalog, not arbitrary program synthesis.

**Hard JSS promotion gates, PRIOR to submission:**
- 2+ separately sourced actual upstream software defects from different repositories or versions (e.g. ManiSkill controller/converter and LeRobot action contract) with frozen BEFORE/AFTER code and issue/PR URLs; demonstrate replayable source-level fault and fix semantics, not just a conceptual analogy.
- Real source patch byte diff and version-bound source contract, with change-count and edit-surface/minimal intervention; author-generated test signing roots must be disclosed and issuer-independent from software authoring in any external corroboration.
- Scope beyond special 2x2 reciprocal shear: 7/16-dimensional contract synthesis and two unrelated defect mechanisms; 10+ meaningful negative attack categories including false-source trust and disallowed but performance-positive patches.
- Explain that numeric 2-tap inversion, Ed25519 signature verification, simple induced-norm interval bounds and finite-catalog minimization are established techniques; the NOVEL software engineering contribution, if any, is the compositional repair permission and executable proof bundle.
- Package a repeatable Open Science artifact with pinned actual upstream code, Git SHAs, source owner permission model, JSON manifest, fixed initial seeds, raw per-case runs and exact reproduction command on standard Ubuntu without specialized GPUs or robot hardware.
- Compare signed + semantics + robust constraints vs signature-only, nominal semantics only, always atomic repair, no repair, and action-only adaptation. Evaluate repair edit cardinality, false authorization, refusal coverage, compilation/runtime, input headroom and original task reward. No double-counting 960 correlated episodes as independent systems.

**Current judgment:** thematic match HIGH, evidence readiness CONDITIONAL. With only authored 2D shear and fake issuer identities, desk rejection for lacking general SE novelty is quite possible. Hardware absence itself is NOT the decisive JSS blocker.

## 3. ROBOTICS AND AUTONOMOUS SYSTEMS (RAS) — FIRST CHOICE if native robot generality outgrows software examples

Official scope: https://shop.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890 .
RAS publishes theoretical, computational and experimental methods for robot autonomy, so a computational paper is in scope. It does not automatically waive the demand for credible ROBOTICS generalization.

Direct accepted RAS no-hardware-aligned comparator:

**RAS 2026, volume 196, article 105240, Gil et al., "A model-based approach for co-simulation-driven digital twins in robotics."**
DOI https://doi.org/10.1016/j.robot.2025.105240 .
Real methodological contribution: automated RoboSim + industry-standard Functional Mock-up Interface co-simulation, model/code generation, manufacturing-cell case study and explicit quantified coverage of the digital twin framework. The paper has ~20 published pages, and the publisher/university report states ~60% DT feature coverage. Even for no real device, it provides externally defined model tooling, not just a one-robot simulated success figure. We must provide comparable modular, transferable engineering substance.

Other 2025 RAS fault-tolerant example:
**Mobayyen et al., "Robust fault-tolerant control of quadrotors under simultaneous multiple faults and input/state delays."** DOI https://doi.org/10.1016/j.robot.2025.105088 .
This highlights a higher-level fault class and robustness under delays, disturbances and multiple uncertainties. Our bounded static 2-stage software source proof is narrower.

**Hard RAS promotion gates:** add independent robotic simulator families (e.g. ManiSkill, robosuite, Isaac Lab) with frozen externally trained manipulation controller rather than another Farama SAC checkpoint; demonstrate more than one source fault mechanism, a fairness-correct horizon-budget comparison against task-aware replanning, timed closed-loop recovery and counterfactual physical action reachability. Report native task metrics, not just policy action tracking. Preserve full-action mathematical guarantees only for their clearly stated constant linear submodels.

**Current judgment:** thematic match HIGH for robotics but robustness/policy-family gap LARGE. RAS is a strong stretch target, not clearly safer than JSS. No empirical proof of cross-Farama generalization at present.

## 4. ENGINEERING APPLICATIONS OF ARTIFICIAL INTELLIGENCE (EAAI) — SECONDARY, not current first choice

Official scope: https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976 .
It explicitly includes fault diagnosis, robotics, real-time automation, and verification/validation of AI-based software. But it REQUIRES the abstract to specify the actual AI innovation and a real engineering application plus reproducibility using PUBLIC data; its submission instructions mention immediate desk rejection for abstract/format deficiencies. Our active algorithm is deterministic constrained repair planning with a FROZEN external SAC policy, not a newly trained AI diagnosis or control architecture. Calling a rule-driven source proof "AI novel algorithm" would be misleading.

Promotion only if a genuinely improved learned fault diagnosis/probing policy or dynamic planner with strong public benchmark evidence is created. Do NOT submit current SemRepair under the VLA/LLM category hoping the venue overlooks the mismatch.

## 5. NEUROCOMPUTING — WRONG PRIMARY VENUE for this manuscript

Official scope: https://shop.elsevier.com/journals/neurocomputing/0925-2312 .
Its core is neurocomputing / learned models and architectures. The proposed multi-source signature/contract verifier has no neural method novelty: using an outside frozen SAC checkpoint does not turn the correction algorithm into neurocomputing research. Consider Neurocomputing for the separate Rank Budget / learned-model paper, not this software source-authority work.

## 6. Conservative publication route and timelines

- Oct 2026: freeze the current 32-seed x 5-severity native MuJoCo benchmarks and original negative conditions. Qualify any model/contract assumption. Run newly submitted 2..32 channel solver, signed clip-only executable patch, source-permission ablation and software regression test on actual external upstream source.
- November 2026: prioritize one reproducible full JSS regular-paper submission IF at least TWO independently sourced software variants and a generic nD algorithm have passed their reproduction gates. Otherwise do not send a recycled manuscript merely to meet a month.
- December 2026: route to RAS only if multi-simulator/controller evidence is sufficient and paper is rewritten to emphasize computational robotics, not speculative software trust. Review turnaround/acceptance by a target month is not controllable and CANNOT be promised. No concurrent submission of materially overlapping work to two journals.
- A public preprint may be beneficial for feedback, AFTER anonymization/licensing and venue preprint policy checks. PR merge, GitHub CI success and preprint publication are separate forms of acknowledgment, none equal peer-reviewed publication.
- The precise current Chinese Academy of Sciences journal partition, CCF category (where relevant), edition/year and specialty must be verified from authoritative published lists before final portal submission; avoid assigning a convenient quartile based on old aggregates or English-language journal reputation. There is no meaningful evidence-backed "high probability of acceptance" estimate today.

## 7. Table/figure requirements matching earlier top journal works

JSS regular-article paper:
Fig 1 actual software components and SOURCE EDIT AUTHORITY; Fig 2 observable small-tap fault indistinguishability versus hidden saturation and permitted actions; Fig 3 end-to-end proof-carrying patch life cycle (candidate -> source signature -> calibration signature -> robust full input-envelope -> byte switch -> runtime verifier); Fig 4 fail-closed lattice under permissions and internal clip caps, not a colorful pipeline diagram.
Table 1 published JSS/robotics adjacent research with precise evidence categories; Table 2 independent source defects and exact old/new upstream SHA; Table 3 baseline fault and oracle action-domain comparisons; Table 4 actual precommitted 7D / 16D min-edit authorization outcomes; Table 5 attack and false-permission rejection; Table 6 complete native Pusher original reward/object distance with bootstrap and limitations; Table 7 CPU runtime and memory/evaluations; Table 8 pre-existing threats to validity with direct file hashes.

RAS alternative:
Fig 1 physical reachable polytopes and learned policy command trajectories; Fig 2 simulation/software interface architecture; Fig 3 normalized source repair threshold versus blackbox corrected action support; Fig 4 original Pusher goal-distance episode trajectories and bad cases; Table 1 robot embodiments and frozen pretrained model hashes; Table 2 five fault levels all outcomes; Table 3 1/3-step native MPC with simulator rollout cost; Table 4 per-robot physics and intervention permission ablations; Table 5 software-source authorization as extra result, NOT main novelty.

## 8. Bottom line

**Best current value / fit: JSS software-engineering paper** if real code-version case studies and generic program-level patch synthesis are actually completed. **Best no-hardware robotics stretch: RAS**, requiring stronger cross-simulator evidence. EAAI/Neurocomputing are weaker fits to the CURRENT core algorithm despite potentially attractive indexing metrics. The added general n-dimensional minimal repair method is a genuine incremental algorithmic change, but does not itself guarantee a paper of top accepted stature.

Reference source URLs:
- JSS journal: https://shop.elsevier.com/journals/journal-of-systems-and-software/0164-1212
- JSS CoCoSaFe: https://doi.org/10.1016/j.jss.2025.112499
- JSS SPL: https://doi.org/10.1016/j.jss.2024.112324
- RAS journal: https://shop.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890
- RAS robot DT: https://doi.org/10.1016/j.robot.2025.105240
- RAS robust quadrotors: https://doi.org/10.1016/j.robot.2025.105088
- EAAI: https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976
- Neurocomputing: https://shop.elsevier.com/journals/neurocomputing/0925-2312
