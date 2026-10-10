# Repair Authority Changes Reachability: Source-Bound Recovery of Compensating Control-Interface Defects

First-party methods/evidence freeze on 2026-10-10. NOT submitted, accepted, externally reproduced, pretrained-VLA evaluated, hardware validated, or cryptographically safe.

## Research question and distinctive contribution

Two erroneous robot software interfaces can compensate and yield correct nominal trajectories while leaving the wrong component authorized for repair. A second failure appears when an INTERNAL actuator control clip lies between them: even a fully informed black-box action adapter may be physically unable to reproduce some valid downstream forces. The work therefore combines concrete source repair permission, deployment action-support bounds, and real MuJoCo task recovery, distinguishing repair-action reachability from mere nominal observational equivalence.

The math is established convex geometry and induced norms, NOT a novel matrix theorem. A top robotics publication would need independently sourced controllers and learned frozen robot policies; this current result supplies mechanistic evidence, not that empirical standard.

## Conditional theorem and proof

For requested two-axis control u, producer P, downstream C and internal symmetric clipping L, actual actuator command is

    t(u) = C clip(Pu, -L, L).

P,C are nonsingular FIXED 2x2 real mappings. With unrestricted upstream black-box commands, the EXACT downstream reachable set is

    Y_blackbox = C([-L,L]^2).

Thus a desired t is unreachable if and only if max_abs(C^-1 t)>L. Additional constraints on upstream inputs/downstream motors cannot invalidate an impossibility witness. Projection of t onto the four parallelogram sides gives the EXACT Euclidean-optimal instantaneous black-box physical torque, not a weak heuristic.

For a full declared input square D=[-R,R]^2, the patched action law equals ideal identity t(u)=u for ALL u in D precisely when CP=I and R*max_i sum_j |P_ij|<=L. Sufficiency: clip is inactive and C P=I. Necessity: near zero, equality implies CP=I; if inner clipping occurs anywhere in D, invertible C cannot cancel the resulting nonzero clipping error. The submitted finite four-choice planner enumerates concrete authorized component patch options and exact rational patch costs, rejecting any source-less, over-budget or model-incompatible candidate.

A separate *reachability-only* check does not imply legal or behaviorally correct repair: changing controller only could make a target reachable but leave a wrong producer mapping. Source authorization is an ASSUMPTION provided externally, not a signed source-chain or actual installed patch.

## Actual successful native MuJoCo studies

### A. Articulated two-link, two-controller recovery, no internal clip

Six frozen interface configurations, two authored controllers (joint-space PD and Cartesian Jacobian transpose), sixteen seeded goals each, seven matched arms, 1,344 episode records. Interface diagnosis reads numeric tap values only, not hidden fault names. Black-box inverse sees equal probes.

| Fault/controller | Untouched | Candidate source patch | Strong matched inverse |
|---|---:|---:|---:|
| Producer swap / Cartesian | 0/16 | 16/16 | 16/16 |
| Controller sign / Cartesian | 0/16 | 16/16 | 16/16 |
| Producer swap / joint PD | 0/16 | 6/16 | 6/16 |
| Controller sign / joint PD | 0/16 | 6/16 | 6/16 |

Healthy references: Cartesian 16/16, joint PD 6/16. **No task benefit over exact black-box inverse in linear unsaturated mappings.** Compensating double faults were already successful before repair. Corrected original benchmark: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38004922699

**Transparent implementation failure:** Initial two-link MJCF used joint ranges intended in radians while MuJoCo default compiler interpreted the XML ranges in degrees, leaving physically invalid tiny angular limits and only 1/16, 2/16 healthy controller successes. Failed initial CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38004643338 remains recorded. Explicit radian compiler plus compiled joint-range assertions and exact end-of-step forward kinematics were added; ONLY the corrected same-target panel is valid for publication.

### B. Exact full-information black-box obstruction, native articulated dynamics

Source mappings P=[[1,0.8],[0,1]] and C=[[1,-0.8],[0,1]] compensate before clipping. L=9. On prespecified 49 target torques with each axis in {-8,-6,-3,0,3,6,8}, 12 targets are provably OUTSIDE the reachable set for an omniscient unlimited-input black-box oracle. Its maximum unavoidable Euclidean force error is 4.216691570992364. Same-initial-state actual MuJoCo single-step joint velocity deviates by up to 0.08243071169028704 rad/s. Correct atomic source repair P=C=I has zero deviation in this simulation. These are 49 correlated force samples in ONE robot family. CI and original JSON: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38004843665

### C. Paired closed-loop tasks with internal clipping

One 2-link MuJoCo arm; 16 fixed goals for both authored controllers; five arms each; 160 episode traces. The black-box oracle knows the full clipped source model and projects the requested physical torque onto the exact reachable set at EACH timestep, paying no probing cost. This is instantaneous action optimality, NOT global future task optimality.

| Controller | Best per-step black-box terminal successes | Source atomic terminal successes |
|---|---:|---:|
| Cartesian Jacobian | 16/16 | 16/16 |
| Joint PD | 8/16 | 9/16 |

The source repair removes the requested-versus-executed torque gap entirely within the frozen policy amplitude <=8 and internal L=9; black-box projection leaves a positive gap when unreachable. Task success DOES NOT broadly improve. Mean integrated task error is Cartesian 0.02978 oracle vs 0.02936 source, but joint PD is 0.06286 oracle vs 0.06312 source: no uniform advantage on every outcome. Raw trace CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38005241107

### D. Minimum-cost authorized deployment-envelope planner

With the same P,C and internal L=9: action box radius R=2 needs noop with declared cost 0; R=8 requires atomic source repair cost 2 to preserve ALL desired commands; R=10 has no feasible permitted patch. Missing externally provided source attestation makes every candidate REFUSE. 7 tests passed in public CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38005402665 . The test passes externally_verified=True as a controlled assertion, NOT an actual authentication of any publisher.

## Closest accepted work, gaps, and novelty priorities

- ICML 2026, Yan et al., Can VLMs Diagnose and Recover from VLA Manipulation Faults?, https://proceedings.mlr.press/v306/yan26r.html : 20 VLMs, LIBERO and REAL robotics fault recovery. Substantially stronger learned-policy external validity than this authored two-link actuator assay.
- IEEE RA-L 2026, CEI: A Unified Interface for Cross-Embodiment Visuomotor Policy Learning in 3D Space, DOI:10.1109/LRA.2026.3656802 : policy transfer across 16 simulated embodiments and real arms. This new two-link model is NOT 16 embodiments.
- HRI 2026, Banerjee et al., A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies, https://emprise.cs.cornell.edu/modularhil/ : module attribution, calibrated intervention cost and actual assistive robot study. Stronger external end-to-end intervention evidence.
- AISTATS 2014, Javdani et al., Near Optimal Bayesian Active Learning for Decision Making, https://proceedings.mlr.press/v33/javdani14.html : decision-region queries are prior art, as are source matrix identification, box geometry, Hamming bounds and Lipschitz covering.

## Red-team review and top-journal decision

There is a real mathematical and simulator-mechanical phenomenon (source repair changes actuator feasible set), and a strict finite-domain permission planner. But neither is by itself a top-tier robotic learning contribution. Source-provenance identity, independent tap issuers, actual code-byte patch installation, real external controller/robot, and frozen policy task gain are unimplemented. The black-box comparator is pointwise force optimal, not a globally optimal task controller. Sixteen correlated targets are NOT independent systems and do not justify confidence intervals about fleet transfer.

**Required next experimental paper gate:** frozen external pretrained manipulation policy on a distinct robot/control architecture, permission-qualified trusted interface observations, matched probe/action cost across strong active adaptation baselines, true unsafe authorization and abstention risk, prospective unseen controllers, and outside-lab replay. Until then label work as Draft; do NOT claim accepted CAS-2 / CCF-B level or universal safety.


## 2026-10-10 first external frozen pretrained robot-policy result (new and stronger evidence)

The prior report lacked any pretrained policy. We have now actually executed official Gymnasium Reacher-v5 with the third-party Farama/Minari pretrained SAC-expert checkpoint, pinned to full source commit ad8fadfbe3ab40ce9c1bae5b33641041056b3090, downloaded model byte SHA256 095cd4cfe742a07c9e2c5587ca95fe1ce97360efaaa02c7e6569fe9ca00c6454, Stable-Baselines3 2.6.0 and Gymnasium 1.1.1. There were ZERO author training steps. This is an external RL policy, NOT vision-language-action (VLA).

The INITIAL 16-seed 80-episode pilot with (shear=.8, inner L=1) showed NO effect: official mean reward -2.593024 for healthy, faulty, inverse-blackbox, inner projection and source fix; raw green CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007320983 . This negative result led to a post-pilot, fully enumerated five-severity *new-seed* protocol. It is NOT a pure preregistration of the entire study.

The new independent reset seeds 23011..23042, five fixed fault levels and six same-seed arms were committed before test evidence; 960 paired episode rows, no rejected severities, official original Reacher reward. The strongest blackbox arm is an EXACT 2D GLOBAL PROJECTION of the desired instantaneous action into the physically reachable output set after BOTH hidden software clip AND original Reacher action clamp. A separate fitted numeric low-amplitude two-tap+holdout method classifies the two component mappings as atomic; the simulator then performs source-known adapter substitution. Crucially this method DOES NOT attest source bytes, authenticate sensors, or execute a signed production patch.

| Fault shear, inner clip L | Unpatched mean original reward | Perfectly informed action-wise blackbox oracle | Proposed numerical source candidate | Mean source minus oracle |
|---|---:|---:|---:|---:|
| k=.8, L=1 (pilot severity on fresh seeds) | -2.993996 | -2.993996 | -2.993996 | 0 |
| k=2, L=1 | -2.994736 | -2.994310 | -2.993996 | +0.000314 |
| k=3.5, L=1 | -3.322207 | -3.024224 | -2.993996 | +0.030227 |
| k=2, L=.4 | -4.375293 | -3.217566 | -2.993996 | +0.223570 |
| k=3.5, L=.4 | -8.154021 | -3.476482 | -2.993996 | +0.482485 |

At strongest severity, paired seed outcome for source-vs-oracle original reward: **30 better, 0 worse, 2 ties among 32 reset seeds**; 32/32 goal-ever-reaching at 0.075 for source and oracle, fault untouched 29/32. At minimal severity 32/32 seeds tie exactly. The high-severity effect therefore concerns normalized physical action fidelity and official cumulative reward, **not an extensive improvement in the 0.075m reach-success outcome**. Fault setups were authored, and there is ONE external robot model / ONE published expert checkpoint in this experiment.

Full heldout CI and raw JSON: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007699504 . Protocol committed pre-holdout-result: https://github.com/lindicaphxag-tech/kaggle/blob/research/six-line-submission-gates-20261010/research/semrepair/paper/EXTERNAL_FROZEN_SAC_POSTPILOT_PROTOCOL_20261010.md . All five fault levels are reported, and uncertainty analysis must use per-seed paired differences, not 960 i.i.d. robots. After observing the expert results, a second distinct Farama/Minari SAC-medium checkpoint was queued as sequential replication, NOT a blinded before-any-result multi-checkpoint trial. Its outcome must not be stated until the external CI completes.

**Updated review-grade assessment:** this materially reduces the earlier zero-pretrained-policy evidence gap and gives a nontrivial external-policy action-reachability outcome that exceeds an exact per-step action blackbox oracle for some fault configurations. Remaining limitations: SAME Reacher embodiment, manually constructed extra action ABI and clipping, source-trust assertion, no actual installed software patch, no VLA/visual task, no globally task-optimal MPC blackbox, no third-party independent replication. These preclude present T-RO/RA-L/ICML-level parity or acceptance guarantees.


## Independent published checkpoint replication: SAC-medium (executed after expert inspection)

A DIFFERENT Farama/Minari pretrained SAC-medium checkpoint has now been pinned to source revision **7366bc42655652e36705974ec234a6456d695222**, actual downloaded bytes SHA256 **ee557cdd49c0a41b94256ab36a05a776f3bcf4ad2d7800d15051c56b747c4ea9**. The exact SAME official Reacher-v5 environment, 32 seed identities, 5 source fault configurations and 6 arms were evaluated WITHOUT new training or changing faults after the expert result. This is sequential cross-checkpoint replication with two policies from one publisher; NOT independent robot/embodiment generalization and NOT fully blinded pre-pilot experimental design.

Green 960-episode run and raw artifact: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007987369

| Predeclared (shear k, hidden inner software clip L) | Medium faulty mean return | Medium exact action oracle mean return | Medium numeric source candidate mean return | Source minus oracle |
|---|---:|---:|---:|---:|
| (0.8,1) | -3.342713 | -3.342713 | -3.342713 | 0 |
| (2,1) | -3.345532 | -3.342686 | -3.342713 | **-0.000027** |
| (3.5,1) | -3.659634 | -3.366105 | -3.342713 | +0.023392 |
| (2,0.4) | -4.270007 | -3.544794 | -3.342713 | +0.202081 |
| (3.5,0.4) | -9.560426 | -3.770020 | -3.342713 | +0.427307 |

At strongest fault the second checkpoint exhibits 30/32 better and 2 ties (0 worse) source-vs-exact-action-oracle in OFFICIAL cumulative reward, but both still reach the crude 0.075m goal proximity threshold in all 32 seeds. At moderate shear k=2,L=1 there is a **slight source disadvantage** relative to the projection oracle; retaining it rules out universal method dominance. These two checkpoints are not independent manufacturers and the severe injected adapter is purposely engineered by the authors. The desired real-paper result must survive an **independent environment/robot body**, software provenance, tamper control, truly external policy variation beyond two SAC variants, and stronger global task-optimized MPC.

### Descriptive uncertainty audit from original unmodified raw files

An independent GitHub Action, using actions/download-artifact to fetch the immutable prior expert run **without rerunning/tuning it**, verifies the full 32-seed denominator, checks all five predeclared fault-severity comparisons, excludes exact ties for two-sided exact sign tests and applies Bonferroni correction for five contrasts. The strongest expert severity yields paired seed-bootstrapped descriptive 95-percent range **[+0.372819,+0.593425]** original reward, mean **+0.482485**, two-sided sign p adjusted for five severities **9.31323e-9** (30 better, 0 worse, 2 ties). The bootstrap distribution is only a sensitivity measure over 32 pseudorandom seed identities within ONE robot model, not proof of statistically independent robots or safety. Expert frozen original raw SHA256 **5048325bd46d98b04b1e2877b71c9d11c3edd2ab74c38cfc38e8084e3d4b288d**. [Green original paired statistical audit](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38008067450). Separate medium-checkpoint paired uncertainty audit is being run from its original raw artifact; do not present it as completed until CI confirms.

**Net scientific claim (conditional and narrower than top robotics):** In official Reacher-v5, two independently published frozen SAC checkpoints exhibit mean official-reward gain from source interface repair relative to an oracle instantaneous action reproducer under *severe purpose-built software clipping*, while their nominal weak-fault cases are neutral or slightly worse. Not demonstrated: learned VLA recovery, overall best task controller, secure patch provenance, cross-robot family generalization, or acceptance probability at top venues.
