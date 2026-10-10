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


## Major 2026-10-10 advancement: SECOND official robot embodiment, Pusher-v5

**New first-party experiment on a truly different externally supplied seven-actuator MuJoCo robot and its independently published frozen RL checkpoint.** This is no longer only the Reacher-v5 model. Publisher and environment are still both Farama Foundation; not independent hardware, VLA, or a non-Farama controller implementation.

- Official Pusher-v5: 7D joint torque, 23D observation, **object pushing** with original unmodified 100-step reward (object-goal distance + fingertip-object proximity + action penalty). Gymnasium 1.1.1, Stable-Baselines3 2.6.0.
- Published SAC-expert: farama-minari/Pusher-v5-SAC-expert pinned full commit **3e103602f3fba3500ee6f21d762c0ff3e94892ff**, verified model ZIP SHA256 **6a2a51ddc1f2539aaa779bb7390c3022a0e42effc096962ee2bdaa1fb74f37b3**. Zero author retraining, weights unchanged.
- To avoid conflating robot dimensions, the authored source interface P=[[1,k],[0,1]],C=[[1,-k],[0,1]] affects ONLY first two shoulder controls (0 and 1), with a planted intermediate source clip; other FIVE physical robot channels are unchanged. Final Pusher official [-2,2]^7 actuator contract is untouched. Relative clip severities from Reacher rescaled to official limit 2, not hand-tuned on Pusher results.
- Prespecified 32 initial-state seeds 23011..23042, five unchanged fault strengths, six paired methods, **960 genuine official Pusher episodes x 100 steps**, all cases and negative outcomes preserved. Numerical two-tap+heldout diagnostics select an atomic patch from internal values without symbolic true fault labels. Source patch outcome still relies on author's simulated adapter substitution and unverifiable internal tap trust; NOT a securely installed signed patch.
- Strongest comparator: an exact FULL-KNOWLEDGE one-step Euclidean-optimal blackbox actuator-output projection onto the reachable set after BOTH authored intermediate clipping and actual Pusher final actuator clipping. Thus a poor naive inverse is not the only baseline; however it is still NOT a global future-reward-optimizing MPC policy.

### Observed COMPLETE five-condition test results (not cherry-picked)

| Authored shear k / internal clip L | Unpatched official Pusher mean return | Instant-action optimal blackbox return | Numeric source candidate return | Source-oracle reward gain | Mean object-goal distance reduction from oracle |
|---|---:|---:|---:|---:|---:|
| 0.8,2 | -25.532645 | -25.532645 | -25.532645 | 0 | 0 |
| 2,2 | -26.020487 | -25.867955 | -25.532645 | **+0.335309** | 0.002253 m |
| 3.5,2 | -38.138008 | -27.075393 | -25.532645 | **+1.542748** | 0.010995 m |
| 2,0.8 | -33.868015 | -29.412782 | -25.532645 | **+3.880137** | 0.028355 m |
| 3.5,0.8 | -55.793656 | -34.271298 | -25.532645 | **+8.738653** | 0.058898 m |

At the strongest severity, reported mean ORIGINAL physical object-to-goal distance is faulty **0.237735 m**, best instantaneous action oracle **0.224518 m**, and numeric source candidate **0.165620 m**. This is importantly a true task-geometric effect, not merely reward shaping: absolute object-goal distance drops **5.89 cm versus the oracle** on average across the same 32 reset seeds. It does not imply a predeclared discrete task success-rate increase, and it is not an independent multi-manufacturer test. Paired reward directions: **30 source-better, 2 source-worse, 0 tied**. At mild fault all 32 ties. Because the source candidate is a simulator implementation-level ideal repair, its task return exactly equals the healthy checkpoint reference and should not be marketed as field-installed code.

Actual GREEN external-robot native MuJoCo CI with all 960 raw episodes, reward decomposition, source checkpoint SHA, and every fault level:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009127179

Protocol prepared before seeing the Pusher results (but after seeing earlier Reacher pilot and severity results):
https://github.com/lindicaphxag-tech/kaggle/blob/research/six-line-submission-gates-20261010/research/semrepair/paper/PUShER_CROSS_ROBOT_FROZEN_PROTOCOL_20261010.md

Independent audit re-reads the archived raw Pusher file rather than rerunning selected cases. It separately computes **10 predeclared paired seed contrasts**: 5 original cumulative rewards and 5 *actual object-to-goal physical distance* endpoints with joint multiplicity adjustment and separately decomposes fingertip-proximity and official control penalty, to check for spurious energy-only improvements. Its outcome must be reported only after its CI actually passes. A second externally published SAC-medium Pusher checkpoint is a FOLLOW-UP chosen after seeing expert results, not a pre-results fully blinded independent study. Its result must not be added until the second checkpoint CI runs.

### Current novelty and realistic comparison to accepted 2026 top results

**Empirically stronger than the previous SemRepair draft:** across Reacher-v5 and Pusher-v5, actual external robot physics, two independently pretrained reward functions and frozen RL checkpoints now show strong severity-dependent source repair improvement versus the exact best instantaneous actuator-input blackbox. On Pusher, the gain also appears in the physical object-to-goal metric.

**Do not overclaim:** the reachability statement and optimal projection are standard geometry; the original scientific idea must be about identifiable, source-bound, authorized repair UNDER actuator non-invertibility. The object-level improvement may partly stem from the engineer-authored interfaces and specific shoulder-channel choice, and does not prove a statistical population effect across independently built robots. Authors have not authenticated producer/controller module origins or actual patch-byte installations. A globally task-optimal adaptive controller or full-horizon MPC could differ substantially from the strong instantaneous action comparator. No vision or language agent is involved.

- **ICML 2026** Yan et al. *Can VLMs Diagnose and Recover from VLA Manipulation Faults?* https://proceedings.mlr.press/v306/yan26r.html — 20 VLMs, real VLA and real robot manipulation. Our work has less task/policy coverage and no real-world proof.
- **IEEE RA-L 2026** *Reliable Robotic Task Execution in the Face of Anomalies*, DOI 10.1109/LRA.2025.3632090 — real Kinova and xArm deployments with anomaly-triggered repair; our current simulator cannot claim corresponding validation.
- **CVPR 2026** *FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation* https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html — visual-language learned manipulation and retry/reset, not a software source authenticity study but much stronger learned policy evidence.

**Go/no-go:** one additional independent manufacturer/policy architecture, actual source attestation with executable signed patch switch, globally task-aware adaptive baseline, and outside-party replay are required for T-RO/RA-L/CoRL excellence. Neither performance transfer in two Farama tasks nor 960 trial count alone is a top-tier paper acceptance certificate.


## Executed Pusher source-file statistics and second independently published actor — added after above frozen expert test

### First-party SEED-PAIRED reanalysis from independent GitHub workflow, not hand-picked new runs

Original Pusher SAC-expert raw file from GitHub Actions #38009127179 was independently read by a separate audit workflow #38009661288; full original raw row SHA256 is **7abafeb9a4b9bed0f74a5d5ccf00a5ba0b4932ae1a6219e1c66697af9c0ea58c**. It refuses missing/duplicate conditions, incorrect reward components and corrupted source-to-healthy identities. It uses the same 32 reset seeds in each of all five fixed faults and 10 simultaneous exact sign tests (original return and true object-to-goal distance) with a ten-test Bonferroni multiplier.

For the strongest k=3.5,L=.8 comparison (source candidate minus full-information instantaneous action oracle):

- Original reward gain mean **+8.738653**; descriptive 32-paired-seed bootstrap 95% **[+7.123533,+10.347212]** original reward units.
- TRUE physical **object-to-goal distance reduction mean 0.058898 m**; descriptive paired-seed bootstrap 95% **[0.0455503,0.0716533] metres**.
- 10-test familywise corrected exact two-sided sign p ≈ **2.46335e-6** for both reward improvement and object-goal-distance reduction.
- Fingertip-to-object distance also reduces mean **0.079979 m**; the SOURCE method has a **more negative** mean official control penalty by 0.0115012 reward units per step than the action oracle, so the cumulative reward gain CANNOT be attributed solely to a cheaper action penalty. Physical object progress and fingertip proximity more than compensate for higher source action control cost.
- Mild k=.8,L=2 shows 0 reward and goal-distance effect as expected; intermediate k=2,L=2 has reward improvement +0.335309 with an object distance bootstrap CI slightly crossing zero (about [-0.000002,0.004585] m). Do NOT misstate all five cases as uniform large physical benefit.

Successful independent original-artifact audit with both major endpoints and full metrics:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288

### SECOND externally published Pusher-v5 SAC-medium checkpoint — sequential cross-actor generalization

This second policy was selected after seeing the first published SAC-expert cross-robot result; all first Pusher faults/seed IDs were reused WITHOUT changes. Farama/Minari published Pusher-v5-SAC-medium is pinned to full revision **d7d16ee5dfaa1f9e43df96cbdd6846cead1dabb6**, verified exact ZIP SHA256 **6ad4f083db0713cb5c6ec649cbcff50c5205e25c78756aed89f9921d75b1d2b8**; zero author training. Full second 32*5*6 = **960 official Pusher episodes GREEN**:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009508516

| Fixed Pusher fault (shear k, clip L) | Medium policy: unpatched return | Full-information instant action oracle return | Numeric source candidate return | Source-oracle mean original reward difference | Physical object-goal distance reduction |
|---|---:|---:|---:|---:|---:|
| 0.8,2 | -31.553906 | -31.551123 | -31.551519 | **-0.000396** | 0.000008 m |
| 2,2 | -31.869618 | -31.684600 | -31.551519 | +0.133081 | 0.001033 m |
| 3.5,2 | -45.414064 | -32.497173 | -31.551519 | +0.945654 | 0.005904 m |
| 2,0.8 | -47.818633 | -33.941309 | -31.551519 | +2.389790 | 0.011218 m |
| 3.5,0.8 | -92.944053 | -37.096160 | -31.551519 | **+5.544641** | **0.020738 m** |

Strongest source versus oracle result: **32/32 positive reward differences**, no negative or tied seed; source returns -31.551519 and oracle -37.096160; object-to-goal distance source 0.215279 m, oracle 0.236017 m. Weakest fault has a tiny negative reward difference in the published medium checkpoint, so this is not universal dominance.

**Conclusive current evidence scope:** two externally provided Farama robot environments (Reacher and Pusher) and four pinned published SAC actor checkpoints, each evaluated on carefully engineered 2-component clipping source defects; positive source-vs-exact-instant-action-oracle original reward and object-goal-distance effects at severe Pusher faults. This is materially stronger than the first first-party two-slider apparatus, but there is no real VLA, no external independent signer roots or installed robot software patch, and no full-horizon globally reward-optimal competing recovery controller. Not an independently confirmed academic top-paper or guaranteed accept-ready.


## Further 2025–2026 prior-art challenge: actuator saturation/reachable control set is already studied

A focused literature reread uncovered closely relevant accepted actuator-fault papers which MUST appear in prior-art discussion rather than calling our parallelogram geometry a new standalone control result:

1. Yoon et al., "Practical fault-tolerant control allocation based on attainable control set analysis for a coaxial dodecacopter", *Control Engineering Practice* **156:106235 (March 2025)**, DOI **10.1016/j.conengprac.2024.106235**, https://pure.kaist.ac.kr/en/publications/practical-fault-tolerant-control-allocation-based-on-attainable-c/ . This already studies achievable control sets, clipped actuator allocation, pseudo-inverse and QP compensation and experimentally measures a real test-jigged actuator fault. It undercuts any novelty claim that simply drawing a reachable-force polytope or outperforming a naive pseudoinverse is revolutionary.
2. López-Araujo et al., "Fault tolerant adaptive control under actuator saturation for robot manipulators", *ISA Transactions* **168:247–256 (January 2026)**, DOI **10.1016/j.isatra.2025.12.004**, https://pubmed.ncbi.nlm.nih.gov/41381325/ . It covers manipulator power limitations and adaptive fault tolerance with analytical stability claims. Our fixed linear-ABI box proof is strictly narrower in fault-control theoretical sophistication.
3. Yan et al., "Can VLMs Diagnose and Recover from VLA Manipulation Faults?", *ICML 2026*, https://proceedings.mlr.press/v306/yan26r.html . This benchmark includes 20 VLMs, LIBERO, and real robot fault recovery. Ours still lacks VLA, perception and physical-hardware evidence.
4. Zhao et al., "FLARE: A Failure-Aware Framework for Autonomous Correction and Recovery in Visual-Language Robotic Manipulation", *CVPR 2026*, https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_FLARE_A_Failure-Aware_Framework_for_Autonomous_Correction_and_Recovery_in_CVPR_2026_paper.html . Real learned recovery for contact-rich tasks underscores the policy/scale gap.

**Reframed non-overstated contribution candidate:** deploy-time **component-source legitimacy and signed byte-level patch activation** under *noninvertible compensating software ABI*, with a precise declared command envelope and prospective real task effects. The pure reachable control polytope, generic QP actuator allocation and simple system identification are background, NOT our novel intellectual claim.

A NEW author-generated-key Ed25519 two-of-three validator and actual on-disk official Pusher runtime adapter activation have executed (old non-semantic gate full task CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010249329, 48 episodes). A further source-identity-alias attack, exact signed task-envelope semantic certificate and an action-level task-reward model baseline have been added and require new CI before any verified claim. Signer labels alone cannot be regarded as distinct roots if they resolve to the same 32-byte public key.

Research competition implication: demonstrating a reachable set is NOT enough for an excellent Control Engineering Practice or RA-L paper. The genuinely differentiating evidence must be a real byte-installed source-authorized recovery system with legitimate external source truth, plus an information-strong *true task reward* and planning comparator on independent policies and physical bodies.


## 2026-10-10 final higher-standard methods update: signed PHYSICAL semantic checks versus privileged task planning

A prior line in this research could claim "source was trusted" solely from an author-supplied Boolean. That line is superseded by a real author-key **Ed25519 2-of-3 signed actual source adapter file activation** with old/replacement config SHA-256, fixed actor family, deployment envelope digest and independently supplied action-radius, current/target anti-rollback epochs and challenge. A fully-signed patch is STILL rejected when its new mapping violates the entire declared 2D action box's required identity semantics CP=I, or when its producer amplification exceeds the signed internal clip, as well as for patch outside signed component scope. This fail-closed check is conditional on a known fixed nonsingular linear source model. The verified installed file is then read anew by the frozen third-party original SAC policy on official seven-axis Pusher.

**Latest strengthened successful real native CI:** https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050 . 16 adversarial control/provenance/signer tests + 4 live task-integration tests passed. The 16 already inspected seed native original Pusher tasks each are executed via actual old config, native reference and newly installed signed config (48 episodes). Latest run: corrupted source -56.395976 return and 0.245791m object-goal distance; signed real FILE-backed restored -26.379423 and 0.175368m; official native healthy reference exactly -26.379423 and 0.175368m in that same run. 0 real manufacturer signing identities, 3 author-generated test private roots, no independent sensor provenance, no ROS/vendor production install, and no new held-out efficacy sample. This is an actual controlled software-repair execution path, **not** independent external adoption of a certified patch.

### More difficult global task-control counterargument

The previous source-vs-"exact blackbox" gains across two Farama robot bodies, four pretrained SAC checkpoints and genuine physical object-to-goal distances were **exact for per-step action reachability**, never globally reward-optimal black-box policies.

A much stronger, fail-fast red-team protocol is now committed: [original-reward oracle and 3-step MPC protocol](https://github.com/lindicaphxag-tech/kaggle/blob/research/six-line-submission-gates-20261010/research/semrepair/paper/NATIVE_TASK_REWARD_MPC_REVIEWER_REBUTTAL_20261010.md). The baseline is GIVEN the exact original Pusher native MuJoCo state, true model, exact hidden fault and official task reward, plus frozen policy inference in lookahead branches. It simulates physically feasible candidate actions, with true original task reward used instead of action Euclidean deviation. Distinguish greedy one-step from finite 3-step receding horizon. It has privilege and compute vastly exceeding our source repair, and neither is yet a full-horizon optimal MBRL controller.

**Critical current preflight:** the first task-reward oracle's live native reward identity test FAILED with a 0.00013573 discrepancy (cloned model -0.60885615, official -0.60872042) in CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010613448 . A later variant copied full MuJoCo INTEGRATION state and removed an extra forward pass, but the same discrepancy still appeared in https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010871956 . Both invalid runs are preserved; neither can be used as positive/negative task performance evidence. A per-component qpos/qvel/body-geometry diagnostic test has been added, without weakening the 1e-7 accepted reward precision. The 3-step lookahead and retry CI must pass the SAME native reward identity gate before displaying meaningful winner/loser outcomes. NO unverified planner score is included here.

### Higher standard peer comparison: closest accepted fault/recovery studies

1. [ICML 2026 VLA-FixBench](https://proceedings.mlr.press/v306/yan26r.html): 20 VLMs, perception, planning and control faults, LIBERO and real-robot fault recovery. Their task/architecture externality and independent failures far exceed two Farama SAC policies.
2. [AAAI 2026 TCoT](https://ojs.aaai.org/index.php/AAAI/article/view/37577): local/global trajectory reasoning for failure recovery of visual-language-action robotic manipulation, not a source-authenticity study but closer to deployment task recovery.
3. [IEEE RA-L 2025 Fault Handling in Robotic Manipulation Tasks for Model Predictive Interaction Control](https://doi.org/10.1109/LRA.2025.3592069): model-predictive interaction, wrench and pose fault pipeline, trajectory planning and low-level control. A native Pusher one-step or 3-step candidate enumeration is not a substitute for an end-to-end comparative MPC study.
4. [CoRL 2025 Simple Masked Training Strategies Yield Control Policies That Are Robust to Sensor Failure](https://proceedings.mlr.press/v270/skand25a.html): real and simulated robot morphology variation for sensor faults, complementing our software action-source focus.

**Meaningful venue gate:** signed candidate source repair and physically privileged reward planning are promising ingredients, not proof of independent manufacturer authorization, no physical VLA, no true external natural source failure, and no proof of full-horizon task optimality. If genuine reward planning eliminates the advantage, the paper thesis must be narrowed to verified software semantics and action reachability; it cannot continue claiming task-level superiority. Manuscript and PR remain DRAFT.
