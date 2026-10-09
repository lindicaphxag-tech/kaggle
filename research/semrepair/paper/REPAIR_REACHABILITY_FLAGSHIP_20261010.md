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
