# Source Authority, Not Action Compensation: Verifiable Repair of Noninvertible Robot Middleware

Consolidated research manuscript, methods and evidence as of 2026-10-10. **PRE-SUBMISSION DRAFT**, not accepted, independently replicated or robot-manufacturer authenticated. Paper strengths are a bounded mechanistic result, two public pretrained RL policies per two official robot embodiments, external software-patch activation mechanism, and hard numerical controls. All five original source fault severities and negative outcomes remain in the original artifact.

## Abstract

Robotic manipulation failures are usually treated as errors in perception, policy outputs or downstream motion planning. Yet robot software can contain a less visible fault: two individually incorrect control interfaces may cancel near the origin while becoming noninvertible after an intermediate actuator clip. We study whether source-authorized repair can restore behavior that no upstream action compensator can reproduce pointwise. Our method identifies a two-stage linear middleware defect from numerical taps, computes a full-action-domain semantic repair certificate, and requires separately signed patch and calibration evidence before activating exact source-file changes. A robust extension rejects configurations whose claimed behavior is not certified under bounded coefficient drift. We evaluate original Gymnasium Reacher-v5 and Pusher-v5 physics with four independently published, unchanged Farama SAC checkpoints, 32 paired seeds per condition, five fixed defect levels and strong fully informed action-space comparators. For the severe Pusher interface fault, source restoration improves the original task return by 8.739 and 5.545 over the exact instantaneous action projection for SAC-expert and SAC-medium respectively, and reduces actual mean object–goal distance by 5.89 and 2.07 cm. We then introduce falsification controls with privileged true native MuJoCo physics and official reward: on 16 previously inspected SAC-expert seeds, a greedy one-step planner and three-step actor-in-the-loop MPC return −47.697 and −47.480 versus −26.379 for the healthy source reference, with 28,800 and 30,400 model branch evaluations respectively. Separately, a demonstrative Ed25519-signed real file activation restores 16 previously inspected Pusher tasks to healthy reference. These results establish a narrow form of authority-constrained action reachability and practical proof-carrying middleware restoration, not general dominance over task-optimal planning, robust safety in unknown systems or VLA transfer.

## 1. Introduction: the fault class is the controllable software boundary

Robots contain multiple software owners: policy authors, system integrators, control-interface maintainers and actuator controllers. A policy can request an appropriate command while an intermediate source-level ABI changes its physical meaning. Two erroneous interfaces can also compensate exactly in small-signal tests. Consequently an apparently correct nominal command-to-actuator relation can conceal an unauthorized or unreachable configuration when an internal clip activates.

The crucial distinction is intervention authority. A black-box compensator may replace the command entering a middleware module, but may not replace the bytes of that module; an authorized source correction has a different intervention surface. A performance comparison is meaningful only after making these authority sets explicit.

This paper deliberately DOES NOT introduce a new theorem of convex geometry, a pretrained VLA, a secure hardware manufacturer PKI or an end-to-end VLM reasoning loop. Our candidate contribution is the combination of (i) an observability-limited fault diagnosis, (ii) explicit source-side reachability constraints, (iii) an executable signed-and-semantically-certified source repair, (iv) quantitative task evidence on external frozen robot policies, and (v) unusually strong falsifying model-based comparators.

## 2. Threat, authority and statistical models

Plant: official Farama/Gymnasium Reacher-v5 (2D action) and Pusher-v5 (7D action, first two joint torques corrupted). Models and rewards are completely unmodified. The published SAC policies from Farama/Minari are loaded from immutable commits and exact model-file SHA256; no author retraining.

Defect: a first-party injected 2x2 real linear producer P followed by a componentwise inner clip with magnitude L followed by a 2x2 linear controller C. The control commands for Pusher joints 2 through 6 are unchanged. All declared P,C are invertible; for the compensating family C=P^{-1}. Fault parameters are fixed per predeclared severity, not learned from healthy evaluation scores.

Source sensing: two linearly independent small-signal numeric taps and an independent held-out tap estimate the producer and controller stages. Taps are author-instrumented and not authenticated physical sensors. When source is unavailable for modification or the external trust identity is absent, source repair must be refused.

Authority trust: patch signers and calibration signers must belong to distinct Ed25519 public-key domains and each meet a separately verified quorum. Test keys and their domain selection are first-party and ephemeral, not independent manufacturer certification. Calibration coefficient intervals are declared exogenously, not statistically estimated by our source code. The host operating system and offline signing-key provisioning are trusted.

Statistical unit: one official environment RESET seed is an evaluation unit within a fault level, NOT one independent robot. Multiple defects and controllers on the same seeds remain correlated. The 16-seed planning work is a retrospective subset of the prior 32-seed expert experiment and must NOT be called an untouched holdout.

## 3. Conditional physical action obstruction

Let the requested 2D action be u in D=[−R,R]^2, with interface output

    F(u) = C clip(Pu, [−L,L]^2).

Suppose an upstream black-box controller can choose any pre-interface input v but cannot alter P, C or L. Since P is nonsingular and its input is unrestricted, the EXACT set of possible downstream commands is

    Y = C([−L,L]^2).

For Pusher the official actuator clips again to [−2,2]^2, so the exact physically delivered two-channel set is

    K = clip(Y, [−2,2]^2).

No upstream input can produce a command outside K, even with perfect knowledge of the fault, infinite computation and no communication noise. This statement concerns replication of individual actuator commands, NOT whether an entirely different action sequence might still succeed at the task. Our exact action oracle projects the frozen actor's requested first two controls into K, globally minimizing instantaneous Euclidean error, and receives the full true P,C,L for free.

An authorized software switch can change P,C and L. Exact identity output for EVERY u in D is equivalent, for nonsingular C and P and full-dimensional D, to both

    CP = I,
    R max_i sum_j |P_ij| <= L.

The first is forced by the derivative at the unsaturated origin. If the second fails, the stage clips for some input and a locally flat output direction contradicts global identity. This is an established induced-infinity-norm fact; no novel matrix algebra is claimed.

## 4. Robust and signed source activation

A valid signature alone is insufficient to guarantee correct physical behavior. A signed patch is denied unless its exact file bytes pass the full deployment semantic condition above, its component changes lie in the authorized patch class, and its immutable old/new SHA256, epoch, nonce, robot identifier and trusted deployment action radius all match caller-pinned values.

Let actual producer/controller matrices be P+δP and C+δC with independently bounded coefficient uncertainty |δP_ij|<=dP_ij and |δC_ij|<=dC_ij. A SUFFICIENT fixed-model certificate for every u in D is

    R max_i sum_j (|P_ij|+dP_ij) <= L,

and

    R max_i sum_j (
       |(CP−I)_ij| + sum_k (
          |C_ik| dP_kj + dC_ik |P_kj| + dC_ik dP_kj
       )
    ) <= ε.

Then the intermediate clip never activates under the stated error bounds and the worst-case executed action satisfies ||F(u)−u||∞ <= ε. The bound is conservative and conditional on the validity of the exogenously supplied intervals; it is not a statistical or learned OOD certificate.

The stricter implemented gate verifies two distinct threshold-signed materials: an owner-approved software source patch and a separately signed calibration profile bound to exact replacement bytes, action radius, actor family, epoch and deployment digest. A public key cannot be reused across owner and calibrator domains. An insufficient clip, forged signer, replayed nonce, downgraded action contract, wrong component mutation or source-byte mismatch fails BEFORE replacing the live adapter config. The source is atomically written and then checked again at runtime.

This is runnable cryptographic software, with test-generated signers. It does not mean external robot manufacturers agreed to the patch, the coefficient bounds were truly measured or sensor taps came from secure hardware. It also does not protect against an adversarial operating-system root user between verification and file replacement.

## 5. External frozen policy evidence

All original task rewards are preserved. Pusher outcomes use genuine object-to-goal geometry from the official model, not a manually shaped success indicator. Five predeclared severities are retained for each policy.

### Reacher-v5

Two distinct published frozen SAC expert/medium weights; 32 held-out environment reset seeds × five source strengths × six matched interventions = 960 actual task episodes per checkpoint. Most severe inner clip L=.4 and shear k=3.5: original task-return gain over the exact physically reachable instantaneous blackbox action oracle is +0.482485 (expert) and +0.427307 (medium). Source candidate equals the healthy-reference actor only because the authored interface is actually corrected. Mild source faults can be exact ties, and a medium-checkpoint moderate fault shows a small negative difference. These results do not involve physical object pushing.

Original expert strong-fault CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007699504
Second checkpoint https://github.com/lindicaphxag-tech/kaggle/actions/runs/38007987369

### Pusher-v5

The second distinct official robot embodiment has seven actuator controls and an object-pushing task. Faults affect only joints 0 and 1. Reusing the same dimensionless fault intensities after scaling internal clip by the official action bound, 32 seeds × five severities × six arms again yield 960 tasks per external frozen SAC checkpoint.

| Fault k / clip L | SAC-expert source − instant oracle mean original return | SAC-medium source − instant oracle mean original return |
|---|---:|---:|
| 0.8 / 2.0 | 0 | −0.000396 |
| 2.0 / 2.0 | +0.335309 | +0.133081 |
| 3.5 / 2.0 | +1.542748 | +0.945654 |
| 2.0 / 0.8 | +3.880137 | +2.389790 |
| 3.5 / 0.8 | +8.738653 | +5.544641 |

At severe k=3.5,L=.8, the mean ACTUAL object-goal distance declines 0.058898m for expert and 0.020738m for medium relative to exact instantaneous oracle. This is not exclusively action-energy reward shaping: source correction uses MORE control-energy penalty while attaining lower true physical object distance.

The expert original 960 rows were downloaded by an independent workflow and independently audited. Across 32 previously fixed seeds in the most severe case, expert source beats instantaneous oracle in official task return for 30 and loses for 2, with paired-seed descriptive bootstrap 95% interval [+7.123533,+10.347212], and true object-distance reduction bootstrap [0.0455503,0.0716533] metres. Ten planned severity × endpoint exact sign tests used Bonferroni adjustment; no 960-independent-robots inference is made.

Original external expert CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009127179
Original source-controlled physical endpoint audit https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288
Second frozen Pusher medium checkpoint https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009508516

## 6. Reward-aware, physics-privileged black-box falsifiers

Instantaneous action projection is not the globally best task recovery algorithm. We therefore grant subsequent black-box comparators MORE privileged information: exact native MuJoCo physics and full contact state, original official task reward, complete hidden fault matrices and no diagnosis cost. Candidate actions remain within the faulty output set K; unchanged five Pusher joints follow the frozen actor.

A GREEDY 1-step planner maximizes original immediate task reward over a finite source-admissible grid, replanning at every real robot simulation step. A stronger H=3 true-physics planner evaluates each candidate's future 3 official action rewards; future tail actions come from the SAME frozen published SAC policy after reading the simulator's exact official 23D observations and projecting their desired torques into K.

A hard identity gate originally failed: copying only MuJoCo integral physics variables and using a different reward evaluation boundary produced ~1.36e-4 disagreement with the genuine official Gymnasium reward. Both planners were rewritten to use full MjData copies (native MuJoCo mj_copyData), exact native mj_step(frame_skip) and official reward component timing. Validation now passes at 1e-7; the H=3 selected action's maximum measured real reward difference is approximately 1.1e-16.

All methods below use the SAME 16 previously inspected expert Pusher initial seeds and the strongest fixed authored fault. These are retrospective tests, not fresh independent generalization.

| Method, 16 actual native Pusher episodes | Mean ORIGINAL return (higher better) | Mean physical object-goal distance (lower better) |
|---|---:|---:|
| healthy actor / source restoration upper bound | −26.379423 | 0.175368m |
| exact instantaneous source-unreachable action projection | −34.382247 | 0.228390m |
| true official reward greedy H=1 black-box planner | −47.696714 | 0.259913m |
| true official reward H=3 SAC-tail MPC | −47.480364 | 0.259913m |

H=1 actual native candidate evaluation count 28,800; H=3 executes 30,400 model branch evaluations, 440,000 internal physics integration steps and 57,600 frozen-SAC tail predictions. Source reference yields +21.317291 original reward versus H=1 and +21.100941 versus H=3, with 16/16 source wins on the previously evaluated seeds. The reward gap is accompanied by lower true physical object-goal distance.

Official original-reward greedy strong-baseline CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013779442
Official full 3-step native MPC CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013970154

**Crucial qualification:** these reward-privileged baseline CONTROLLERS have no source authority and a short planning horizon. H=3 optimizes a finite initial-action grid and then follows projected frozen SAC tails. It is NOT unrestricted H=100 globally optimal contact-rich control; under different costs or longer search it could match or outperform source restoration. The observed inability of THIS finite planner to outperform source restoration must not be restated as impossibility of all black-box task success.

## 7. Exact on-disk patch activation and dual-root safety checks

On 16 previously inspected Pusher SAC-expert seeds, a genuine Ed25519 quorum-signed source config is atomically installed, re-read from disk every task step and evaluated for 100 official MuJoCo steps. It exactly restores the original health-reference actor task reward and object-goal distance, unlike the corrupted disk config. The latest signed action-semantic CI has 16 attack/permission tests and 4 end-to-end execution tests green: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050 . These 48 total episodes are a software execution VALIDATION, not novel held-out efficacy evidence.

The dual-domain gate is stricter: separate owner and calibration roots each need 2-of-3 valid Ed25519 signatures; external-profile SHA, source-bytes SHA, action radius and epoch must match. Only if the signed calibration uncertainty assumptions certify the full action interval does actual source file activation proceed. Ten adversarial and native Pusher integration tests have passed: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38014106998 . The calibrator keys and coefficient intervals are still FIRST-PARTY/ASSUMED, so they do not validate source provenance in an independent deployed robot.

Conditional robust uncertainty interval fuzz: 6000 bounded action-perturbation comparisons and fail-closed refusals, green https://github.com/lindicaphxag-tech/kaggle/actions/runs/38012354585 .

## 8. Comparison to published work and precise differentiator

Yan et al., "Can VLMs Diagnose and Recover from VLA Manipulation Faults?", ICML 2026 (PMLR 306:143967–143983), actually study 20 VLMs on a broader perception/planning/control fault framework including LIBERO and real robots: https://proceedings.mlr.press/v306/yan26r.html . Their breadth and real manipulation deployment outweigh ours; our focus on software source authority and its controllability obstruction is DISTINCT, not evidence of higher general VLA competence.

Other published contact-rich MPC and fault-handling works already predict dynamics, contact and task reward; a short privileged H=3 SAC-tail model control is a FALSIFIER, not a newly invented MPC optimizer. For example, "FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation", IEEE Robotics and Automation Letters 10(6), 2025, DOI 10.1109/LRA.2025.3560871, uses real force/torque signals. We have no analogous force sensor or actual hardware experiments.

## 9. Open vulnerabilities before a main-track strong paper

The policy and task evaluations now provide more credible evidence than a hand-authored 2D slider. Nevertheless a top venue reviewer could still reasonably reject on the following grounds:

1. Natural source defects are NOT established: all middleware shear+clipping cases were planted by us, and correcting the planted source to identity is partly an oracle intervention in the efficacy benchmark.
2. External learned policies are state-based SAC, not frozen pretrained VLA models. The two official physical robot bodies come from the same Farama family.
3. The patch signers and calibration signers are generated by our own test harness; no vendor identity, real ROS or robot software patch, trusted-source firmware, signed sensor data or independent outside-lab confirmation exists.
4. Short-horizon full-information MPC competitors do not establish full-horizon model-based optimality, particularly under expensive contact dynamics.
5. We did not run honest independent physical robot or different actuation-type deployments. Parent-child goals, middleware saturation and performance numbers cannot be extrapolated to medical-robot safety.
6. No rate-limited, cost-matched complete-episode source diagnosis/repair timing study is yet published against equivalent blackbox planning costs, nor full policy error penalty under source identity loss.

The submission gate is therefore to obtain evidence on at least one independent non-Farama learned manipulator with a real source defect, trusted signer/calibration origins and properly matched long-horizon baseline. No specific top-tier acceptance probability is justified without that evidence. Current manuscript is experimental research, not accepted or submitted.
