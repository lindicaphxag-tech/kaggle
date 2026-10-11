# SemRepair first-scored blind-interface MuJoCo task result, 2026-10-10

**Status: first-party native simulation. Not hardware, learned VLA, authenticated software installation, third-party independent replication, or submission.**

## Frozen study identity

MuJoCo 3.3.7, two-axis slider-body with independently actuated motors, fixed hand-coded joint-space PD (Kp=12, Kd=4), timestep 0.01s and 400 steps per target. Sixteen seeded targets per interface configuration, all reported without dropping failures. Terminal Euclidean error less than 0.10 is a success. All trials involve one authored robot plant and controller: sixteen targets are NOT independent robot/controller populations.

Raw JSON, 5 native task tests, 6 numeric identification tests: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38001307456

Native actuator saturation stress + exact envelope CI on Python 3.10, 3.12, 3.13: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38000739015

## Experiment A: source-known intervention counterfactual

| Interface condition | Unpatched | Wrong singleton patch | Appropriate source-oracle patch |
|---|---:|---:|---:|
| Producer swap only | 0/16 | 0/16 controller-only | 16/16 producer-only |
| Controller sign only | 0/16 | 0/16 producer-only | 16/16 controller-only |
| Mutually compensated double swap | 16/16 | 0/16 either singleton | 16/16 atomic |
| Mutually compensated inverse rotations | 16/16 | 0/16 either singleton | 16/16 atomic |

The appropriate oracle patch uses hidden source truth. On the compensated cases, the original unpatched system already completes tasks: atomic repair does NOT increase its success rate. This identifies nonmonotone global effects of locally plausible changes, not automatic policy improvement.

## Experiment B: numeric-probe fault-label-free reconstruction

Two independent 2D input probes provide a requested action u, producer-output p and controller-output t. The inference function recovers independent linear interface maps P and C from the two input-output columns; an additional held-out numeric probe tests consistency before returning one of four candidate patches. The evaluator supplies numeric tap results without disclosing its hidden symbolic fault name or implementation matrices to the inference function. The native simulator is stepped to read its actual actuator input boundary.

| World | Untouched tasks | Recovered patch family | Replay |
|---|---:|---|---:|
| Reference | 16/16 | noop | 16/16 |
| Producer swap | 0/16 | producer-only | 16/16 |
| Controller sign | 0/16 | controller-only | 16/16 |
| Mutually compensated swap | 16/16 | atomic | 16/16 |
| Compensated rotations | 16/16 | atomic | 16/16 |
| Additional invertible shear and inverse, absent from original permutation catalog | 16/16 | atomic | 16/16 |

These experiments are conditional on honest numeric internal taps and an actually linear interface. A fresh algebraic shear mapping is NOT a second independent robot family or unseen software vendor. The candidate patch is not legally or cryptographically authorized, and no live software patch installation occurs.

## Experiment C: off-support actuator saturation

Replayed original MuJoCo motor interface worlds with low maximum command 0.75 versus frozen high maximum command 13. In 3/3 controlled fault worlds, wrong and correct source patches have matching low-action reference trajectories; at high command magnitude wrong-patch trajectories diverge in 3/3 while correct patches remain reference-equivalent in 3/3. A separate finite-breakpoint analytical routine gives an exact sup bound for scalar normalized clipped motor commands on the requested actuation envelope, under its explicitly restrictive physical assumptions.

## Nearest peer-reviewed baselines

- Yan et al., ICML 2026, Can VLMs Diagnose and Recover from VLA Manipulation Faults?: https://proceedings.mlr.press/v306/yan26r.html . Broader across 20 VLMs, robot manipulation fault taxonomy, LIBERO and real robot. Our self-designed classical controller is not comparable in empirical reach.
- Javdani et al., AISTATS 2014, Near Optimal Bayesian Active Learning for Decision Making: https://proceedings.mlr.press/v33/javdani14.html . Decision-region identification is prior art; so are elementary linear matrix inversion and saturation breakpoints.
- Robot Trains Robot, CoRL 2025: https://2025.corl.org/program/main-conference . Real robot policy adaptation remains unverified in SemRepair.

## Non-negotiable next gates

1. Execute externally sourced pretrained policy / task with hidden interface conventions, varied controller families and matched strong decision-region baselines.
2. Add stochastic, falsified, stale and nonlinear tap evidence with explicit safe refusal and unknown-world coverage.
3. Enforce real source attestation and implementation-byte-to-executed-patch integrity. The present source string is not an authenticated issuer.
4. Independent re-execution on never-used model architectures or robot tasks. Bootstrap over independent source/controller groups, not sixteen goals.
5. No general robot safety, optimal active probing, higher journal readiness or actual third-party validation claims based on these controlled results.

**Working core claim:** repair permission requires interface-localization evidence and deployment-support validity; nominal trajectories alone cannot justify a source patch. The present studies establish testable physical counterexamples and conditional numeric diagnosis, not yet an external-domain robot self-healing algorithm.
