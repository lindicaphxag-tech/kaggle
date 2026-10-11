# Mechanism-to-system upgrade: signed semantic-validity patch + native physics reward-model blackbox

**Created before observing the new privileged-planner or signed-semantic-gate GitHub CI outcome. Research PR remains DRAFT.**

## Why this is a stricter paper than the previous version

Three separate, NONINTERCHANGEABLE claims are needed:

1. **Mathematical reachability:** For engineered intermediate-clipped t=C clip(Pu,-L,L), desired downstream torques outside C[-L,L]^2 are impossible even for an oracle upstream adapter. Standard geometry, NOT original linear-algebra theorem.
2. **Authority AND behavioral correctness:** Real Ed25519 signatures alone do not ensure a behaviorally valid patch. Byte-executable 2-of-3 source activation additionally tests the complete deployment action box [-R,R]^2: CP=I and R*max_abs_row_sum(P)<=L. Malicious but validly signed maps, insufficient clipping and downgraded declared action radius MUST be refused. This is conditional fixed-model validity, not whole-robot safety.
3. **Task-level recovery vs adequate adaptive control:** The old exact blackbox was globally closest to the requested actuator action PER STEP but not necessarily maximizing actual task return on a contact-rich pushing robot.

## Already executed external pretrained policy evidence

Pusher-v5 pretrained SAC-expert with 32 seeds * 5 fault levels * 6 arms:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009127179

Independent published second SAC-medium actor, same task and faults:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009508516

Original expert Pusher severe source vs exact-action-oracle: +8.738653 original cumulative return and -0.058898m TRUE object-to-goal mean distance, from unchanged original evidence with reward-component decomposition:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288

Under severe faults, source has a MORE negative action control penalty than exact-action-oracle. Therefore the task reward increase is not only due to saving energy.

## Executed first-party signing-root native task

Ed25519 issuer signatures, true bytes and nonce/epoch checks, plus full frozen Pusher 100-step loops from an ACTUAL signed replaced and re-read on-disk source config, completed green:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010249329

16 PREVIOUSLY INSPECTED reset seeds * 3 methods = 48 real native robot episodes. Old actual on-disk faulty config mean return -56.395972 / object-goal distance .245791m; actually activated signed new on-disk config -26.400801 / .175498m, same as healthy external SAC. These are author-generated signing roots and author-created adapter configs, NOT manufacturer's actual signing roots, installed ROS source binaries, or independently authenticated sensor taps. A new additional mathematical semantic gate for signed config is being validated separately and must not be claimed passed until its CI finishes.

## New stronger full native model task reward blackbox comparator

The new blackbox gets an ENTIRE true privileged official Pusher-v5 MuJoCo state, exact model, native three-part reward, planted hidden software matrices and constraints, and a finite set of physically reachable seven-joint torque candidates each step. It clones the true MuJoCo state, simulates each candidate one official control step in separate MjData, evaluates exact native task reward, chooses the best for this one-step reward, then continues the SAME frozen policy. It pays potentially tens of thousands of native model evaluations. The old exact Euclidean closest-action oracle is evaluated alongside it.

Prespecified exploratory panel is the PREVIOUSLY INSPECTED Pusher severe k=3.5/L=.8 and reset seeds 23011..23026, 100 step original official tasks, three methods: healthy reference, exact action tracking oracle, stronger privileged native original-reward planner. Results are NOT independent prospective efficacy and cannot be given independent holdout significance.

The planner has stronger INFORMATION, more COMPUTE and official reward than our source candidate; however finite candidate search and one-step horizon are NOT globally optimal long-horizon MPC. Its predicted one-step score must equal true official environment step reward on its actually selected action; nonidentity fails the entire experiment rather than suppressing errors.

Pending CI workflow: SemRepair official Pusher full-information native reward planning oracle; no result or improvement may be claimed until real passing CI and full raw episode artifact.

## Accepted-paper comparison

ICML 2026 VLA-FixBench: https://proceedings.mlr.press/v306/yan26r.html evaluates VLM/VLA diagnoses and recovery in LIBERO and real robots; our current policies are STATE-input SAC, not VLA.
ICML 2026 VLA-Arena: https://vla-arena.github.io/ spans 170 manipulation tasks; our published benchmark footprint is far smaller.
Prior convex geometry, active decision-region sampling, and system identification are NOT claimed novel.

No task-optimal globally reward-maximizing blackbox, no trusted robot manufacturer roots, no true patch provenance in ROS deployment, no non-Farama simulator or independent lab replication. Hence not yet top-tier-acceptance-ready despite genuine experimental/engineering advancement.
