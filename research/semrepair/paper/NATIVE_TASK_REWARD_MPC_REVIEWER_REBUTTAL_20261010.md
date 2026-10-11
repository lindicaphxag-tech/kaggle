# Beyond action matching: reviewer-grade black-box actual-reward planning controls

**Protocol committed before viewing any new successful three-step task results. The older Pusher-v5 SAC-expert/severe-clip outcomes are already known, so this is deliberately RETROSPECTIVE red-teaming, not an untouched preregistered test cohort.**

## Strongest previous finding, with precise scope

Previously, in Pusher-v5 seven-actuator contact-rich pushing on a published frozen SAC-expert policy, our source patch candidate was compared to a mathematically **exact nearest reachable instantaneous action** black-box oracle under deliberately planted internal source clipping. Across 32 previously fixed seeds, source repair improved the original official average cumulative reward by +8.738653 at strongest source fault (shear=3.5, clip=.8) and mean physical object-to-goal distance by 0.058898 metres. An independent raw-file statistical audit verified this original task improvement and original reward/energy decomposition: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288 .

But this does not establish task-level optimality against an adaptive black-box that can change actions in anticipation of future robot contact dynamics. It is misleading to call our earlier projection comparator a globally optimal control baseline, even if its instantaneous Euclidean action projection was exact.

## Stronger baselines (before actual outcomes)

**B1. Privileged one-step original-reward planner:** for each actual Pusher time step, clone the FULL MuJoCo integration state, enumerate a physically reachable candidate set under the full true hidden software fault, physically integrate the same official Pusher frame_skip, and choose the candidate with largest immediate **ORIGINAL** official reward (weighted true object-to-goal, fingertip/object and action penalty). This black-box knows the actual robot state, simulator, original task reward and complete fault parameters; it pays no inference cost. It leaves the frozen external SAC actions for the other five joint controls untouched. The cloned reward must equal the actual official reward of the selected action within 1e-7; otherwise the study is INVALID. It is a finite-candidate one-step baseline, not a globally optimal dynamic-programming oracle.

**B2. Stronger three-step receding-horizon Pusher MPC:** the same physically privileged black-box enumerates 3x3 extremal/center feasible first two-channel source controls plus the exact Euclidean projected actor action, then simulates **three full future official Pusher physical action steps** through MuJoCo for each first-action candidate. After the first branch action, each future action is produced by the SAME external frozen SAC actor applied to the simulated official 23D observation and then projected to the full two-clip reachable set. Score branches by the sum of the original three-task rewards and execute the highest-valued first action in the real official simulator, then replan at the next real step. Every candidate forward model rollout and frozen actor inference is counted. This intentionally grants more physics/reward/fault-state information to blackbox than the source repair diagnostic.

Neither B1 nor B2 globally optimizes the entire 100-step Pusher reward: B1 is greedy, B2 has finite first-step candidate grid and actor-followup tail, not unrestricted optimum. Therefore even if source patch wins these comparisons, claiming it beats ALL blackbox adaptation remains invalid.

## Fixed trials / archival discipline

- Same pinned published Farama Pusher-v5 SAC-expert checkpoint and official Farama native Pusher-v5, published checkpoint sha256 6a2a51ddc1f2539aaa779bb7390c3022a0e42effc096962ee2bdaa1fb74f37b3.
- Same severe authored fault shear k=3.5, internal clip L=0.8, original Pusher 7D torque limits [-2,2], only channels 0 and 1 faulty; all other channels copied exactly.
- **16 ALREADY-INSPECTED reset seeds 23011..23026**, not a new heldout panel, and each official episode has 100 actions.
- B1 reports healthy original actor reference, exact instantaneous action projection, true-reward one-step planner. B2 reports the same healthy and action projection comparator plus three-step true-reward-optimizing MPC.
- Main results are ORIGINAL mean Pusher cumulative reward and original physical mean object-to-goal distance; report per-seed better/worse/tied counts, TRUE physical model evaluations and actual action execution divergence. If the planner defeats healthy source repair on some seeds, KEEP them; method has no guaranteed reward superiority over adaptive control.
- Fail closed for prediction mismatch, clone pollution, invalid Pusher action bounds or missing episodes. No fabricated PR merge, source-signing vendor approval, learned VLA adoption or hardware usage.

### Parallel actual source activation gate

A separate CI performs Ed25519 2-of-3 threshold cryptographic source-manifest approval (author-generated test keys), exact OLD+NEW file SHA, atomic source-file installation, and **full declared action-envelope semantic validation** before allowing signed activation. It then re-loads the actually installed file for native Pusher SAC-expert task replay. The matched prior 16 seeds are deliberately repeated; this is an engineering activation proof, not new efficacy evidence. Successful green run for latest strengthened gate: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050 . Neither author-generated signing keys nor source file location prove independent publisher trust or protected telemetry.

## Venue comparison and make-or-break research conditions

ICML 2026 VLA-FixBench (https://proceedings.mlr.press/v306/yan26r.html), IEEE RA-L 2026 cross-embodiment learning and contemporary contact-rich model-based control work use more authentic learned manipulation architectures, stronger real hardware and more natural failure families than our two Farama robotics benchmarks. Novelty of R=C clip(Pu) geometry alone is insufficient. Publication strength rests on real-source trust, task-aware recovery under noninvertibility, independent policy/robot stacks, robust advantage versus the best budget-matched MBRL/MPC and outside-party replication.

**This note records methods, not their not-yet-verified outcomes.** Fill only real GitHub Actions run IDs, raw manifests and observed signs after CI completes.
