# Official Pusher-v5 seven-actuator external-policy transfer — frozen protocol

## Provenance / causal chronology

Created in this Draft research branch after seeing the official Farama Reacher-v5 SAC-expert and SAC-medium gains under severe planted software control clipping. This is a **second-robot, sequential mechanistic transfer**, not an independent preregistration of the entire research programme. Pusher-v5 task outcomes had **not been inspected** when this protocol and predeclared workflow were committed. All five conditions must be reported, including no effect, harm, and any failure to run.

## Genuinely new embodiment and third-party policy

- Official Farama Gymnasium Pusher-v5: 7 physical joint control channels in [-2,2], 23-dimensional state observation, original object-pushing task, original native weighted distance/contact/control action reward, 100 simulation actions per fixed reset.
- Exact externally published Farama/Minari SAC-expert checkpoint: https://huggingface.co/farama-minari/Pusher-v5-SAC-expert , pinned commit **3e103602f3fba3500ee6f21d762c0ff3e94892ff**, checkpoint **pusher-v5-SAC-expert.zip**, expected LFS SHA256 **6a2a51ddc1f2539aaa779bb7390c3022a0e42effc096962ee2bdaa1fb74f37b3** (from original publisher commit), verified at runtime and never trained/fine-tuned by us.
- Runtime Gymnasium 1.1.1, Stable-Baselines3 2.6.0, CPU deterministic predict. This is independent pretrained **reinforcement learning**, not a pretrained VLA or vision-language policy. Authored interface logic is placed before official actuator inputs without altering official MuJoCo model or pretrained weights.
- Different physical robot morphology and task from Reacher-v5; both from Farama, both state-based continuous action RL. This is NOT cross-manufacturer hardware validation.

## Frozen intervention grid / original metrics

32 predeclared seeds 23011 to 23042 inclusive, **each** of five interface configurations, each of six same-seed and same-initial-state arms = **960 complete original Pusher-v5 episodes**, 100 steps each. Six arms: healthy official policy, unpatched compensated fault, matched composite-inverse upstream blackbox (identity for this compensating pair), exact full-information action-optimal 2D blackbox projection accounting for both internal and official action clips, three-numeric-tap blind source-patch candidate, source-known atomic reference repair.

Compensated source fault acts ONLY on the first two shoulder control channels P=[[1,k],[0,1]], C=[[1,-k],[0,1]], with intermediate clip magnitude L. All remaining **five joint channels are identical to original pretrained policy**. Source clip ratio is rescaled from original Reacher [-1,1] to official Pusher [-2,2]: five fixed (k,L) conditions (0.8,2), (2,2), (3.5,2), (2,0.8), (3.5,0.8). No tuning to unseen Pusher successes.

Original Pusher per-timestep official reward decomposes into object-to-goal term (weight 1), fingertip-to-object term (weight 0.5), and squared-control penalty (weight 0.1). Report **both original cumulative reward and mean object-to-goal distance** and mean control penalty. A gain in reward caused exclusively by using less action energy is NOT proof of better pushing. A gain in fingertip-nearness alone is NOT success at moving the object to goal. Report action deviation and internal clipping events per arm.

## Method/Oracle scope

The true full-information action oracle receives planted P,C,L, the desired policy action, and exact outer official clipping. For first-two action channels its output is a mathematically exact Euclidean closest point in the image clip(C[-L,L]^2,[-2,2]^2). No learned estimation overhead is charged. Its action cost advantage is strong, but it is NOT a global-horizon task optimal policy and is not allowed to directly mutate physical inner software sources. On the other five channels it matches the frozen policy exactly.

Blind numeric repair classifies producer and downstream maps from two small probes plus one heldout check in their unsaturated domain, then an **assumed-replaceable** source adapter is reset to healthy behavior. Source identity, tamper-resistant provenance, software installation authorization and runtime cryptographic signatures are NOT established. No safety certification or hardware claims.

## Journal-grade negative evidence and fairness

A first cross-morphology task transfer can fail. If it does, leave all failed/harmful scenarios. Do not relabel arbitrary nonzero reward changes as a fixed success threshold after observing them. No 960-independent-robots inference; within each severity only 32 random environment reset identities are statistical units. Learned policy checkpoints are external while planted interface faults and choice of first two affected channels are authored. Reacher and Pusher are both Farama source family, although their physical morphology and rewards differ.

Comparison bar: ICML 2026 VLA-FixBench (https://proceedings.mlr.press/v306/yan26r.html) has real robotic manipulation and pretrained VLA/VLM faults; the present SAC source-repair mechanism still has no real VLA or physical code-origin attestation.

Evidence code research/semrepair/external_pusher_transfer.py ; workflow .github/workflows/semrepair-cross-robot-pusher.yml . Record exact Actions run ID, result, original result JSON and SHA in post-experiment report. Do not claim success until CI steps complete.
