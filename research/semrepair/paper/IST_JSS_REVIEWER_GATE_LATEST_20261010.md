# SemRepair — IST/JSS reviewer gate and evidence ledger (2026-10-10)

**Recommendation:** draft as *Evidence-Indexed, Authority-Constrained Repair of Stateful Robot Action Software*. Practical current journal scope fit: **Information and Software Technology (IST)** first; **Journal of Systems and Software (JSS)** as an ambitious general software-engineering alternative. Neither venue accepts generic "VLA self-healing" marketing without a real learned VLA algorithm, and neither offers guaranteed acceptance or a known CAS partition from this code.

## Highest defensible contribution

A source repair is justified only if a full-action model under current observation evidence proves a relevant behavior cannot be reproduced by a cheaper allowed intervention. The method therefore needs (1) physical command-reachability proof, (2) observation-information equivalence/fiber analysis, (3) allowed source-owner repairs, (4) active program-test counterexample search, (5) whole-declared-domain semantic admission and (6) actual pinned source-byte/version identity. Actual external signing/calibration remains hypothetical in these test harnesses.

Finite-catalog minimum-cost repair and exact counterexample test synthesis are **not new mathematical optimization theorems**. Literature on APR patch overfitting predates this manuscript. The research question is whether their careful composition across independent real source defects yields a genuinely reusable code-level repair method.

## Independent prior published work

- JSS 2025 Nenchev et al., *Compositional code-level safety verification for automated driving controllers*, DOI 10.1016/j.jss.2025.112499 — genuine code-level, multitype-controller formal verification.
- RAS 2026 Luckcuck et al., *A compositional approach to verifying modular robotic systems*, DOI 10.1016/j.robot.2026.105648 — real ROS Contract Language and Vanda monitor generator; formal compositionality alone is established prior art.
- Empirical Software Engineering 2024 xTestCluster: https://link.springer.com/article/10.1007/s10664-024-10503-2 — plausible test-passing patches can still be wrong; independent behavioral patch evidence is essential.
- Real published robot VLA 2026 ICML VLA-FixBench: https://proceedings.mlr.press/v306/yan26r.html — 20 VLMs and real robots; our present work is NOT comparable in VLA coverage.

## Actually verified results (do not inflate these denominators)

| Mechanism | True execution scope | Observed value | Evidence |
|---|---|---|---|
| Licensed source edit synthesis | author model 2..32 channels | minimum licensed finite-catalog cost/rigorous refusals | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015187912 |
| Physical signed config activation | original official Pusher SAC 16 retrospective seeds x 4 arms | 3 source fields -> 1 with same observed -26.379 original return and .175368m object-goal distance | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015338180 |
| Actual reported open ManiSkill #1469 | immutable authentic controller + helper, 128 ORIGINAL source function calls | original mean rotation L2 defect .1718415663, 1-AST-attribute fix zero, equal-informed blackbox negate rotation also zero | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38017412604 |
| Separate HF LeRobot original conversion API | original source SHA-pinned, 64 authored two-world pairs | current-only minmax max error .233774, verified historical source zero; equal-history blackbox may TIE | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016101114 |
| Orthogonal physical reachability vs hidden information | 2D author-created geometry / two worlds, not native robot task | neither .7 / source-only .2 / history-only .7 / both 0 | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016915175 |
| H=3 privileged actual Pusher robot MPC | 16 previously observed genuine native MuJoCo seeds, 30,400 model branches | source-reference -26.379 vs finite three-step oracle -47.480, but NOT full 100-step global optimum | CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013970154 |
| Exact weighted distinguishing-probe solver | local standalone Python, 5 unit tests | passed including independent brute-force optimality and cost cap refusal; illustrative one test of assumed cost 1 eliminates 5 wrongly plausible templates | LOCAL execution only; dedicated GitHub CI QUEUED |

## New real published and MERGED upstream program-fix case — CI pending

External HF LeRobot [issue #3425](https://github.com/huggingface/lerobot/issues/3425) has a real fix [MERGED PR #3711](https://github.com/huggingface/lerobot/pull/3711) on **2026-06-03**, merge SHA **19fe315971079e116b917ed4d482bfbf3c93bbbe**. Old code silently forced processor_pretrained_path to None for relative action training, potentially losing rename_map and the paired policy-processor checkpoint contract. The merged change preserves the pretrained pipeline and uses named overrides; config and registry source were changed in the same merge. Preliminary PRs #3428/#3511/#3512/#3514 CLOSED without merge, documenting competing partial approaches. This paper DID NOT author or merge #3711 and gets NO credit for its upstream recognition.

Original FULL Git blob files at pre-merge 906b585826c0e34235a347554abfb94e0d65484b and post-merge commit are pinned for THREE actual source paths; an AST harness executes original before-and-after training setup statements and actual merged config guard against SIX declarative software configurations. Pretrained VLA checkpoint/factory internals are STUBBED, not a real 4B-parameter training success. Implementation:
research/semrepair/real_merged_lerobot_processor_contract.py.
Pending CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38018735755 .

## New research method implementations (submitted, do NOT call passed yet)

1. Typed actual ManiSkill source AST patch grammar (7 candidates); a symmetric calibration test overfits to 5 incorrect variants; active asymmetric source probes plus independently varying positive rotation-bound symbolic contract, 128 reserved asymmetric control inputs. Source writes need explicit candidate license. Implementation active_contract_patch_synthesis.py, CI https://github.com/lindicaphxag-tech/kaggle/actions/runs/38018321643 .
2. Exact minimum-cost finite source-program counterexample probe selection by DP+coverage witness, budget/incomplete-banking fail closed, plus synthetic 5..18-hypothesis CPU timing program. Local tests **5 passed**; genuine GitHub source integrated CI queued https://github.com/lindicaphxag-tech/kaggle/actions/runs/38018321717 .
3. Finite-catalog source/evidence global decision over **physical reachable polygon union × hidden history observation partition**. Given licensed programs and error tolerance: choose no intervention / source clip / source+history, or REFUSE if no valid licensed pair. Formal geometry core has independent green tests; new decision solver test is queued https://github.com/lindicaphxag-tech/kaggle/actions/runs/38019206791 .
4. Temporal LeRobot source anchor permission logic: an earlier CI FAILED because it accepted a contradictory verified historical witness while selecting a cheaper current-only control. Code now validates claimed source and cache witnesses BEFORE selecting interventions. Until the updated CI passes, no claimed zero false-authorization rate.

## Local CPU scope (machine-specific, SYNTHETIC incidence only)

Exactly priced test-cover search timings in the author's available Python environment, median of three repeats: 5 wrong patches/11 probes ≈0.101ms; 8/17 ≈0.236ms; 12/25 ≈2.623ms; 16/33 ≈19.514ms; 18/37 ≈36.586ms. These do NOT include original vendor PyTorch executions, network fetching or arbitrary program repair. Worst-case DP is exponential and capped at 18 wrong candidates. Reproducible script benchmark_counterexample_scaling.py.

## What still blocks a high-confidence JSS/IST accept

- Only one externally MERGED natural bug currently included and one externally reported OPEN bug (ManiSkill #1469 with PR #1472 not merged). Both source ecosystems are robotics, and there is no large external independent repair benchmark like Defects4J; top APR contributions evaluate many real defects.
- The natural merged LeRobot patch is AUDITED, not automatically synthesized by our algorithm; the actual automatic AST synthesis experiment has a tiny hand-specified 7-term ManiSkill grammar and first-party controller contract.
- No legitimate external manufacturer signing root, physically attested software clip calibration, independent maintainer adoption of SemRepair, or full upstream production patch installation. Authored test keys are not independent third-party signatures.
- Real robot tasks under ROS/SAPIEN and frozen vision-language-action models have not been executed for these new source regression cases; a native Pusher two-joint fault is not enough for general robot policies.
- Stronger CPU/probe-budget empirical ablations and failure cases for a second independently merged issue are needed; release a clear code artifact and journal-compatible licensing/reproducibility package.
- Journal indexing/CAS zones and acceptance times require source-specific contemporary verification; GitHub CI and paper drafting do not imply acceptance.

**Paper status:** DRAFT. Main public https://github.com/lindicaphxag-tech/kaggle/pull/82 . Avoid simultaneous substantially overlapping JSS/IST submissions.
