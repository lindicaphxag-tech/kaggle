# SemRepair vs published robotics research — reviewer benchmark (2026-10-10)

**Venue decision:** IEEE Transactions on Robotics (T-RO) is a *stretch*, not currently submission-ready. T-RO has required genuine advances in robotics, and a regular first submission <=18 two-column pages, including evidence, figures, tables, references; abstract <=200 words; it does **not** review separate supplementary text/code as an extension of the scientific manuscript. Policy: https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/

## Exact relevant papers — not generic "similar titles"

| Status / published work | What that work already contributed | Requirement for our manuscript |
|---|---|---|
| **AAAI 2025**, *Self-Correcting Robot Manipulation via Gaussian-Splatted ...*, https://ojs.aaai.org/index.php/AAAI/article/view/34866 | Future-state prediction, detecting deviation, embodied self-correction. This is task-level manipulation recovery, not merely patch consistency | Show actual policy/task-level recovery, and prove our source-authorized repair is needed in compensating-semantic cases rather than just invoking existing closed-loop recovery |
| **T-RO published/plenary-listed**, *Task and Motion Planning for Execution in the Real*, https://www.ieee-ras.org/publications/t-ro/t-ro-list-of-conference-papers/ | State-of-the-art expectations include whole robotic task and execution in physical realities, not only isolated finite examples | Multi-task native runtime, unseen robot/controller family, policy outcomes and cost/risk, not 4 parameter worlds only |
| **AISTATS 2014** Javdani et al., *Near Optimal Bayesian Active Learning for Decision Making*, https://proceedings.mlr.press/v33/javdani14.html | Active information gathering can stop at a sufficient decision region instead of identifying all latent hypotheses | **Do NOT claim** singleton repair-decision-region stopping or optimal decision-tree search invented by SemRepair. Match this paradigm as a strong baseline |
| **ArXiv 2024 preprint (NOT asserted accepted journal)**, *Failure Diagnosis for Improving Manipulation Policies (RoboMD)*, https://arxiv.org/html/2412.02818v2 | Automatic failure-mode diagnosis in robot manipulation; competing notion of active failure explanation | Report policy recovery and failure-mode identification separately from implementation-correct repair |
| **OOPSLA 2025 PRISM** (adjacent software semantics; not a robotics result), https://sigplan-pages.sigplan.hosting.acm.org/OpenTOC/oopsla25-2.html | Wrong plausible code patches survive task tests; implementation semantics matter | Do not treat source-code patch correctness as a brand-new concern |

## Head-to-head audit of present evidence

| Dimension | Our current true source | Honest score |
|---|---|---|
| Exact finite identification and minimax | 4 known worlds, 3 held waveforms; independent Bellman cost = method cost **4**, no advantage | Mathematically demonstrated; **not** better than active diagnosis |
| Real simulator authorization | MuJoCo x12 cells: full-tap 12/12; external 3/12; risk-restricted 6/12 | Author-run CPU physics; same small family, not independent embodiments |
| **Evidence authority** failure | MuJoCo x6 force-tap substitution cases: naive 6/6 wrong implementation, untrusted-source refusal 6/6 | Positive *falsifier*, NOT physical authenticated instrumentation |
| Incorrect-interface repair trajectory success | MuJoCo 72 dependent condition cells with compensating wrong patches | Strong mechanism of non-identifiability; no task-performance advantage |
| Proof-carrying patch bytes | Combined belief + trust whitelist + concrete patch/evidence/dependency SHA read | Integration tests GREEN; attacker spoofing trusted flag remains open |
| Real high-dimensional robotic policy | ManiSkill original demo conversion 6/8,6/8,0/8,6/8; diffusion policy tiny two-step 0%/0% | Real negative result, no recovery improvement |
| Genuine generalization | Unknown controller family or unseen policy | **MISSING** |
| Independent acceptance/third party | External maintainer retained feature or external user reproduction | **MISSING** |

Sources: https://github.com/lindicaphxag-tech/kaggle/actions/runs/37984576868 and https://github.com/lindicaphxag-tech/kaggle/actions/runs/37985266948; original 12-cell run https://github.com/lindicaphxag-tech/kaggle/actions/runs/37975041561; original 72-cell source https://github.com/lindicaphxag-tech/kaggle/actions/runs/37973635217.

## T-RO-shaped manuscript, not a slide deck

**Claim and writing sequence:**
1. Introduction (one concrete wrong-patch case; separately state why correct trajectories are insufficient).
2. Related work (AISTATS decision regions, robotics active perception, robot self-correction, semantic program repair); distinguish source attestation as assumption not contribution.
3. Problem (latent world/repair region, risk/admissibility, patch and dependency provenance, abstention).
4. Main observation-equivalence propositions and limitations (short proofs in main text).
5. Method (frozen sources, policy, trusted observation provenance, byte verification and the unavoidable spoofing boundary).
6. Experimental protocol (MuJoCo known-family and hidden-unseen controller family; strong active baselines).
7. Results **four distinct outcomes**: wrong authority, abstention, physical closure, actual learned-policy task success. Include false-approval cost, discovery time, and risk.
8. Related negatives, failure cases, open-source release and limitations.

**Main figures:** (F1) *one physical look-alike, two repair identities*; (F2) causal system diagram and observation boundaries; (F3) trust-spoof fail/abstain timeline; (F4) unknown-controller generalization; (F5) complete competing-method risk–coverage curve. No fabricated source photos or synthetic plots passed off as real robots.

**Primary results table:** four independent unseen controller families x official downstream policy tasks x source-bound/provenance-aware decision policy / optimal adaptive diagnosis / generic information gain / no repair. Row unit is *controller family/task episode*; 72 or 12 known construction cells are not independent robots. Include per-family success + 95% uncertainty intervals; independent model-source provenance.

### Fatal review questions
- What is new beyond established **decision-region active learning**? Invariants on evidence-origin and concrete repair-object identity must add information or outcomes unavailable to strong diagnosis competitors.
- Is the "trusted source" independently verified? **No**. Calling a Python Boolean trusted does not solve adversarial authenticity.
- Is there real task improvement? **Not yet**.
- Are abstract and figures paper-ready? Not until unseen-family study delivers an affirmative result.

**GO threshold:** genuinely unseen family + predeclared matched baseline with substantive task recovery and verified observations; otherwise a scoped SE/robotics systems paper on compensated errors, **not** falsely T-RO-ready.
