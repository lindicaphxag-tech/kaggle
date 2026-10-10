# Intervention Surfaces Matter: Evidence-Indexed, Authority-Constrained Repair of Robot Action Software

**Journal of Systems and Software — targeted research-article draft, October 2026**

Status: **AUTHOR WORKING PAPER, NOT SUBMITTED OR ACCEPTED**. Results below are labeled as verified or awaiting CI. This article focuses on portable software engineering rather than claiming a new pretrained VLA method or real robot deployment.

## Abstract (research-article version, ~210 words)

Robot action middleware can fail in ways that are difficult to distinguish from policy errors: internal saturation limits which commands can be delivered, while stateful preprocessors may silently lose their associated policy checkpoint or historical reference. Existing test-based repair can also approve edits that behave correctly on a small calibration set but violate a broader software contract. We investigate an evidence-indexed repair decision problem combining three distinct obligations: a shared upstream action that works across observationally indistinguishable middleware versions, explicit authority to alter software, and source-level semantics valid outside the original test inputs. A bounded-input minimax formulation distinguishes world-specific physical actuator outputs from the optimistic assumption that the same chosen physical output is available in every hidden world. For a finite source-patch grammar, we combine distinguishing test selection, symbolic parameter-domain checking, and permission-aware refusal; for typed two-stage interfaces, we use interval search to produce feasible upstream witnesses or bounded-precision counterevidence. Original published ManiSkill source reproduces a reported rotation-sign defect; among seven candidate AST repairs, six pass symmetric calibration and five are invalid under asymmetric limits. One additional source-level probe eliminates all five incorrect candidates, with the contract-valid patch passing 128 reserved tests. We also audit the actual upstream-merged LeRobot cross-version processor repair, separately from model-training execution. Previously validated MuJoCo robot simulations demonstrate that a one-field authorized middleware configuration change can match a three-field rewrite across 16 retrospective Pusher starts. The evidence supports a constrained source-repair methodology, not generic VLA self-healing, industrial deployment, or an unconstrained automatic program repair solver.

## 1. Why this is a software engineering question

Prior robot fault recovery often intervenes at observations, desired controller output or learned action policies. Our object is a different intervention boundary: the **source-controlled transforms** between a pretrained policy and the physical command. The fix can belong to a source owner rather than the runtime policy maintainer. Two independently wrong transforms can cancel around nominal commands and create an unobservable failure under saturation; a correct temporal transform can also receive the wrong historical reference at its callsite, making actions history-dependent.

The proposed contribution is not the elementary algebraic fact of interval clipping, the Chebyshev center theorem, public-key signatures, exhaustive finite search or AST rewriting individually. The research proposition is that **a repair system must jointly constrain (i) what actions are physically deliverable; (ii) which hidden software worlds are observationally distinguishable; (iii) which source edits and evidence reads are authorized; and (iv) what complete program bytes are actually executed.**

The differentiation from generic monitoring is explicit: detecting a violated control invariant does not determine whether a permitted repair exists; and cryptographic signatures alone do not make the repair semantically correct.

## 2. Corrected formal problem and explicit quantifier order

**Important methodological correction (2026-10-10):** A previous simplifying formulation chose one physical actuator output independently of the uncertain source map. That may give a BLACKBOX an output it cannot actually achieve with its single bounded upstream command. The corrected program asks whether a **common pre-interface control input** is safe under ALL source worlds sharing the same observations. This correction changes the problem's quantifier order and is not equivalent to selecting a point in the union of physically reachable output polygons.

For the software state/world x, an upstream policy observing only h=H(x) emits the same u(h) for every x in the observation fiber X_h. Source world x applies its own actuator mapping F_x(u). In the two-stage interface,

  F_x(u) = clip_{[-B_x,B_x]^2} [ C_x clip_{[-L_x,L_x]^2}(P_x u) ].

Each u is restricted to the actual allowed **upstream** input box U=[-R_1,R_1]×[-R_2,R_2]. The correct robust immediate-action objective is

  rho_up(h) = min_{u in U} max_{x in X_h} ||F_x(u) - a*(x)||_infinity.

An output y independently chosen after the unknown source interface is physically executable only if a common input u can actually induce it in **all** compatible source worlds. Its relaxed output-domain optimum can be strictly more optimistic than rho_up. If all F_x are the same known controllable interface, the problem may simplify to minimizing over that interface's exactly reachable image; the simplification is **NOT permitted** for unknown F_x or when the input box has been omitted.

**Algorithm 1a — known-source geometric diagnostic:** For a fixed C and a trusted independently available interstage clipping domain, partition the doubly clipped output image into the nine piecewise-linear regions, intersect candidate action-error boxes with convex subregions, and compute a numerical relaxed output-distance witness. This remains a useful obstruction or optimistic comparator, but its witness is NOT a sound executable upstream action unless the original bounded-input preimage is separately constructed or verified. Source: research/semrepair/joint_information_reachability_operator.py. Previously validated diagnostic geometric example CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016915175 . Its six-mode 0.7/0.2/0 action-geometry values describe an authored RELAXED output model, NOT robust unknown-source control.

**Algorithm 1b — uncertainty-aware executable upstream optimization:** Maintain 2D axis-aligned boxes in the true upstream command domain U. For each source x, use interval arithmetic to enclose P_x U_box, the inner clip, C_x and outer clip. The maximum distance between desired action and its enclosed physical output interval is a lower bound on worst-world error. Independently obtain the Lipschitz lower bound from the measured box-center action loss and ||C_x||_infinity ||P_x||_infinity times the box radius. The max of both is a valid real-arithmetic lower bound. A box-center upstream command is always an actual feasible **upper-bound witness**, replayed through every F_x. Branch-and-bound on the lowest unresolved bound until a finite accuracy or budget limit: SAT if the witness is below the requested tolerance, UNSAT if the global lower bound exceeds it, and otherwise **UNKNOWN, refusing authorization**. CPU complexity can be exponential. Floating-point code does not provide outward-directed rounding or a hardware-grade proof near numerical margins. Code: research/semrepair/robust_multiworld_upstream_control.py; new dedicated CI pending verification.

**Algorithm 1c — information-versus-source permission synthesis:** Each information intervention splits the observation fiber X_h by exposing newly authenticated history/versions. Each source intervention changes F_x for all still-possible worlds. Exhaustively evaluate the FINITE licensed Cartesian product and run Algorithm 1b on every remaining observation fiber. Select the smallest declared-cost candidate only after ruling out every cheaper candidate as infeasible or unauthorized; if a cheaper candidate remains UNKNOWN after reaching the proof budget, withhold any assertion of minimum-cost completion. Source: research/semrepair/robust_observation_authority_optimizer.py; new CI pending verification. This does not cryptographically authenticate externally asserted permission bits or actually install source bytes.

**Cross-world falsification test:** World 1 F_1(u)=u and world 2 F_2(u)=(-u_1,u_2), with both targets (0.7,0). Each world independently has a zero-error actuator action. But **the same upstream u** must be used across the indistinguishable worlds, so rho_up=0.7; no claimed repair under tolerance 0.2 is valid. If trustworthy source version evidence excludes the incompatible world, an upstream witness near the desired target becomes possible. This case is FIRST-PARTY constructed and not a real robot rollout.

The solver's claimed certificates remain conditional on finite complete source-world sets, correct linear-clip source models, known action bounds, and numerical tolerances. These are not substitutes for manufacturer signing roots, real sensor authenticity, or future-horizon robot task guarantees.

### Algorithm 2: source-authorized finite-catalog repair choice

An approved catalog offers candidate source edits c with declared positive or zero costs, producer/controller mappings P_c,C_c, internal per-channel bounds L_c, and a required edit scope. For n=2..32 channels and a policy input hyperrectangle with radii R_j, the candidate must satisfy robust source semantics under exogenous coefficient drift bounds dP,dC:

  sum_j (|P_ij| + dP_ij)*R_j <= L_i for all internal channels i,

  sum_j R_j [
    |(CP-I)_ij| +
    sum_k (|C_ik| dP_kj + dC_ik |P_kj| + dC_ik dP_kj)
  ] <= epsilon_i for all outputs i.

Candidates must additionally obey the externally signed component edit mask and each trusted hard actuator/software clip cap. Exhaustive selection minimizes declared cost over the **supplied finite approved catalog only**. A non-empty catalog can still yield REFUSE if all candidates fail; no supposed universal synthesis of all Python programs is claimed.

Code: research/semrepair/authority_aware_repair_synthesis.py.
Green 2..32 channel and authored 7D catalog CI: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015187912 .

### Algorithm 3: real source-execution authorization

Patches to the authored actuator-adapter config require exact prior/replacement source hashes, actor/deployment/epoch/non-replay nonce binding, two distinct Ed25519 signers in a source-owner domain and two separate signers in a calibration domain. The calibration profile itself is pinned, binds to the exact proposed bytes, and contains an assumed error envelope plus a separately authorized maximum software clip. Only after semantic and robust verification does an atomic on-disk source replacement occur; the unchanged frozen SAC policy then executes commands through the loaded source config.

All signing identities are created by our tests; all calibration error intervals and signed hard maxima are first-party assumptions. This is not manufacturer approval, trusted sensors or a production filesystem adversary model. It demonstrates the algorithmic consequence of provenance assumptions, not their real-world establishment.

Successful ordinary source-activation CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38010872050 .
Successful dual-domain and coefficient uncertainty CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38014106998 .

## 3. Two real external software interfaces, two distinct semantic fault classes

**Case A — official Gymnasium robotics control/action middleware.** Frozen Farama/Minari SAC checkpoints have independently published weights and immutable model SHA256; Reacher and Pusher use genuine native MuJoCo physics and original task rewards. The reciprocal-shear plus intermediate clipping is expressly injected by the authors into the adapter. There is no claim that a Farama upstream issue contained this exact software bug. Because the source mappings compensate near zero, low-amplitude probes cannot certify the entire bounded control domain.

**Case B — Hugging Face LeRobot time-anchored relative actions.** The actual, unmodified publisher implementations of to_relative_actions and to_absolute_actions are loaded from commit b9cb121cb7d4e3e26ec5c906d08dda68d151ba52, with original Git blob SHA1 3405402904cf15ca18227b3a3fe006d7936f9d53 and verified original source byte SHA256 1c99f2a66649579cf6536d1970f71172b0b50bcb8decd226a9aa8969c0b0f171. We create a FIRST-PARTY caller error that replaces the original chunk-start reference with the current state; both broken and corrected calls execute the original PyTorch conversion function. This is **not** a discovered LeRobot upstream bug. The question is whether a current-only repair is identifiable without history.

Successful 64 pairs of indistinguishable historical worlds, each with two alternatives, 8 temporally dependent chunk steps: current-only minimax max absolute action error averaged 0.233774; the true correct-anchor caller has exactly zero numerical error. A blackbox that retained correct original chunk-start state CAN tie the source correction; strict source repair superiority is therefore false under equal historical evidence. Actual baseline:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016101114 .

Additional AST source patch work transforms exactly one positional source argument in an AUTHOR-INJECTED callsite, while executing the pinned actual vendor tensor conversion kernel. No production code or upstream vendor source is modified. Original-case CI **awaiting verification**.

## 4. Methods and actual executed experiments

### 4.1 Pusher minimal edit

On 16 previously inspected seeds under the same frozen SAC expert, four 100-step official simulator arms were evaluated. The artifact genuinely signs and installs a source config; it compares the original broken source, official health reference, a one-field clip-only source repair, and the original three-field matrix+clip atomic rewrite.

| Source variant | Authored source fields changed | Original official mean return | Mean true object-target distance (m) |
|---|---:|---:|---:|
| Broken adapter | 0 | -56.395976 | 0.245791 |
| Healthy external actor | 0 | -26.379423 | 0.175368 |
| Signed one-field clip-only | 1 | -26.379423 | 0.175368 |
| Signed three-field atomic | 3 | -26.379423 | 0.175368 |

**Observed source edit cardinality reduction: 3 to 1 (66.7%) at equal observed original reward and task geometry.** This is an *authored* interface where the intermediate clip may be expanded to 9.05 because a 9.1 software max was **assumed**. If the true hardware/software licensing cap is below 9.05, this patch is NOT admissible and a more invasive edit or refusal is needed. Do not call it universally safer or less risky just because fewer fields changed. Completed 64 native-task raw records:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015338180 .

### 4.2 Prior task-level external policy evidence

Two frozen SAC expert/medium checkpoints on each official Reacher-v5 and Pusher-v5 task were evaluated under the same planned five fault severities and six same-seed arms. A strong blackbox with exact fault parameters globally minimizes **instantaneous actuator deviation** over the actual doubly-clipped reachable physical output set. The source-restored policy outperforms that named comparator under severe planted faults; weak faults tie or sometimes slightly favor the comparator.

Pusher expert severe fault: original mean cumulative reward gain +8.738653 and mean true physical object-goal distance improvement 0.058898m versus exact instantaneous action projection; Pusher medium +5.544641 and 0.020738m. Original archive with independent 32-seed two-endpoint audit:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38009661288 .

Stronger blackboxes granted exact native dynamics, true physical state, planted interface identity and complete original reward. On 16 **previously inspected** Pusher expert starts, greedy one-step and SAC-in-the-loop three-step MPC return -47.696714 and -47.480364, compared with -26.379423 for the healthy/source-repaired reference; native rewards match simulator execution to 1e-7 and numerical parity at ~1e-16 for selected MPC actions. These are finite candidate, short-horizon baselines, not globally reward-optimal arbitrary blackbox policies. CI:
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013779442
and
https://github.com/lindicaphxag-tech/kaggle/actions/runs/38013970154 .

### 4.3 Source temporal information/fault interaction

A conceptual two-world pair using identical current observations and different historical anchors can be represented through the minimum physical control error operator rho(h). Source clip repair changes K(h); history read changes the world equivalence class. Full numerical six-mode ablation has code and tests, but must remain labeled **pending CI** until confirmed. This model experiment is not an end-to-end combined LeRobot+Pusher robot benchmark: it reuses established two-stage clip geometry with author-constructed world targets.

## 5. Editorial-level baseline and ablation requirements

To avoid an easy self-generated benchmark:
- A1: no repair vs unconditional atomic matrix+clip rewrite vs minimal licensed repair vs best privileged physically feasible action-only compensation.
- A2: signature-only vs signature+nominal semantics vs separate calibration-signature+drift-envelope vs hard-max cap and fail-closed constraints.
- A3: current observation only vs equal-information history-enabled blackbox vs source-anchor read/repair vs strict missing-trust abstention on the pinned LeRobot code.
- A4: full-action-domain certificate vs single numeric sample, one random point, or held-out numeric probe, including adversarial clipping boundary cases.
- A5: seven-source case in which clip-only permitted then physically HARD-forbidden; measures correct refusal, not numeric robot score improvement.

Evidence denominator always equals distinct task seeds or constructed world pairs, not individual 100-step time samples or 960 statistically independent robot bodies.

## 6. Most relevant *actual accepted* literature and positioning

**Nenchev et al., Journal of Systems and Software (2025), "Compositional code-level safety verification for automated driving controllers."** DOI 10.1016/j.jss.2025.112499. This is a much stronger independent CODE-LEVEL verification comparison: contracts plus bounded model checking, scalability to neural-network-based controllers and seven controller types, reported runtime in minutes. We currently do not match its reusable production verification or independent code case count. Our differentiated problem is licensed source repair selection and observation-bound recovery, not general software safety verification.

**Päßler et al., JSS (2025), "Analysing Self-Adaptive Systems as Software Product Lines."** DOI 10.1016/j.jss.2024.112324. A software product line/probabilistic formal model with an underwater robotic case study. Demonstrates that a software METHOD with a robot case can fit JSS without hardware, but only because the software-level contribution and model-checking framework are the main results.

**Luckcuck et al., Robotics and Autonomous Systems (accepted July 2026), "A compositional approach to verifying modular robotic systems."** DOI 10.1016/j.robot.2026.105648. This RAS paper contains an implemented first-order-logic ROS Contract Language, a reusable Vanda parser/monitor generator and a separate modeled robotic case study. It ALREADY solves important compositional contract verification issues. Merely adding signatures and an elementary interval inequality without verified code/source patches is not competitive with its formal breadth. Our candidate difference is repair intervention permission and information-physical joint obstruction; RAS is therefore a less precise lead journal for the PRESENT software claim.

JSS official aims require contributions to software engineering independently of a single application domain, and permit formal proofs, simulation, empirical work and negative results:
https://shop.elsevier.com/journals/journal-of-systems-and-software/0164-1212 .

RAS official scope includes theoretical/computational robotics but expects robotics contribution:
https://shop.elsevier.com/journals/robotics-and-autonomous-systems/0921-8890 .

## 7. Strongest remaining reject reasons and exact improvements needed

1. There are currently TWO external published SOFTWARE KERNELS/ENVIRONMENTS, but **not two independently documented genuine vendor bugs**. Both claimed fault-injection source cases are author-created. Need actual upstream before/after released implementations where the defect and fix are externally corroborated, or explicit JSS fault-injection-methodology framing with multi-project code mutation coverage.
2. The AST source repair handles one predeclared argument substitution. It is not currently a sound general source-program synthesizer, and general nD finite catalog optimization is exact ONLY relative to a manually supplied candidate set.
3. The two-world history ambiguity model assumes the entire possible historical-anchor set is given. In actual robots, learning this set and authenticating it would be an important separate problem. The source-read flag is not independent proof of valid issuer credentials.
4. Real-world signer root keys and calibration intervals are author-generated. Strong cryptographic tests are meaningful SOFTWARE TESTS, not evidence of manufacturer trust.
5. Shared publisher and same base family for the four external SAC actor checkpoints; no pretrained VLA, heterogeneous software maintenance team or real robot source failure evaluation.
6. CPU time, program analysis wall-clock cost, false authorization, timeout, certificate precision and AST-patch search coverage need measured cross-version controls, ideally on two independently maintained software repositories.
7. Novelty is a research direction, not a demonstrated top-accepted-paper contribution until independent code history and stronger generality are shown. No SCI-2/CCF-B acceptance probability is justified.

**Editorial decision:** concentrate on JSS *regular research paper* (software-source semantics and authorized proof-carrying patch choice). RAS is alternative if real robotics control recovery over independent non-Farama simulator families becomes the primary contribution. EAAI/Neurocomputing are poor fits absent a genuinely new learned repair model. No physical device is required by the research design, but broad independent reproducible evidence is.

Do not submit a duplicated version of the same findings simultaneously to JSS and RAS; freeze original source hashes, open scripts and truthful negative outcomes.


## 8. Newly discovered GENUINE upstream issue: ManiSkill #1469 (not a fabricated vendor bug)

This improves the preceding claim that all examined defects are author-injected: ONE REAL, publicly reported upstream software bug is now available independently of the Farama/LeRobot synthetic caller faults. Note that the author's role in filing the issue or preparing the open PR is not external maintainer endorsement, and no upstream merge is claimed.

- Original upstream issue: https://github.com/mani-skill/ManiSkill/issues/1469 , titled **PDEEPoseController scales rotation actions by rot_lower (negative), inverting commanded rotation direction**. The public report contains reproductions on ManiSkill 3.0.1 CPU / Panda PickCube and measured signed pose target and actual EE delta figures. Those task-level numbers are claims from the UPSTREAM ISSUE reporter, NOT independently measured by this new isolated kernel experiment.
- Upstream proposed PR: https://github.com/mani-skill/ManiSkill/pull/1472 , **OPEN, NOT MERGED** at this check. It proposes one-line replacement of rotational scale rot_lower by rot_upper; its tests describe sign and translation symmetry.
- Original unmodified publisher file pinned to exact main commit **62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3**, Git blob SHA1 **0f136173e9636461779654221f872dacd0a46d61**, path mani_skill/agents/controllers/pd_ee_pose.py. Real unmodified helper file **mani_skill/utils/gym_utils.py**, blob **5a3643ec837ae0a359bf20758663aebce43cf713**. Both original source bodies are extracted by AST and invoked using real PyTorch tensors, without installing heavyweight SAPIEN. This isolated callsite-level reproducer does NOT prove full sim task improvement.
- The developer-controlled repair operator changes EXACTLY ONE AST Attribute from rot_lower to rot_upper, conditional on source SHA, explicit source edit authority and the existing public PR number. It creates a FIRST-PARTY test candidate and does NOT change the real upstream source repository or imply maintainer approval. Dedicated source-kernel reproducer/tests: research/semrepair/maniskill_upstream_rotation_sign_case.py and test_maniskill_upstream_rotation_sign_case.py. CI was queued when this manuscript was revised; any positive case metrics remain PENDING.
- CRITICAL NEGATIVE blackbox control: for standard symmetric rot_lower=-0.1, rot_upper=+0.1, a blackbox that KNOWS the sign error can simply negate normalized rotational actions before feeding them to the original controller. It then exactly matches the one-node source patch at the actuator action output. Thus the original mistake is source CODE semantic inconsistency, but NOT a non-invertible actuator reachable-set obstruction. This counterexample is essential to reject any universal source-vs-action-only superiority claim.

**Updated ground truth taxonomy for JSS:** (i) actual reported ManiSkill controller sign bug with unmerged proposed upstream repair; (ii) author-injected callsite stale anchor using REAL pinned LeRobot conversion source; (iii) author-injected Pusher/Reacher software source clipping for true actuator noninvertibility. Three source families, but only **one upstream-reported natural issue** and zero verified external merges. Obtain a second independently reported bug/fix pair, preferably with a different root cause, before claiming a broad external software defect corpus.

**Additional accepted-paper standard:** JSS 2025 Nenchev et al. CoCoSaFe (DOI 10.1016/j.jss.2025.112499) verifies real code-level controller properties; RAS 2026 Luckcuck et al. RCL/Vanda (DOI 10.1016/j.robot.2026.105648) composes ROS assume–guarantee contracts. The credible novelty relative to them must be *minimal authorized executable software repair under exact information/actuation obstructions*, plus real before/after source evidence, not simply formal verification or monitoring.


## 9. Active test-overfitting controls and genuinely MERGED independent LeRobot source defect

The intended JSS/IST research claim is no longer just anomaly detection and source correction. It is the constrained choice of executable patches under scarce behavioral tests, whole-input semantics, source-code authority and cross-version component contracts. Test-suite overfitting is known APR prior art (see the 2024 Empirical Software Engineering xTestCluster patch-assessment work and the prior IST Crex paper); finite template-based synthesis and finite set-cover optimization are likewise established, not new mathematics.

### 9.1 Actual pinned ManiSkill: active counterexamples to plausible incorrect source patches

Original ManiSkill controller and original scaling helper are byte-verified at official git commit 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3, related to independently reported issue #1469 and upstream PR #1472 (unmerged). An explicit 7-term typed AST grammar replaces ONLY one rotation multiplier in the original source method: unchanged negative lower, positive upper, negated negative lower, absolute lower, hardcoded training upper=0.1, half of the rotation-range width, and averaged positive gains.

Under a symmetric training configuration lower=-0.1, upper=0.1, SIX candidates are plausible, but only the positive-upper variant satisfies the declared contract under arbitrary independently varying lower<0, upper>0. The other FIVE are test-overfitting candidates, NOT five genuine submitted vendor fixes. We submit:
- a real frozen ManiSkill original-Torch-kernel seven-candidate executable AST experiment;
- one adaptive counterexample chosen from a fixed bank of asymmetric calibration configurations;
- exact minimum-cost distinguishing probe selection over the finite candidate bank via weighted set-cover DP (independently verified against brute-force);
- a symbolic polynomial identity verifier for the independently varying rotation bounds;
- 128 reserved asymmetric configurations and actions not accessed by patch selection;
- license-check refusal even when the correct patch is known but not explicitly authorized.

Code: research/semrepair/active_contract_patch_synthesis.py and minimum_counterexample_probe_cover.py plus tests and GH Actions. The pure-stdlib source-candidate test-cover solver PASSED five local container tests, including exact-optimum cross-check and missing-budget refusal. Full original upstream ManiSkill kernel GitHub CI is not yet verified green. The intended within-grammar symmetric-test false plausibility (five incorrect programs) is not a generic automatic program repair success rate or externally validated robustness result.

### 9.2 Genuine independent publisher MERGED fix: LeRobot #3425 / PR #3711

Upstream issue https://github.com/huggingface/lerobot/issues/3425 documents that rename_map was skipped for pretrained relative-action training when the policy training path silently rebuilt a FRESH processor pipeline. Preliminary upstream PRs #3428, #3511, #3512 and #3514 CLOSED without merging, proposing variants that forwarded rename_map to freshly created processors. Later upstream PR https://github.com/huggingface/lerobot/pull/3711 MERGED by actual LeRobot maintainers on 2026-06-03, with commit 19fe315971079e116b917ed4d482bfbf3c93bbbe. This is NOT our patch and its maintainer acceptance is NOT recognition of SemRepair.

The externally merged repair preserves the pretrained actor+processor checkpoint contract and enables relative/absolute actions via named processor overrides. It also guards rename_map when no pretrained checkpoint exists, and renames the actual step registry key from delta_actions_processor to relative_actions_processor. Thus a naive patch that only forwards rename_map can satisfy a superficial data-key test while violating provenance and checkpoint pipeline semantics.

We added an independent no-hardware, no-GPU executable source audit with original LeRobot Python training setup statements AST-extracted from the before-merge git commit 906b585826c0e34235a347554abfb94e0d65484b and the accepted merge SHA. All SIX original Git blobs for three distinct files are pinned: scripts/lerobot_train.py, configs/train.py, processor/relative_action_processor.py. It executes the unchanged upstream processor setup code across six configuration cases and checks the cross-component policy/processor contract before and after merging. Expensive pretrained checkpoints, policy inference and actual factory behavior are STUBBED; this is a source-level control-flow case study and cannot be reported as independent complete PI05 training success. New program and regression workflow: research/semrepair/real_merged_lerobot_processor_contract.py and .github/workflows/semrepair-real-merged-lerobot-multifile-contract.yml. GitHub CI was not yet verified when this document section was written.

### 9.3 Publication strength and honest remaining gaps

The current original AST repair search works for a declared one-expression grammar; it does NOT yet synthesize the multi-file maintainer merged LeRobot fix. The LeRobot tool AUDITS genuine before/after source, not automatic discovery of the maintainer patch. Both distinctions must remain explicit.

Prior top JSS formal software papers such as Nenchev et al. CoCoSaFe (DOI 10.1016/j.jss.2025.112499), and general program-repair methods on Defects4J, test far more actual software systems than one real merged bug plus a controlled fake caller mutation. IST remains a plausible first submission target based on official software-testing, maintenance and empirical evidence scope; JSS remains higher generality stretch. Neither promises 2026 CAS 2, guaranteed acceptance nor an immediate November/December acceptance. Acquire a second genuinely externally resolved bug in another codebase plus CPU-cost and repair false-authorization audit before labeling this a top-paper-level general methodology.


## 10. Actual conditional minimum-cost joint repair operator and updated validation statuses

Previous descriptions defined two obstructions (the inability to output a required actuator command, and the inability to infer a hidden historical target from current observations), but did not yet provide a **single algorithmic decision rule** to choose between them under repair authority and expense.

This update adds a finite-catalog, cost-minimizing *joint* repair planner:
- Input: finite set of target 2D actions in observation-indistinguishable worlds; candidate software actuator interfaces, each with a full polygon UNION for the actually reachable control actions; permitted source changes and declared costs; candidate information/evidence interventions that partition the possible hidden-world set, each with authorization and cost; maximum allowed worst-world action discrepancy.
- For each authorized (source,information) pair, compute the previously verified minimum worst-world L-infinity actuator discrepancy by intersecting EVERY world's radius-r action box with the TRUE nonconvex-union actuator image. When historical information reveals the actual world, solve separately for each new observation fiber; require the maximum over ALL such fibers to be within the declared tolerance.
- Choose the certified intervention with minimum declared joint software+information cost over the entire given finite catalog. Hard refusal if no licensed plan meets the physical discrepancy bound. No source bytes are automatically written by the selector; separately signed source/config, calibration and version attestations must still be checked before deployment.

Implementation: research/semrepair/finite_authorized_joint_repair_planner.py, tests test_finite_authorized_joint_repair_planner.py and independent GH workflow semrepair-mincost-information-action-repair.yml. The algorithm is exhaustive over a small finite intervention product and uses the *already tested* conditional polygon Chebyshev solver, not a novel theorem that generalizes to unknown nonlinear closed-loop robotic tasks.

The illustrative DOUBLE fault uses the same FIRST-PARTY two-dimensional shear+clip geometry and two hidden historical target worlds from the earlier verified six-mode test (GitHub Actions https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016915175). With a DECLARED source clip repair cost of 2 and independent historical evidence cost 1, the intended policy cases are:
- loose tolerance 0.8: no software modification or history acquisition (error bound 0.7, declared cost 0);
- medium tolerance 0.25: only one-field source clip repair (error bound 0.2, declared cost 2);
- strict tolerance 0.05: source clip + true historical world evidence (error bound 0, declared cost 3);
- strict tolerance and no source edit permission: REFUSE.

**As of authoring:** this new joint decision CI is queued, so the exact decision/cost values above are expected analytic test assertions, NOT newly confirmed experimental CI results. The underlying six-mode geometry and witness coordinates did pass separate CI. The separately run finite catalog min-cost source optimizer for 2..32 channels and 64 Pusher replay source edits have passed previous GH CI.

**Latest genuine upstream controller experiment now verified:** original ManiSkill fixed-code commit 62ff3a... sign error executed in unchanged genuine publisher Python function/helper, with 128 rotation samples and five green tests: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38017412604 . Original mean L2 difference to proposed corrected actions = 0.1718415663; one-attribute source repair max error 0, and same-information action negation max error 0. This REAL source is from an OPEN reported issue, but PR #1472 is still NOT MERGED, and no SAPIEN task was run. The negative control matters: source repair is NOT fundamentally needed for an invertible sign-only actuation fault.

**Latest genuinely merged LeRobot version case:** external LeRobot PR #3711 merged June 3, 2026, entirely separately from our project. Three exact before/after upstream file blobs and actual AST-executed train setup/guard code have been committed to local GitHub tests. Real factory behavior remains STUBBED, no full 4B VLA run. The new workflow was queued when this section was authored; do not claim the cross-file contract experiment is green until its CI verifies the statement.

**Strong journal target:** IST for evidence-indexed source program repair and testing; JSS for a substantial upgrade only if this becomes a general reproducible repair methodology across independently maintained codebases and true software version history. Existing APR works on Defects4J and JSS code-level formal papers have MUCH larger independent real-defect coverage. Empirical gains on authored two-world mathematics are mechanism evidence, not significant task performance generalization across robots.
