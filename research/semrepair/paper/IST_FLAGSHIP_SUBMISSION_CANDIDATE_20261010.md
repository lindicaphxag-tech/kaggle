# When Can Action Software Be Repaired? Evidence-Indexed, Authority-Constrained Source Repair Across Stateful Robot Software

**Target journal: Information and Software Technology (IST) — ORIGINAL RESEARCH ARTICLE, WORKING DRAFT**  
**Dated:** 10 October 2026. **Status:** NOT SUBMITTED, NOT PEER REVIEWED. All pending measurements must be verified against actual CI artifacts before the submission version is frozen.

## Highlights — current journal manuscript candidates

- We identify the missing quantifier in source-level robot action recovery: all indistinguishable middleware versions must receive the SAME upstream action, not independently selected actuator outputs.
- We formalize action, historical-state and version-origin repair obligations, with a three-valued SAT / UNSAT / UNKNOWN verifier; source patch activation is refused when permissions or mathematical evidence are missing.
- Original-source experiments audit published LeRobot processor recovery (#3711) and reproduce independently reported ManiSkill actuator-sign error (#1469); two additional independently merged LeRobot defects (#4656, #4452) are staged for reproducible original-code validation.
- A finite typed AST repair grammar coupled to counterexample selection and independent symbolic contracts rejects weak-test-overfitting source edits, with an equal-information black-box competitor permitted to tie.
- Original MuJoCo/SAC task results retain the full reward and physical distance, the fact that only 16 starts were retrospectively reused for the signed one-field Pusher source replacement, and every failure/unknown authorization case.

## Abstract — structured for IST

**Context.** Learned robot policies depend on software pipelines for action scaling, relative coordinates, temporal state caches, and checkpoint-specific feature ordering. Middleware faults can remain invisible to tests focused on individual functions, while candidate patches can overfit small calibration sets or cross software ownership boundaries.

**Objective.** We investigate when source modification is necessary, permitted, and semantically justified compared with action-only compensation or recovery of missing observation history.

**Method.** We formulate bounded-input recovery as a minimax problem over source versions indistinguishable from the available observations. A repair planner evaluates explicit source-code and information-acquisition candidates using shared-upstream-action feasibility, source-specific software contracts, and authority constraints. For a restricted patch grammar, original-source execution, active distinguishing tests, and symbolic source-parameter checks are combined with fixed-version code identity. We report original vendor-code and software simulation case studies, including an independently merged LeRobot processor fix.

**Results.** In a real ManiSkill controller-source test, six of seven typed AST candidate programs pass the symmetric initial test, but five violate the declared asymmetric parameter contract. One additional source-level counterexample eliminates these five programs; the admitted source expression passes 128 reserved tests. In a separate signed-author config experiment using an externally published frozen SAC policy on official Pusher, changing one internal software-clip field matches the task reward and object distance of a three-field repair on 16 previously examined starts. A real upstream LeRobot processor repair restores the inspected cross-file checkpoint contract in six modeled configurations, with the two additional published upstream defects still requiring dedicated experimental CI confirmation.

**Conclusion.** The preliminary evidence favors separating source necessity, world identifiability, parameter-wide correctness, and permission from test-only patch plausibility. It does not establish general automated repair rates over industrial projects, trained VLA performance or production robot safety.

**Keywords:** automated program repair; software contracts; code provenance; robotic middleware; semantic verification; counterexample-guided testing; uncertainty-aware repair.

---

## 1. Introduction and the unsolved problem

Software recovery for an embodied policy is neither conventional robot feedback control nor conventional test-suite patch repair. The policy may output a correct bounded vector while a hidden middleware transformation reverses joints, silently discards the preprocessor associated with a pretrained model, clips intermediate commands, or uses the wrong cached reference from a previous action chunk. A policy-side workaround can sometimes invert the wrong mapping; other source mappings lose action reachability, making any upstream compensation inadequate. A correct action in a single trace does not establish correct software, and a publisher-signed source file alone does not establish semantic correctness.

This work studies a deliberately narrower question than general VLA fault recovery: **given a bounded family of possible middleware versions, observation/evidence policies, source-edit permissions and explicit behavioral contracts, which intervention, if any, may be authorized?** Our design acknowledges that obtaining original history or mapping information can make a black-box compensator as powerful as source repair. It therefore treats evidence acquisition and source edits as competing/compatible resources.

The distinctive empirical opportunity is public versioned robot software with independent recorded repairs. In addition to constructed actuator fault cases on official Farama Reacher/Pusher, we use genuinely published and hashed source versions: ManiSkill #1469/#1472 (reported sign error, proposed fix not yet upstream merged) and LeRobot #3425/#3711 (merged multi-file processor patch). Newly identified independent merged LeRobot #4656 and #4452 respectively concern temporal state alignment and a compensating joint-order permutation. No publisher merged patch is represented as our own achievement.

### Research questions and falsifiable expectations

**RQ1 (reachability and evidence).** Does a common bounded upstream action exist across observationally equivalent source worlds, and when does resolving version/history ambiguity change this feasibility?

**RQ2 (patch overfitting).** How often do program candidates that pass nominal tests still violate independently parameterized or cross-component software contracts? Can a small, explicitly costed distinguishing-test suite eliminate these candidates without consulting reserved tests?

**RQ3 (versioned natural defects).** Do original upstream source programs before and after genuine merged fixes violate or satisfy their actual dependency and state contracts under source-level execution, instead of merely matching manually reimplemented formulas?

**RQ4 (minimal intervention).** Given actor/issuer-permitted repairs and explicit edit/evidence costs, can the algorithm certify a minimum-cost feasible intervention or refuse unsafe alternatives, including action-only parity cases?

**RQ5 (evidence and overhead).** What are source-kernel execution counts, typed patch search completeness, proof-search complexity, aggregate task metrics and failure coverage? Do results persist across unrelated real upstream commits and varying source controllers?

Do not claim RQ1–RQ5 are fully answered until the outstanding workflows, across-project comparisons and negative outcomes are independently audited.

## 2. Methods

### 2.1 True common-upstream-source minimax

Let the source world x comprise the middleware implementation, version-specific mappings, clip limits, and hidden persistent state. Let h=H(x) denote the evidence visible to a runtime controller, and X_h the finite compatible world set. The controller sends one input u in the *actual upstream* legal box U. World x turns it into the real actuator command F_x(u). The desired actuator command in that world is a*(x).

The task-level distinction is important: F_x may be known only approximately; the present theorem concerns **immediate control error**, not total nonlinear task reward.

The robust action discrepancy is

    rho(h)=min_{u in U} max_{x in X_h} ||F_x(u)-a*(x)||_infinity.    (1)

For an authored two-stage linear-and-clipping contract,

    F_x(u)=clip_{B_x}(C_x clip_{L_x}(P_x u)).                         (2)

An earlier relaxed oracle optimized independently over physical outputs in each world and was insufficient to certify a single executable upstream action. Equation (1) corrects that overclaim. For known homogeneous mappings, the problem may reduce to an output-reachability optimization only after the upstream-domain preimages have been verified.

**Proposition 1: separate-output reachability does not imply one-input recoverability.** Take worlds x1 and x2 with F_1(u)=(u1,u2), F_2(u)=(-u1,u2), both indistinguishable to the controller, both with desired action (0.7,0) and u1 in [-1,1]. Each can individually realize its target with zero error. But every common upstream command has worst-world first-coordinate error at least 0.7, with equality at u1=0. Hence rho(h)=0.7. Selecting an optimal physical output separately in each world would falsely return zero. This is an identifiability/actuation obstruction under the stated observation set, not a previously unknown mathematics theorem.

For numerical certificates, we maintain a branch-and-bound queue of rectangular upstream input cells. The source's interval transforms of each cell provide a sound real-arithmetic lower bound on the worst-world error; an actually replayed center input provides an upper bound. Independent Lipschitz constants constrain unresolved boxes. A plan is provisionally feasible only when an executable input witness meets the tolerance in all compatible source worlds. The verifier refuses if the lower bound disproves feasibility and reports UNKNOWN if its finite search budget cannot establish either conclusion. Python float operations are not outward-rounded formal interval arithmetic; no industrial machine-precision guarantee is claimed at equality thresholds.

### 2.2 Observation refinement and licensed intervention selection

An evidence intervention e partitions X_h into source worlds distinguishable after that evidence becomes available. A source intervention c modifies version-specific maps subject to permissions and a validated patch contract. For each (c,e) candidate, solve (1) over each remaining observational equivalence class and take the largest certified discrepancy. Select the least **declared** cost among feasible allowed candidates only if every strictly cheaper alternative is proved infeasible or forbidden; a cheaper UNKNOWN candidate blocks the claim of a provably minimum-cost intervention.

**Proposition 2 (conditional monotonicity).** Holding source mappings and legal upstream inputs fixed, refining the observation partition cannot increase the ideal minimum worst-world error, because the controller can always ignore the new information and use its previous input. This conclusion does not imply that merely having a software source file is sufficient to authenticate the observation.

The finite selection engine does not attest manufacturer signing roots or install production files. Separate proof-carrying source activation tests bind exact source bytes, version epochs, source and calibration signatures, and declared actuator clip maxima, all under first-party experimental signing identities.

### 2.3 Bounded typed AST program repair and source contracts

A source patch grammar enumerates only modifications permitted for a known software component. For ManiSkill #1469 it includes seven scalar rotation-gain expressions, six of which appear plausible under a common symmetric rotation-range test. We execute the actual hash-pinned original vendor controller and scaling helper with real PyTorch tensors, choose additional source-parameter counterexamples, and use symbolic algebra to prove identity with the declared upper-bound rotation contract for independently varying lower<0 and upper>0. The available test subsets form a finite weighted distinguishing-test cover; dynamic programming returns a minimum-cost cover or refuses incomplete/over-budget catalogs.

This is **restricted AST grammar search**, not unrestricted LLM code generation or a novel CEGIS theorem. Crucially, the expected rotation sign is a stated semantic contract, not a fact learned from the held-out test set.

### 2.4 Cross-file and cross-temporal contracts on actual merged software

**LeRobot #3711**: the original relative-action training branch rebuilt a fresh processor even when loading a pretrained policy. The official merged patch preserves checkpoint-associated preprocessing, adds relative/absolute overrides, checks rename requirements for fresh training, and aligns a processor registry key. Our audited execution extracts original train setup AST before and after upstream merge and runs it on six source configurations with a stub for expensive model/processor factories. All three relevant upstream source files at both source commits are individually hash-verified. A new bounded four-operator grammar additionally builds sixteen candidate three-file variants *from the original source without copying the merged patch*, checks their actual source control flow and cross-file contracts, and compares the selected candidate with the real publisher patch only after source candidate generation.

**LeRobot #4656**: the publisher merged a fix for temporally stacked state input (B,T_obs,D) to action transforms. The correct reference is frame 0, matching the pretrained model input semantics, and processor caches must reset between batches. We compare source-level versions, plus six original-code AST mutants. An especially important negative control is using the last frame or temporal mean in both relative and inverse transformations: those implementations may pass an exact roundtrip test while violating the model-conditioned anchor reference. The final results require the dedicated CI.

**LeRobot #4452**: the publisher merged a fix for source compensation in synchronous inference, where joint names could be double-remapped relative to state/action feature order. We execute real source-method before/after code with a stubbed policy output and separately execute the actual downstream context alignment and fail-closed guard. Numerical action-level testing uses distinctive per-joint labels and separate checkpoint/hardware name orders; no bimanual robot platform is emulated in that test. Its workflow awaits verification.

**Proposition 3 (roundtrip inadequacy).** For any historical reference s, relative E_s(a)=a-s and absolute D_s(r)=r+s satisfy D_s(E_s(a))=a. This identity remains true when the reference s is the wrong historical frame. Thus roundtrip tests cannot establish whether the transform is consistent with the *separate model conditioning contract* that expects frame s0. This supplies a falsifiable reason to add cross-component obligations to patch validation.

## 3. Empirical design

### 3.1 Exact source versioning and denominators

Each independent upstream issue/PR is ONE externally documented software defect, regardless of the number of state tensors, action timesteps or case mutations. Individual tensor rows from one implementation are correlated numerical regression cases; report total numerical rows separately. Maintainer merged PRs are already public before these experiments; the protocol is retrospective, not prospective blind preregistration. Author-created source patch templates are *not* independent developer-submitted candidate fixes.

Record version SHA, file Git blob IDs, original Python source AST, chosen patch grammar, edited source file/AST-node count, configuration matrices, explicit source license assertions, and raw outcome per source version. Compare initial local plausibility and the complete contract, not only total number of passing tests.

### 3.2 Baselines (do not discard null or winning comparators)

B0 unchanged source; B1 nominal test-passing AST candidates; B2 a second asymmetric behavioral test without symbolic checking; B3 the finite least-cost counterexample-set selection; B4 + whole-domain declared symbolic contract; B5 + component-level version and registry contract; B6 + explicit source-authority refusal; B7 a fully informed black-box input correction; B8 the actual externally merged publisher source; B9 separately-allowed cached-history recovery; B10 optimistic independently selectable physical-output oracle; and B11 the corrected shared-upstream-input uncertainty solver. Not all baselines are applicable to all bug families; record applicability and prohibit fabricated aggregate scores.

For CPU probe search, distinguish time spent on typed AST candidate generation, original vendor PyTorch execution, symbolic proof, and branch-and-bound numerical certificates. The exact finite-test cover is exponential with the number of wrong candidates; results on n<=18 are not an industrial APR scalability guarantee.

### 3.3 Frozen policies and original robot tasks

Published SAC expert/medium checkpoints on official Gymnasium Reacher-v5 and Pusher-v5, native physics and original task reward remain the only executed external policy task-level study. The severe software fault is author-injected; no claim of released upstream robot source defect is made for that simulation. On 16 previously inspected Pusher expert starts, the signed author-level single-field clip-only repair yields -26.379423 mean original return and 0.175368 m mean object-to-goal distance, exactly the observed values of the three-field atomic repair; the defective source returns -56.395976 and 0.245791 m. **Edit-cardinality improvement 3 to 1 is not a risk reduction, general success gain or actuator manufacturer approval.** The software clip hard cap was author asserted. Strong finite-horizon native MuJoCo stateful MPC is included as a comparator but is not a full-horizon globally optimal feedback controller.

## 4. Results — VERIFIED as of this manuscript revision

| Evidence unit | Genuine external origin | Case count / scope | Observed and supported outcome |
|---|---|---|---|
| ManiSkill issue #1469 | real original upstream reported sign bug; proposed PR #1472 open | 128 actual source-kernel PyTorch actions | original mean L2 0.1718415663 against declared repair; one-node rot_upper repair max 0; equally informed input sign negation max 0. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38017412604 |
| ManiSkill active AST program grammar | real original vendor source, 7 AUTHOR templates | initial 1 symmetric test, one added test, 128 reserved asymmetric inputs | 6/7 initial test plausible, 5 wrong; active test eliminates all 5; declared-contract-valid patch passes 128/128 reserved tests and six wrong variants fail all. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38018321643 |
| LeRobot upstream merged #3711 | real publisher merged 2026-06-03, 3 files before/after, 6 Git blobs | 6 setup configurations, lightweight factory stub | three original code configurations fail model/processor source contract before merge and all six pass modeled contract after merge. Original AST code and guards are executed. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38018735755 |
| Signed-source Pusher config | original third-party SAC and MuJoCo; first-party fault and signing keys | 16 retrospective starts x 4 arms | 1 authored source field changed instead of 3; same original reward -26.379423 and true object-goal distance .175368 m in either repaired arm. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38015338180 |
| LeRobot original temporal caller fault | authentic LeRobot PyTorch function; FIRST PARTY injected caller error | 64 author-constructed observation-world pairs | mean best current-only worst-action error 0.233774 vs anchored zero; equal-history-information blackbox may tie. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016101114 |
| Six-mode geometric demonstration | author-created 2D source/action geometry | 2 hidden worlds | output-domain 0.7/0.2/0 lower-bound witness depending on geometry/info restoration. This is a RELAXED output geometry calculation, not a proof of common upstream executable control. https://github.com/lindicaphxag-tech/kaggle/actions/runs/38016915175 |

### Newly implemented, STILL UNVERIFIED in this manuscript revision

- Bounded input, true multiworld robust branch-and-bound plus a fail-closed UNKNOWN state: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38019762387 .
- Information + source intervention planner using that common-upstream verifier: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38019900047 .
- LeRobot #3711 sixteen actual three-file template program repairs against real publisher merged source: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38020299248 .
- LeRobot #4656 actual before/after stack source and wrong-but-roundtrip plausible variants: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38020563692 .
- LeRobot #4452 original merged source action-order regression: source and CI available in public Draft PR #82, but no observed pass/fail result at this paper revision.

**Never transfer these pending results into the verified table until their raw CI checks succeed.**

## 5. Relation to actual accepted research — and remaining novelty risk

**CAMP, IST 2026** (DOI 10.1016/j.infsof.2026.108282) and **TRACE, IST 2026** (DOI 10.1016/j.infsof.2026.108058) use rich diagnostic/feedback and repair templates in far broader program populations. TRACE evaluates Defects4J/DebugBench and reports 204 Defects4J fixes over many state-of-the-art comparisons. We make NO claim to beat their generic APR coverage.

**A comparative analysis of LLMs and LoRA in APR, IST 2026** (DOI 10.1016/j.infsof.2026.108238) separates iterative repair orchestration and LoRA on Defects4J/HumanEval-Java. Our state/authority contracts are of a different kind and no new trained repair model is included.

**JSS 2026 MENTOR** (DOI 10.1016/j.jss.2025.112690) combines MaxSAT fault localization and LLM-guided CEGIS on hundreds of actual student code bugs. Our bounded program family is much smaller; novelty cannot be claimed from CEGIS alone.

**FSE Companion 2024 Petke et al.** (DOI 10.1145/3663529.3663776) found typical per-bug plausible patch counts surprisingly small (median 2) for ten APR systems on Defects4J. This challenges the importance of inflated 5-of-6 synthetic overfit counts as standalone evidence. The real research need is failure of cross-component version/temporal semantics, not cherry-picked large plausible-patch inventories.

**RAS 2026 Luckcuck et al.** (DOI 10.1016/j.robot.2026.105648) supplies a real ROS Contract Language and runtime monitor generation. Monitoring alone is established prior art; our source intervention authorization must demonstrate actual repair benefits beyond that verification functionality.

**CI-Repair-Bench, 2026 preprint** (arXiv:2604.27148) contains hundreds of true CI failures and notes low actual generalized repair rates. It is a meaningful challenge for any field-wide automated repair claim; we have not evaluated on its 567 cases.

The novel research hypothesis here is not a new elementary theorem, but a potentially reusable **cross-observation-and-source evidence-indexed repair necessity and program authorization contract**. It will be validated only if actual independently developed source repairs show gains over test-only, contract-only and equally informed compensators.

## 6. Threats to validity, including falsifiers

- **Selection bias:** publishing only three curated LeRobot bugs and one open ManiSkill issue is too weak for general software repair claims; the true number of independent upstream repository maintainers is only TWO.
- **Program synthesis scope:** typed patches for #3711/#4656 are consciously constructed after public fixes existed, so searching such grammars is NOT discovery of unknown upstream corrections; the static after code must be hidden from the algorithm and used only for evaluation.
- **Source proof scope:** exact source Git blobs authenticate byte identity relative to the named commit, not the source owner's intended safety contract or a manufacturer signing chain.
- **Mask and metric choice:** author chosen linear action error can correlate poorly with contact task success or runtime safety; original robot score is reported only for native MuJoCo tasks.
- **Numerical honesty:** interval bounds are analytically valid under real arithmetic but floating-point code lacks outward rounding; a boundary UNKNOWN is safer than misreporting certified UNSAT.
- **Equal-information blackbox:** negative result MUST be retained when history is available or the sign/permutation defect is invertible; no unconditional source superiority.
- **Source version evolution:** different upstream commits may change function names, action mask design, registry names and source contracts; fail closed instead of silently matching fuzzy text patterns.
- **External validity:** source-level training processor factory is stubbed for memory/compute reasons; full pretrained VLA and a real robot experiment were NOT executed.
- **Reproducibility:** tests that download frozen upstream source require network; publish source checksums, a permitted archived offline cache, full dependencies and an independent clean environment before submission.
- **Statistics:** 96 torch regression samples on one public defect DO NOT mean 96 independent natural bugs. Do not calculate p-values by counting correlated tensor dimensions or related paired seeds as distinct independent robots.

## 7. Decision and submission gate

**Primary matching venue: IST regular research article.** Its 2026 archive explicitly publishes program repair, code tests, and CI/software quality studies. **Secondary: JSS regular research**, but its accepted code verification/repair methods have deeper formal generality. No claim of a guaranteed Chinese Academy of Sciences Zone-2 partition or particular acceptance percentage is made without authoritative edition-year verification.

**NOT SUBMISSION-READY UNTIL ALL are true:**
1. All five newly added CI workflows that exercise genuine publisher source and shared-upstream uncertainty pass with complete downloadable raw traces, or their negative results are analyzed and manuscript conclusions corrected.
2. The source repair generator actually produces runnable before/after PROGRAM PATCHES for at least two independent source-defect families, not just post-hoc audits of publisher merged PRs.
3. An explicit equal-information strong blackbox, test-only program selection, signed-only gate, semantic-contract-only gate, and full proposed method are compared on the same cases with real CPU time and artifact identifiers.
4. A second independent maintainer/repository family has a documented upstream-merged defect with runnable original-source regression, not only a proposed unmerged fix; otherwise frame it strictly as a narrow multi-bug case study.
5. Build a clean reproducibility package with pinned dependency list, sources, checksums, fixture licenses, full original raw result JSON, scripts, and an exact source-identity manifest.
6. Freeze once under the chosen journal's authorship, prior-publication, ethics, data and abstract/length guidelines. This work has not been submitted and must not be submitted simultaneously to IST and JSS.

Final editorial assessment: **more credible and deeper than the earlier single manufactured shear fault**, especially now that there are multiple authentic third-party code changes, but still **not comparable to the empirical breadth of leading IST/JSS accepted APR papers**. The rational goal is to eliminate falsifiable source-semantics overclaims, enlarge the actual natural-defect evidence, and submit when the methodological claim is reproducible rather than asserting a speculative journal acceptance timeline.
