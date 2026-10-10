# Causal Evidence Closure for AI-Assisted Patch Review
## Working JSS regular-research manuscript — 10 October 2026

**Status: research draft, NOT submission ready.** The full-paper hypothesis is under construction. There is no independent patch-correctness improvement or journal acceptance claim.

### Abstract

A difference-exposing execution establishes a local behavioral distinction, not that a proposed fix satisfies all intended requirements. In modern tensor software, a patch can fix device placement while leaving dtype, gradient, module-state, aliasing, optimizer or shape obligations violated at an unchanged consumer. We investigate Causal Evidence Closure (CEC), which integrates (i) a source-pinned typed semantic-effect IR, (ii) provenance-labeled obligation hypotheses derived separately from the change, (iii) native replay with minimal counterfactual interventions and preserving controls, (iv) fail-closed scope checking, and (v) budget-aware experimental acquisition. The implementation already supports native pinned source replays and a conservative structural checker; an eight-repository frozen transfer retrieval successfully generated typed effect *hypotheses* following a documented API-transport retry. This is not a benchmark of patch correctness. A falsifiable evaluation must show a positive-coverage reduction in wrong whole-PR authorizations, independent specification adjudication and cost/utility gains over strong baselines.

### 1. Originality boundary

The scientific question is not "why passing tests are incomplete?" Existing differential testing and formal-verification methods have established that. The proposed investigation asks whether **authentic provenance + typed effect closure + executable falsification + explicitly limited inference** can *materially reduce incorrect reviewer decisions* without reducing coverage or multiplying reviewer workload.

The distinctive research contribution must be demonstrated by full integration and blinded performance, not merely relabeling observations as certificates. Any root derived from a BEFORE use is a **candidate requirement**; it is NOT automatically a faithful specification.

### 2. Proposed executable method

**TSE-IR:** Parse immutable before/after source and emit nine families of typed effects: state, gradient, precision, layout, masking, optimizer, numerical domain, device and representation. Source digests, function boundary, changed calls and pre-change parameters are recorded. Code: research/pcpr/source_effect_ir_v05.py. Current syntax tags are deliberately broad and may be spurious.

**CEC provenance DAG:** Vertices carry source origin, semantic facet and digest. Edges carry a typed dependency claim, verified/unknown status and trust origin. Patch-added assertions cannot self-justify a correctness claim; cycles, altered context hashes, missing negative controls and unsupported inferential promotion cause abstention. Code: research/pcpr/causal_evidence_closure.py. Its local digests are NOT secure external Git attestation.

**Native counterfactuals:** Intervene on one matched AST mechanism (drop dtype keyword, remove clone, drop requires_grad forwarding, demote register_buffer), then run pinned original source code and the synthetic wrong control under matched input. Refuse ambiguous mutations and preserve explicit non-target behavior. Code: research/pcpr/semantic_interventions_v06.py and run_v06_native_counterfactuals.py. These are **known development cases**, not blind incorrect patches.

**Claim scope:** A measured observation is indexed by requirement, input context, actual runtime tier, source hashes and budget. A local result must never be promoted automatically to whole-patch correctness. This finite-testing non-promotion fact is known logic, not a novel theorem.

**Cost-aware experiment scheduling:** For a deliberately finite hypothesis family, adaptive_scope_planner.py computes the exact worst-case-minimizing decision tree, with a fixed-order comparator. Future novelty requires robust cost modeling, actual uninspected case acquisition and comparison against strong adaptive baselines, not just this known minimax dynamic program.

**Anti-vacuity metrics:** selective_review_eval.py requires independently adjudicated ground truth, positive predeclared coverage and matched PR cohorts. An always-abstain policy has zero wrong authorizations but fails the coverage gate.

### 3. Evidence matrix — no denominator pooling

| Evidence unit | Observed | Interpretation |
| --- | --- | --- |
| 12 PRs, original locked v7 | 0/12 witnesses, 12/12 abstentions | Negative generalization result |
| Six new PRs, locked v0.1 | 0/6 source candidates | Failed narrow semantic transfer |
| Eight new PRs, locked v0.2 | 0/8 effect-level native probes | Failed four-template transfer |
| Eight different repos, locked v0.5 | Initial 8/8 fetch failures; authentication-only same-model retry: 8/8 typed semantic *hypotheses* | Source candidate discovery only; 0 native witnesses, 0 correctness decisions |
| DeepChem #5190 | 14 native field registration observations over two classes; 13 field abstentions | Correlated development units, not 14 PRs |
| Kornia #5446/#5005 | Actual SHA-pinned package API replays with observed state/shape effects | Previously inspected development cases |
| Kornia #5026, SenseVoice #358 | Original pinned function AST native output/dtype/CTC effects | Native function-tier development replays |
| MONAI #9103 | Original pinned BarlowTwinsLoss methods; in float16/bfloat16 original and one-edit counterfactual produce float32, fix preserves half precision; float32/float64 controls unchanged | Retrospective external-repository class-method replay, not full MONAI package and not new blind PR |
| Synthetic minimax planner | 48 worlds, mean worst-cost 5.104 vs fixed-order 10.563 | 51.7% reduction against one weak synthetic comparator only |
| CEC checker | Structural provenance, contradiction and non-promotion tests passed | Implementation behavior, not APCA accuracy |

Generic single-edit native counterfactual CI: 3 previously inspected pinned examples passed (run 38029919629). Separate MONAI native class-method replay passed in four CPU dtype contexts (run 38030555883); these are development examples, not a prospective correctness detector. Do not retroactively upgrade prior locked experiments.

### 3.1 New high-scale transfer falsifiers (same research day; PR-level denominators)

**Sequential temporal-method holdouts; the stages are not interchangeable.** The 48-PR stage was selected and locked before opening its patch content for the v0.7 method; the 64-PR stage was newly locked after the v0.7 failure and before opening its patch content for the v0.8 method. The 32-PR third stage was locked before its v0.9 first inspection but its original retrieval was blocked by HTTP errors. These strata were searched from title-keyword-enriched, already merged PRs; they are **not a randomly sampled representative software population, nor independently labeled ground-truth repairs**.

| Temporal stage | Frozen method, independent unit | Authoritative first-run result | Valid inference |
| --- | --- | --- | --- |
| Stage I (48 distinct PRs, 47 repos, six effect groups) | v0.7 exact factory-dtype AST replay, before/after Git SHA, up to eight edited Python files | **0/48 PRs with a successful native operator observation**, 38/48 with v0.5 syntax candidates, 4/48 unsupported; all PR metadata retrieved | Local factory-only model refuted as a broadly runnable patch evidence method on this corpus. [CI 38031785312](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38031785312) |
| Stage II (64 different PRs, 40 repos, eight effect groups) | v0.8 symbolic module-field/local dtype replay and unchanged BEFORE caller graph | **0/64 PRs with a successful native operator observation**, 52/64 with v0.5 syntax candidates, 30/64 with before-only **syntactic** caller paths, 6/64 unsupported; metadata fetch failures=0 | v0.8 generalization claim falsified; caller graph paths are **not independently verified semantic specifications**. [CI 38032410155](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38032410155) |
| Stage III (32 different PRs, 25 repos, four groups) | v0.9 nested cast-expression replay, plus paired old-method observations | **32/32 PR metadata HTTPError on original CI, source analyzed=0**, so no scientific assessment of v0.9 transfer can be made from this run | Transport failure, **not 0/32 algorithm sensitivity**. Immutable original [CI 38033004178](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38033004178); diagnostic and transport-only retry are separate. |

The first two stages comprise **112 distinct PRs** across the two freezes, on which the exact expression-centered native validators produced **zero prospective automated native witnesses**. These are falsifying negative results, and no developer-only positive replay can turn them into prospective successes. Third-stage 32 PRs bring the *registered prospective sample count* to 144, but are **not 144 successfully assessed samples**, because all third-stage original source retrievals failed. Unit regression totals (up to 203 tests on verified CI 38032847794) measure software correctness of the tools, not empirical APCA coverage.

**Mechanism-development positives (not a held-out accuracy improvement):** SHA-pinned original SpikingJelly #769 yielded four native operator contexts (three dtype gains and one preserving control), while original TorchJD #821 yielded two AST cast sites, six native expression contexts, two previously failing operation recoveries, two dtype-reference changes and two preserving controls. Both had already been inspected in Stage I and cannot be counted as independently discovered defects. [Developer CI 38032847794](https://github.com/lindicaphxag-tech/kaggle/actions/runs/38032847794).

**Audit caveats:** A Git blob digest verifies retrieved bytes against a pinned source in these selected cases; the routine's local SHA is not a signed third-party attestation. A Python AST read/unchanged caller path is evidence of syntax-level relation, not objective intended behavior. Native expression success says only that one hypothetical context produces particular PyTorch operator results, not that an upstream task-level patch is correct. All whole-PR correctness authorizations on these runs equal zero, and no independent human correctness labels or fair execution-budget comparisons to ChangeGuard/Mokav/LLM judges exist.

### 4. Related work with concrete reviewer standards

- ChangeGuard (FSE 2025; DOI 10.1145/3715760) reports 77.1% precision, 69.5% recall on 224 manually annotated changes; its goal is semantic-change detection, not patch intention.
- Mokav (JSS 230, 2025, 112571; DOI 10.1016/j.jss.2025.112571) reports 1,255/1,535 difference-exposing tests; the task and denominators differ.
- SWE-Proof (arXiv 2609.21190; September 2026) shows machine-checked verification is bottlenecked by specification fidelity.
- APCA in the Loop (ESEM 2026; DOI 10.4230/LIPIcs.ESEM.2026.77) evaluated 549 bugs and 1,328 patch candidates; search cost and false rejection are first-class outcomes.
- How Reliable Is LLM-as-Judge for Patch Correctness Assessment? (ESEM 2026; DOI 10.4230/LIPIcs.ESEM.2026.21) studied 326 labeled patches, showing source-identifying metadata, presentation and explanation reliability must be audited.
- Improving Automated Patch Correctness Assessment by Designing LLM-Based Oracles (ICST 2026; DOI 10.1109/ICST69053.2026.00041): stronger prompting alone does not constitute novel source-anchored semantics.

Published performance percentages above must NOT be compared with our different tasks as if evaluated on the same data.

### 5. Required full-paper RQs and decisive falsifiers

**RQ1: Independently grounded obligation extraction.** Among candidate effects discovered automatically from unchanged code and public API semantics, what fraction are human-adjudicated actual requirements? Does CEC beat diff-only patterns at matched costs? A high candidate count alone is inadequate.

**RQ2: Real incorrect / partially correct patches.** With complete source and two independent expert labels, does CEC detect natural omitted/incorrect fixes while retaining correct ones? Reject claims based on synthetic mutants alone.

**RQ3: Selective false authorization at useful coverage.** Primary endpoint: incorrect whole-PR approvals at a prospectively declared *coverage at least 30%*, matched against baseline cost/coverage curves. The 30% is a target, not an achieved number; zero coverage cannot win.

**RQ4: Cost and reviewer benefit.** Measure real CPU/GPU execution, LLM calls, human actions, wrong rejections and missed defects. Compare against strong rerunnable ChangeGuard/Mokav variants on applicable Python cases, project-native tests, randomized probes and sanitized LLM judges.

**RQ5: Independent reproducibility.** Pre-register 100+ untouched real PRs across diverse independent repositories, include correct/wrong/feature/control samples, source SHA/commit/environment dependencies, and obtain at least one external reproduction. No guessed acceptance probability.

### 6. Known blockers (SCIENCE HOLD)

The v0.5 extractor classifies token families across AST changed functions, not independently faithful requirements. Native replay cases are previously inspected and input contexts remain researcher-chosen. The separate CEC checker and native source runner are not yet an integrated fully automatic whole-repository validation pipeline. No comprehensive real-PR ground truth, strong baseline equivalence, reviewer effort study or statistical uncertainty on full-patch decisions exists. Author-side CI pass or more tests cannot remove these blockers. The manuscript is not ready for JSS regular submission; the separate JSS NITP concept-first route remains available after editorial, literature and overlap checks.

### Source-of-truth links

Research PR: https://github.com/lindicaphxag-tech/kaggle/pull/84

Native original-source and synthetic planner: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38028184487

Initial v0.5 API transport failure (preserved): https://github.com/lindicaphxag-tech/kaggle/actions/runs/38029377120

Authenticated same-algorithm retry: https://github.com/lindicaphxag-tech/kaggle/actions/runs/38029566794

**Disclosure:** This working draft is neither submitted nor accepted. Authorship, licence, AI-assistance and funding statements remain for human-author approval.
