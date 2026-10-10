# PCPR vNext: pre-inspection external transfer cohort (2026-10-10)

**Protocol timestamp:** list frozen BEFORE fetching patches/source for the six below in this research iteration. Eligibility is GitHub merged PRs, 2026-07-01 to 2026-10-05, returned by predeclared keyword searches for `nn.Parameter optimizer` / `register_buffer Parameter`. Search matched via *title/body metadata*, so this is a *topic-enriched cohort*, **not** a random sample. Neither model training nor selection is blind to the titles. Prior already inspected DeepChem #5190 is a DEVELOPMENT source and MUST NOT count here. Freeze does NOT imply a new test result or independent gold.

## Frozen targets (do NOT change denominator afterwards)

1. pytorch/torchtitan #4267 — "Preserve requires_grad when distributing parameters"
2. kornia/kornia #5446 — "register plain Quaternion tensors as buffers"
3. huggingface/peft #3481 — "modules_to_save buffer handling"
4. vllm-project/vllm #57163 — "Stop sleep(level=2) from zeroing ... KV scales"
5. jeshraghian/snntorch #453 — "per-channel ... learnable delay time" (negative/feature control)
6. kornia/kornia #5005 — "squeeze ... So2 ... pairwise" (out-of-family control)

## Decisions and measurements frozen now

- Unit of independent evaluation: a **PR**, not each class/parameter/test. Denominator 6 including negative/out-of-language/unsupported. If PR cannot be retrieved, record `FETCH_FAILURE` rather than replacing it.
- Inputs for first prospective pass: whole PR patch from GitHub; base commit and original untouched context file(s) only when independently identifiable from patch and before-source, no after-patch semantic requirement used as its own ground truth.
- Allowed output: `SCOPED_WITNESS`, `COUNTEREXAMPLE`, `ABSTAIN`, and error, with scope and exact line/blob identity. Mere changed method names/AST are candidate evidence, not truth. No claim of whole-PR correctness.
- Main **feasibility** endpoints: source-anchored proposal rate /6, runnable native witness rate /6, false-authority count independently adjudicated, abstention /6, unsupported edits. Accuracy cannot be computed absent independently adjudicated correctness labels.
- Benchmarks of implementation relevance: preexisting repo-local PCPR source-closure (DeepChem-only); a lexical changed-token comparator; standard unit test / random native contexts **only where runnable**. Count actual wall-clock/probe budgets; missing runnable comparator means **no superiority claim**.
- Inspect the six patches only after a versioned implementation is frozen. If implementation revised in response, label all revised results post-hoc. Even if new method attains local witnesses, no genuine cross-project correctness accuracy until gold is adjudicated.

## Strong falsifier

If the implementation cannot generate source-independent runnable obligations in these six, or all evidence depends on patch-written intent, the new broad-transfer claim is REJECTED; do not change selection or labels. Equal-budget compared cost and meaningful selectivity, not zero-authority through abstain.
