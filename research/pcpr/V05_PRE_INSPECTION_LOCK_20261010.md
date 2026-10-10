# PCPR v0.5 — frozen source-effect transfer cohort (2026-10-10)

## Lock chronology

- Algorithm `source_effect_ir_v05.py` was committed as `f68acdb92d59b8cbeef9f317eef779a9569563d1`; pre-freeze tests committed as `cbad45b8738aa1728b503577d84f3169240b6b1a`.
- Eight external merged PRs were selected using TITLE+MERGE METADATA only; their patches have NOT been fetched/inspected when this file is committed.
- Selection enriched for dtype, serialization and tensor changes and includes a Transformers negative-control edit, so **not a random, representative software population**.
- Later analysis, bug fixes, semantic rules or tuning based on observed patches are **POST-HOC** and cannot change this frozen first-score denominator.

## Freeze — eight PRs from eight distinct repositories

1. `vllm-project/vllm#59106` — preserve batch-invariant mean output dtype
2. `NVIDIA/structured-data-models#1052` — preserve residual output dtype during compilation
3. `jeshraghian/snntorch#451` — preserve population-code input dtype
4. `huggingface/transformers#49315` — repeated-ngram counting fix (negative-control family)
5. `Project-MONAI/MONAI#9103` — BarlowTwins loss dtype
6. `pytorch/ao#4921` — persist static fake-quantizer state
7. `pytorch/tensordict#2011` — reshape/split/where tensorclass methods
8. `kornia/kornia#5338` — serialize ZCAWhitening fitted state

## First-score endpoints

- Count 8 unique PRs including failures, errors and unsupported projects. No replacement if a case fails.
- Count `n_typed_changed_functions` and `n_typed_facets` as **hypothesis extraction only**, not detecting genuine bug, patch correctness, or independent specification origin.
- `n_native_executions=0`, `n_adjudicated_requirements=0`, `n_patch_approvals=0` by design. Any contradictory claim invalidates publication.
- Baselines: v0.1 old narrow field detection; naive lexical changed-line/category extraction; always-abstain. Our v0.5 must demonstrate **precision of extracted semantic facet** by independent adjudication before advertising superiority; high candidate recall alone is not evidence.
- Source patch text may trigger *which property to investigate*, but cannot determine that the patched implementation is correct. BEFORE function arguments and unchanged-source context are scope candidates, not proof of intent.

## Failure gate

If all 8 yield no typed candidates, method is still non-general; if all yield candidates but no native witnesses, the software engineering contribution remains incomplete. Under no circumstances convert positive keyword matches into a PR correctness improvement.
