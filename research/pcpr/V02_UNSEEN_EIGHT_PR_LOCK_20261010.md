# PCPR v0.2: independent PR transfer gate — PRE-DIFF FREEZE

Locked on 2026-10-10, after implementation of `typed_effects_v02.py` was committed in `92aeab877d341e8e923f451e2e45757c4b6e7516`, and after v0.1 six-PR failures were inspected. **The eight PR patches below have NOT been opened/inspected in this research iteration as of this freeze.** Names/titles may have been viewed in GitHub search results. This is title-selected, non-random, enriched search; not a population metric.

## Fixed cohort of eight DIFFERENT repositories
1. pytorch/torchtitan #4474 — optimizer state checkpoint
2. NVIDIA/structured-data-models #637 — BufferList state migration
3. pytorch/executorch #22994 — BatchNorm into preceding convolution
4. torch-spyre/torch-spyre #4383 — where fp32/int inputs
5. QwenAudio/SenseVoice #358 — CTC length alignment
6. NVIDIA-NeMo/Automodel #3850 — InfoNCE padded negatives
7. kornia/kornia #5026 — fallback point dtype
8. applied-material-modeling/neml2 #469 — KDE kernel domain bound

## Fixed instrument and metrics
- Use EXACT `typed_effects_v02.py` code as committed at `92aeab877d341e8e923f451e2e45757c4b6e7516`: only four operator patterns, isolated PyTorch microcontexts. No rule additions, threshold tweaks, name special cases, or reranking on this cohort.
- PR-level unique units, denominator eight including unsupported/fetch failures. Count pattern proposals, actual isolated native semantic deltas (only when ran), contextual abstentions, skipped mechanisms, and source identity. Each PR should be reported exactly once.
- Any `source-independent specification correct`, `wrong-patch detection rate`, `full patch authorization` or `statistical significance` claim is **FORBIDDEN** without independently adjudicated ground truth and project-native replay.
- Baselines: v0.1 before-forward parameter candidate detector at original commit `c5167702ebe7e5c8a64afb321c1943a01f967029` (if re-run), lexical diff-only trigger with no runtime evidence, and always abstain. Matching cost and same corpus mandatory before quantitative superiority.
- Follow-up algorithm modifications may use these eight as development, but results after edits are explicitly POST-HOC, not this frozen first score. Do NOT swap failures.

Author-tracked validity: neither a maintainer/adjudicator study nor full-package integration, so this can only test *mechanism transfer coverage*.
