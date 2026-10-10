# PCPR v0.9 — THIRD prospective source corpus lock, 32 merged PRs (2026-10-10)

**Chronology:** first 48 v0.7 PRs (0 native), next disjoint 64 v0.8 PRs (0 native) were frozen and first-scored BEFORE this third cohort. The generic v0.9 cross-expression cast operator was developed from a previously inspected TorchJD PR #821. The new cohort below is frozen without reading any of these 32 patch contents in this research iteration. v0.9 algorithm hash/commit: `471ea5d344b234defa6491c283395233750bcffd`; the implementation is read-only. Any change to predicates based on this cohort is POST-HOC and requires new holdout.

**Sampling bias:** merged PR title/metadata topic-enriched from July 1–October 9 2026, 8 per group including known-negative or other-language controls. Not random, multiple Kornia PRs, no status-to-truth shortcut. Count unique PRs, repository clusters, CPU-only applicability and source-failure subgroups. No cherry-picked replacements.

**Frozen first-score:** n=32; max 8 non-test modified .py per PR; 8 dtype cast operators/file; three CPU dtype contexts `float16,bfloat16,float32`; source-exact AST independent before/head Git SHA; comparison with v0.5, v0.7, v0.8 at identical HTTP/file/context budgets; new outcomes are local dtype restoration/operation recovery only, NOT an external specification or whole-PR correctness label. Report per-group and unconditional 32-denominator witness rates and errors. Unknown runtime or unsupported language => abstain. Independent adjudicated specs =0. Strong LLM/APCA baselines not run, so no SOTA claim.

## cast_and_gradient (8)

1. https://github.com/EleutherAI/bergson/pull/488
2. https://github.com/NVIDIA-NeMo/Automodel/pull/4150
3. https://github.com/tracel-ai/burn/pull/5843
4. https://github.com/meta-pytorch/flex_shard/pull/23
5. https://github.com/DIAGNijmegen/lightstream/pull/346
6. https://github.com/kozistr/pytorch_optimizer/pull/560
7. https://github.com/keras-team/keras/pull/23593
8. https://github.com/intel/auto-round/pull/2232

## dtype_operator (8)

1. https://github.com/kornia/kornia/pull/5571
2. https://github.com/kornia/kornia/pull/5303
3. https://github.com/kornia/kornia/pull/5524
4. https://github.com/kornia/kornia/pull/4898
5. https://github.com/NVIDIA/recsys-examples/pull/495
6. https://github.com/kornia/kornia/pull/5377
7. https://github.com/kozistr/pytorch_optimizer/pull/599
8. https://github.com/kornia/kornia/pull/5353

## mixed_infrastructure (8)

1. https://github.com/apple/coremltools/pull/2833
2. https://github.com/Boffee/piper-kernels/pull/87
3. https://github.com/flagos-ai/FlagGems/pull/6039
4. https://github.com/facebookresearch/spdl/pull/1726
5. https://github.com/rwth-i6/returnn/pull/1891
6. https://github.com/huggingface/pytorch-image-models/pull/2841
7. https://github.com/huggingface/sentence-transformers/pull/4113
8. https://github.com/ModelCloud/GPTQModel/pull/3144

## negative_and_controls (8)

1. https://github.com/scipy/scipy-stubs/pull/2530
2. https://github.com/pandas-dev/pandas/pull/70909
3. https://github.com/stdlib-js/stdlib/pull/15747
4. https://github.com/ml-explore/mlx/pull/4626
5. https://github.com/onnx/onnx/pull/8506
6. https://github.com/kornia/kornia/pull/5526
7. https://github.com/NVIDIA/physicsnemo/pull/2072
8. https://github.com/scikit-hep/uhi/pull/280

Unique repositories: 25. Unique PR units: 32; duplicates: false. Previous development PRs excluded from the numerator.
