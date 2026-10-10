# PCPR v0.8 — 64 NEW merged PR holdout (pre-inspection lock, 2026-10-10)

**Chronology:** This dataset is frozen AFTER the complete v0.7 48-PR result (0/48 native witnesses) and AFTER development of the generalized v0.8 source-exact tensor + self.weight interpreter, but BEFORE opening these 64 PR patches/source in this research iteration. Fixed semantic implementation: `symbolic_native_operator_v08.py` commit `2fd8b21dcc2944d1703315b98767163e216c306b`; graph sidecar `unchanged_consumer_graph_v08.py` commit `dfd111e9c4a5b5bb3d84ef2486a482b56d2a0e05`.

**Cohort selection:** 8 groups x 8 merged PRs returned by public GitHub merged-title/metadata searches in the 2026-07-01..2026-10-09 interval. A deliberately *topic-enriched, non-random* study with an 8-PR feature/refactor negative-control group; not a representative patch population. No patch correctness labels inferred from merge. Some repositories recur; confidence intervals must cluster at repository rather than PR.

**Frozen protocol:** 64 PR units including files/languages not supported; NO replacements or dropping failures. Use base/head pinned source Git blob SHA, at most 8 changed non-test Python files per PR, at most 8 exact torch-factory dtype additions per file; 4 CPU dtype contexts per bounded operator (float16/bfloat16/float32/float64), with no import of untrusted project code. Compare source candidates / native observations vs v0.5 AST features and v0.7 exact local-name interpreter at equal file, context and HTTP budgets. Native operator effect (not requirement truth); unchanged BEFORE consumer graph is separately hypothesis provenance, not validated oracle. Count errors, abstention, negative-control false proposals, costs, and all 64 denominator. Zero whole-patch correctness decisions. Report both per-PR and per-source-function/operation metrics without mixing their denominators.

**Falsification:** 0/64 native means v0.8 generalized interpreter has not transferred; post-inspection modifications are DEVELOPMENT ONLY and need ANOTHER freeze for genuine uninspected claims. No 64/64 candidate matches as efficacy/accuracy. Matched source-level effects are not correct-patch verification. Comparisons with ChangeGuard/FixCheck/LLM judges require independent gold and budgets, absent here.

## precision (8)

1. https://github.com/kornia/kornia/pull/5358
2. https://github.com/braindecode/braindecode/pull/1259
3. https://github.com/vllm-project/vllm-ascend/pull/18062
4. https://github.com/jobovy/galpy/pull/1613
5. https://github.com/torch-spyre/torch-spyre/pull/5152
6. https://github.com/flagos-ai/FlagGems/pull/6614
7. https://github.com/pytorch/helion/pull/3796
8. https://github.com/SimplexLab/TorchJD/pull/792

## state (8)

1. https://github.com/NVIDIA/structured-data-models/pull/642
2. https://github.com/pese-git/llm-arch-research/pull/32
3. https://github.com/kornia/kornia/pull/5559
4. https://github.com/braindecode/braindecode/pull/1235
5. https://github.com/metatensor/metatensor/pull/1165
6. https://github.com/pytorch/torchtitan/pull/5107
7. https://github.com/vllm-project/vllm-omni/pull/7061
8. https://github.com/neuropoly/SmaugLab/pull/95

## numerical (8)

1. https://github.com/kornia/kornia/pull/5329
2. https://github.com/kornia/kornia/pull/5579
3. https://github.com/kornia/kornia/pull/5484
4. https://github.com/Grid-Raven/jammy_flows/pull/12
5. https://github.com/emdgroup/baybe/pull/925
6. https://github.com/knatarajan-lab/cehrgpt/pull/156
7. https://github.com/chrisfinlay/mri-generative-priors/pull/12
8. https://github.com/DeGrootResearchGroup/aquaflux/pull/634

## layout (8)

1. https://github.com/RBLN-SW/vllm-rbln/pull/1242
2. https://github.com/kornia/kornia/pull/5058
3. https://github.com/kornia/kornia/pull/5558
4. https://github.com/pytorch/helion/pull/3789
5. https://github.com/NVIDIA/structured-data-models/pull/942
6. https://github.com/MyoHub/myosuite/pull/507
7. https://github.com/tenstorrent/tt-metal/pull/56069
8. https://github.com/isaac-sim/IsaacLab/pull/8052

## device (8)

1. https://github.com/kornia/kornia/pull/5142
2. https://github.com/MooreThreads/vllm-musa/pull/246
3. https://github.com/unilabsim/unisim/pull/351
4. https://github.com/m87-labs/kestrel/pull/261
5. https://github.com/lvyufeng/relic-core/pull/33
6. https://github.com/kornia/kornia/pull/4164
7. https://github.com/flatppl/flatppl-python/pull/18
8. https://github.com/vllm-project/vllm/pull/54213

## checkpoint (8)

1. https://github.com/AI-Hypercomputer/maxtext/pull/5508
2. https://github.com/google/tunix/pull/2585
3. https://github.com/kozistr/pytorch_optimizer/pull/579
4. https://github.com/unslothai/unsloth/pull/13092
5. https://github.com/pytorch/torchtitan/pull/5114
6. https://github.com/OpenPipe/ART/pull/1001
7. https://github.com/rwth-i6/returnn/pull/1921
8. https://github.com/gke-labs/open-rl/pull/322

## negative_and_features (8)

1. https://github.com/kornia/kornia/pull/5606
2. https://github.com/jeshraghian/snntorch/pull/455
3. https://github.com/flagos-ai/FlagGems/pull/6919
4. https://github.com/tracel-ai/burn/pull/5775
5. https://github.com/NVIDIA/structured-data-models/pull/941
6. https://github.com/pytorch/torchtitan/pull/4904
7. https://github.com/google/tunix/pull/2635
8. https://github.com/huggingface/diffusers/pull/14946

## gradient (8)

1. https://github.com/kornia/kornia/pull/5517
2. https://github.com/kornia/kornia/pull/5563
3. https://github.com/onnxsim/onnxsim/pull/2117
4. https://github.com/braindecode/braindecode/pull/1187
5. https://github.com/NVIDIA/structured-data-models/pull/910
6. https://github.com/kornia/kornia/pull/5409
7. https://github.com/Grid-Raven/jammy_flows/pull/11
8. https://github.com/kornia/kornia/pull/5076

**Ledger dimensions:** n_pr=64; n_repos=40; duplicate_PR=false. All original links are immutable membership; new CI results belong to a single frozen `run_v08_new_64_holdout.py` first-score record.
