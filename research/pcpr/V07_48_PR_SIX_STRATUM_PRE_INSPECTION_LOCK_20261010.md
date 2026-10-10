# PCPR v0.7 — 48-real-PR multi-repository holdout lock (2026-10-10)

**Chronology:** locked BEFORE inspection of any of these 48 patch contents in this iteration. The general v0.7 analyzer source was committed at `deb2f16a660c352cb46d0f1113643f2623a7c784` and tests at `2cf70937a23f9fa6d816405bc4b9a9412254410e`. The six cohort categories are from merged-PR title/metadata search result lists; **not random, not population representative, and not independently truth-adjudicated**.

**Units:** Each PR counted once, including language-unsupported / file-skip / source fetch failure, no replacements, no hindsight removal. The source runner uses a GitHub Actions token for metadata (read only), raw pinned base/head blobs for source, and a strict original-AST factory-call interpreter for PyTorch CPU. Whole source files are NEVER executed by this evaluator. Device-only/CUDA-only tasks likely abstain. Each source excerpt records source digests (not independent external specification). The v0.7 evaluator checks only an isolated addition of `dtype=<existing_tensor>.dtype` to otherwise-identical factory calls: this knowingly has small coverage outside dtype fixes.

**Report:** n_total=48; n_repo clusters; n_python_modified; fetch failures; n_facet/AST candidates; unique PRs with 1+ **successful native operator observations** (separately positive dtype gains / preserving only); abstention; native CPU context count; execution time. Compare same PR sample and budget with the existing v0.5 semantic-hypothesis detector and naïve lexical matched factories. Zero-native always abstain baseline has 0 utility. **No precision/recall or wrong authorization rate until independent gold.** Conditional metrics only; publish unconditional 48-denominator always.

**No leakage:** Code changes after examining any cohort patch go to a POST-HOC variant and must not replace v0.7 first-score. A failed fetch is FETCH_FAILURE not an algorithm miss. The initial attempt and any transport replay must remain distinct. Filter only by selection protocol, not desired effects.

## Stratum dtype (8)

1. [geomstats/geomstats #2133](https://github.com/geomstats/geomstats/pull/2133)
2. [SimplexLab/TorchJD #821](https://github.com/SimplexLab/TorchJD/pull/821)
3. [NVIDIA/Model-Optimizer #2651](https://github.com/NVIDIA/Model-Optimizer/pull/2651)
4. [huggingface/lerobot #4786](https://github.com/huggingface/lerobot/pull/4786)
5. [roboflow/rf-detr #1616](https://github.com/roboflow/rf-detr/pull/1616)
6. [fangwei123456/spikingjelly #769](https://github.com/fangwei123456/spikingjelly/pull/769)
7. [google-ai-edge/litert-torch #1318](https://github.com/google-ai-edge/litert-torch/pull/1318)
8. [pytorch/tensordict #1983](https://github.com/pytorch/tensordict/pull/1983)

## Stratum state (8)

1. [huggingface/peft #3816](https://github.com/huggingface/peft/pull/3816)
2. [huggingface/datasets #8387](https://github.com/huggingface/datasets/pull/8387)
3. [pytorch/executorch #23266](https://github.com/pytorch/executorch/pull/23266)
4. [OpenRAL/openral #305](https://github.com/OpenRAL/openral/pull/305)
5. [decoderesearch/SAELens #749](https://github.com/decoderesearch/SAELens/pull/749)
6. [jjhwilkinson/iwpc #43](https://github.com/jjhwilkinson/iwpc/pull/43)
7. [verl-project/verl #7610](https://github.com/verl-project/verl/pull/7610)
8. [pytorch/ao #4886](https://github.com/pytorch/ao/pull/4886)

## Stratum device (8)

1. [tile-ai/tilelang #3457](https://github.com/tile-ai/tilelang/pull/3457)
2. [MrNeRF/LichtFeld-Studio #2959](https://github.com/MrNeRF/LichtFeld-Studio/pull/2959)
3. [firnlabs/minirl #34](https://github.com/firnlabs/minirl/pull/34)
4. [tracel-ai/burn #5942](https://github.com/tracel-ai/burn/pull/5942)
5. [pytorch/executorch #23547](https://github.com/pytorch/executorch/pull/23547)
6. [LMCache/LMCache #5541](https://github.com/LMCache/LMCache/pull/5541)
7. [facebookresearch/spdl #1727](https://github.com/facebookresearch/spdl/pull/1727)
8. [RBLN-SW/vllm-rbln #1244](https://github.com/RBLN-SW/vllm-rbln/pull/1244)

## Stratum gradient (8)

1. [ICB-DCM/pyPESTO #1776](https://github.com/ICB-DCM/pyPESTO/pull/1776)
2. [onnxsim/onnxsim #2120](https://github.com/onnxsim/onnxsim/pull/2120)
3. [kornia/kornia #5642](https://github.com/kornia/kornia/pull/5642)
4. [keras-team/keras-mlx #65](https://github.com/keras-team/keras-mlx/pull/65)
5. [PyAutoLabs/PyAutoLens #768](https://github.com/PyAutoLabs/PyAutoLens/pull/768)
6. [araffin/sbx #95](https://github.com/araffin/sbx/pull/95)
7. [tensorflow/tensorflow #127742](https://github.com/tensorflow/tensorflow/pull/127742)
8. [rwth-i6/returnn #1919](https://github.com/rwth-i6/returnn/pull/1919)

## Stratum shape (8)

1. [vllm-project/vllm #60529](https://github.com/vllm-project/vllm/pull/60529)
2. [huggingface/transformers #49426](https://github.com/huggingface/transformers/pull/49426)
3. [tenstorrent/tt-metal #53891](https://github.com/tenstorrent/tt-metal/pull/53891)
4. [Genesis-Embodied-AI/genesis-world #3396](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3396)
5. [isaac-sim/IsaacLab #7813](https://github.com/isaac-sim/IsaacLab/pull/7813)
6. [pytorch/rl #4318](https://github.com/pytorch/rl/pull/4318)
7. [open-edge-platform/anomalib #3834](https://github.com/open-edge-platform/anomalib/pull/3834)
8. [PriorLabs/TabPFN #1223](https://github.com/PriorLabs/TabPFN/pull/1223)

## Stratum checkpoint (8)

1. [AI-Hypercomputer/maxtext #5463](https://github.com/AI-Hypercomputer/maxtext/pull/5463)
2. [OpenPipe/ART #992](https://github.com/OpenPipe/ART/pull/992)
3. [google/tunix #2630](https://github.com/google/tunix/pull/2630)
4. [Tencent-Hunyuan/UniRL #579](https://github.com/Tencent-Hunyuan/UniRL/pull/579)
5. [kozistr/pytorch_optimizer #576](https://github.com/kozistr/pytorch_optimizer/pull/576)
6. [unslothai/unsloth #13051](https://github.com/unslothai/unsloth/pull/13051)
7. [marin-community/marin #9460](https://github.com/marin-community/marin/pull/9460)
8. [NVIDIA-NeMo/Megatron-Bridge #6260](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6260)

Repository count: 47; PR count: 48. Repeated repository: pytorch/executorch (2 PRs). Repositories are the clustering unit. Development/previously inspected PRs were excluded by manual choice, but sample selection is topic enriched rather than random.
