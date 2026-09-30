# Known fixes: upstream survey

Survey of what is already fixed upstream and what is still open.
My own engagement log lives in
[contrib_notes.md](contrib_notes.md).

Snapshot taken week 1, extended 2026-09-30. Re-check assignees and comments before acting on any row.

## Known fixes

### vLLM

| Status | PR | Topic |
|---|---|---|
| Merged | [#54231](https://github.com/vllm-project/vllm/pull/54231) | Remove PyAV video backend |
| Merged | [#44205](https://github.com/vllm-project/vllm/pull/44205) | Qwen3-VL EVS device fix |
| Merged | [#57441](https://github.com/vllm-project/vllm/pull/57441) | Transformers backend video support |
| Merged | [#53708](https://github.com/vllm-project/vllm/pull/53708) | Qwen3-Omni deepstack modality fix |
| Merged | [#46836](https://github.com/vllm-project/vllm/pull/46836) | Oversized-video protection |
| Merged | [#50411](https://github.com/vllm-project/vllm/pull/50411) | Image normalization on GPU, uint8 transfer (covers Qwen2.5-VL). Overlaps W2-3. |
| Merged | [#55370](https://github.com/vllm-project/vllm/pull/55370) | Fix: device normalization skipped under encoder CUDA graphs |
| Merged | [#51289](https://github.com/vllm-project/vllm/pull/51289) | Device normalization extended to Qwen3-VL |
| Merged | [#53675](https://github.com/vllm-project/vllm/pull/53675) | NVDEC video decode on encoder-only instances |
| Merged | [#40830](https://github.com/vllm-project/vllm/pull/40830) | ViT CUDA graphs for Qwen2.5-VL |
| Merged | [#17973](https://github.com/vllm-project/vllm/pull/17973) | Faster Qwen2.5-VL vision RoPE setup (has a profile screenshot) |
| Merged | [#28798](https://github.com/vllm-project/vllm/pull/28798) | Cached cos/sin for Qwen-VL vision RoPE |
| Merged | [#54292](https://github.com/vllm-project/vllm/pull/54292) | Pin CPU tensors before non_blocking H2D |
| Merged | [#47736](https://github.com/vllm-project/vllm/pull/47736) | Qwen2.5-VL honors video fps in temporal M-RoPE |

### SGLang

| Status | PR | Topic |
|---|---|---|
| Merged | [#39539](https://github.com/sgl-project/sglang/pull/39539) | Bounded CPU feature hashing for multimodal requests |
| Merged | [#39679](https://github.com/sgl-project/sglang/pull/39679) | Unified multimodal/generate server path |
| Merged | [#41344](https://github.com/sgl-project/sglang/pull/41344) | FlashAttention 4 in ViT |
| Merged | [#41339](https://github.com/sgl-project/sglang/pull/41339) | Fused Qwen-Image attention and projection kernels |
| Merged | [#40439](https://github.com/sgl-project/sglang/pull/40439) | CPU weight-store handling |
| Merged | [#35318](https://github.com/sgl-project/sglang/pull/35318) | PaddleOCR-VL: single-threaded preprocessing capped throughput with idle GPU |
| Merged | [#35349](https://github.com/sgl-project/sglang/pull/35349) | Preprocessing worker pool sized by where preprocessing runs |

### Transformers

| Status | PR | Topic |
|---|---|---|
| Merged | [#45783](https://github.com/huggingface/transformers/pull/45783) | Encode multimodal data once in generate. May change baseline timings; pin the version. |
| Merged | [#48669](https://github.com/huggingface/transformers/pull/48669) | Qwen2.5-VL temporal RoPE truncated fractional video intervals |

## Contribution targets

### vLLM

| Type | Issue/PR | Topic | Fit |
|---|---|---|---|
| Open PR | [#55582](https://github.com/vllm-project/vllm/pull/55582) | Faster GLM video frame selection | Review or benchmark only; work is already claimed. |
| Open PR | [#55583](https://github.com/vllm-project/vllm/pull/55583) | Skip media decode on UUID cache hits | Review or benchmark only; work is already claimed. |
| Open issue | [#24728](https://github.com/vllm-project/vllm/issues/24728) | Long-video CPU preprocessing and GPU underuse | Strong match for W1-2/W1-3 profiling. Now has an assignee; ask before working on it. |
| Open issue | [#55639](https://github.com/vllm-project/vllm/issues/55639) | Redundant encoder work | Test repeated video requests and encoder reuse. |
| Open issue | [#41343](https://github.com/vllm-project/vllm/issues/41343) | fp8_e5m2 KV cache silently corrupts Qwen2.5-VL output | Reproduce with the frozen eval set and score.py. |
| Open issue | [#59381](https://github.com/vllm-project/vllm/issues/59381) | NaN logits give all-"!" output (Qwen3.5, bf16) | Same symptom as W2-2; cause not confirmed to match. Read only. |
| Open issue | [#47860](https://github.com/vllm-project/vllm/issues/47860) | Metrics for pre-engine multimodal preprocessing time | Fits the phase-timing work. |
| Open issue | [#56172](https://github.com/vllm-project/vllm/issues/56172) | Offload heavy multimodal preprocessing from CPU sidecars | Larger; W6-3 candidate. |
| Open RFC | [#43634](https://github.com/vllm-project/vllm/issues/43634) | Sequence parallelism for ViT | Background for W12-3. Read only. |
| Open issue | [#57740](https://github.com/vllm-project/vllm/issues/57740) | `<\|image_pad\|>` typed in user text is treated as an image placeholder | Reproducible with Qwen2.5-VL-3B; thread already measured that model. |
| Open issue | [#58192](https://github.com/vllm-project/vllm/issues/58192) | Queue limit checked before multimodal render, counted after | Fix PR already opened in thread. Review or benchmark only. |
| Open RFC | [#38175](https://github.com/vllm-project/vllm/issues/38175) | ViT full CUDA graph tracker | Qwen2.5-VL is done (#40830). Contributors claim one model each in the thread. |

### SGLang

| Type | Issue/PR | Topic | Fit |
|---|---|---|---|
| Open issue | [#33388](https://github.com/sgl-project/sglang/issues/33388) | Multimodal tensor views amplify pickle memory | Best match for the current video-memory work. |
| Open issue | [#36853](https://github.com/sgl-project/sglang/issues/36853) | Qwen-Image host OOM during loading | Relevant to Qwen video/image memory investigation. |
| Open issue | [#39831](https://github.com/sgl-project/sglang/issues/39831) | GLM vision processor silently falls back | Focused multimodal compatibility/error-reporting fix. |
| Open issue | [#41568](https://github.com/sgl-project/sglang/issues/41568) | MiMo-V2 processor fails without TorchCodec | Focused video-processor dependency fix. |
| Open issue | [#41490](https://github.com/sgl-project/sglang/issues/41490) | SANA-Video 2.0 support | Larger video-model integration project. |
| Open issue | [#40072](https://github.com/sgl-project/sglang/issues/40072) | Tokenizer workers open CUDA contexts despite `--mm-feature-transport cpu` | Reproducible on one L4 with nvidia-smi. |
| Open issue | [#38676](https://github.com/sgl-project/sglang/issues/38676) | Prometheus metrics for multimodal preprocessing latency | Same idea as vLLM #47860; fits the phase-timing work. |
| Tracking | [#39532](https://github.com/sgl-project/sglang/issues/39532) | Reduce multimodal frontend overhead | Has an assignee. Read to see what is already planned. |

### Transformers

| Type | Issue/PR | Topic | Fit |
|---|---|---|---|
| Open issue | [#47217](https://github.com/huggingface/transformers/issues/47217) | VL models fail when the image placeholder is in user text | Fix PR #47386 is linked. Read with vLLM #57740. |

Closed since week 1: vLLM [#57038](https://github.com/vllm-project/vllm/issues/57038) (completed 2026-09-20).

Open PRs are listed for review only. Before contributing to an open issue,
check its current assignee, comments, and linked branches.
