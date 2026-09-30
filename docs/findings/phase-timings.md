# Phase timings and top kernels

Supporting detail for the finding in the [main README](../../README.md).

`clip_08s.mp4` at 2 FPS, bf16 + SDPA, on an NVIDIA L4.

## Request

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
- 16 frames at **364×644 px** (26×46 patch grid) → 8 frame groups → 2,392 video
  tokens; 2,420 input tokens
- 32 generated tokens, greedy, seed 1,234, `cap_pixels_per_frame=True`
- 2 untraced warmup requests; the last is the reported clean run
- Weights 7,275 MiB resident; 55,945 GPU kernels, 1,750.9 ms total kernel time
- Run: `runs/20260928T154432Z_trace` (not committed; `runs/` is gitignored)

SDPA is requested, but the trace does not record which SDPA *backend* PyTorch
selected. A kernel-level trace is needed before attributing the result to
FlashAttention.

## Phase timing

| Phase | Clean ms | Clean share | Traced ms | GPU busy share |
|---|---:|---:|---:|---:|
| preprocess | 762.3 | 25.4% | 601.0 | 0.0% |
| h2d | 10.4 | 0.3% | 10.2 | 0.0% |
| encoder | 741.6 | 24.7% | 1,074.5 | 54.2% |
| prefill | 310.6 | 10.4% | 309.5 | 98.6% |
| decode | 1,171.6 | 39.1% | 2,128.4 | 40.5% |

The clean request took **2,996.5 ms**. 1,824.9 ms (**60.9%**) elapsed before
the first token, and the encoder was 40.6% of that. Decode cost 37.8 ms per
token after the first.

Use the clean column for shares: the profiler inflates decode by ~82% and the
encoder by ~45%.

## Top five kernels by GPU time

| # | Kernel | GPU ms | Share | Calls | Operator |
|---:|---|---:|---:|---:|---|
| 1 | `internal::gemvx::kernel<...bfloat16...>` | 559.0 | 31.9% | 4,496 | `aten::mm` |
| 2 | `internal::gemvx::kernel<...bfloat16...>` | 211.7 | 12.1% | 3,348 | `aten::addmm` |
| 3 | `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128` | 139.5 | 8.0% | 109 | `aten::mm` |
| 4 | `cutlass_80_tensorop_bf16_s16816gemm_bf16_128x256` | 88.2 | 5.0% | 64 | `aten::addmm` |
| 5 | `at::native::elementwise_kernel<...>` | 80.1 | 4.6% | 2,530 | `aten::mul` |

The top two are **GEMV** — matrix-*vector* — 44.0% of all kernel time across
7,844 calls. GEMV is what batch-size-1 decode produces, and it cannot saturate
tensor cores. The CUTLASS GEMMs below have only 173 calls combined; those are
the large batched matmuls of encoder and prefill.

## Phases

- `preprocess` — video sampling, resize, chat template, request prep (CPU)
- `h2d` — tensor transfer to CUDA
- `encoder` — Qwen vision tower
- `prefill` — first LM pass over the full multimodal prompt
- `decode` — remaining token-by-token generation

Boundaries come from forward hooks on the vision tower plus a logits processor
firing on the first generated token. Each boundary synchronizes CUDA, so a
phase covers its own GPU work.

## Reproducing

```bash
python experiments/w01-3-first-trace/first-trace.py
```

Writes `run.json`, `results_trace.json`, `video.trace.json` (open at
[ui.perfetto.dev](https://ui.perfetto.dev)), and `trace_summary.md` to a
timestamped `runs/` directory.
