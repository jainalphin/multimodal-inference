# Timing Helper and Roofline Worksheet — Qwen2.5-VL-3B

## Files
- `roofline_benchmark.py` — timing helper + all four ops, runnable end-to-end on L4.

## Test-case constants (from the model forward-pass trace; processor grid is 74×74 14-pixel patches, or 1036×1036 pixels)
- N_PATCHES = 5476 (ViT patches)
- N_MERGED = 1369 (merger output tokens)
- T_SEQ = 1394 (total LLM prefill sequence length: 25 non-image text positions + 1369 image-placeholder positions)
- INTERMEDIATE_SIZE = 11008 (checkpoint value used by the benchmark)

## Assumptions
- bf16, 2 bytes/element
- 1 multiply-add = 2 FLOPs
- Bytes moved = naive HBM round trip (weights + input + output read/write once each); no credit for cache/tiling reuse a fused kernel might get
- Attention score matrix (**naive only**): written once, read once (softmax's internal read/write folded into this, not counted separately — including it would lower AI further but not change the verdict)
- Fused SDPA does not materialize that score matrix. Its exact HBM traffic depends on the backend, tile sizes, and cache behavior, so this worksheet does not assign it the naive attention AI or a roofline verdict.
- GPU peaks are dense (non-sparse) Tensor Core FP16 numbers from official NVIDIA datasheets
- These are theoretical peaks; achieved performance is an upper-bound comparison, not a prediction of actual runtime

## GPU specs used

| GPU | FP16 Tensor peak | Bandwidth | Ridge point |
|---|---|---|---|
| L4 | 121 TFLOPS | 300 GB/s | 403 FLOPs/Byte |

## L4 rerun (bf16; NVIDIA L4, compute capability 8.9, PyTorch 2.8.0+cu128)

This terminal capture establishes the L4 timings below for model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. It does not identify PyTorch's selected SDPA backend, so it cannot by itself prove that a specific FlashAttention backend executed.

| Operation | FLOPs | Bytes | AI | median ms | p95 ms | Achieved TFLOPS | L4 verdict |
|---|---|---|---|---|---|---|---|
| Encoder linear (QKV) | 53.8G | 65.9M | 817 | 0.788 | 0.805 | 68.32 | compute-bound |
| Attention (naive) | 153.5G | 1975M | 77.7 | 27.835 | 28.196 | 5.52 | memory-bound |
| Attention (fused SDPA) | 153.5G | backend-dependent | n/a | 3.239 | 3.268 | 47.39 | profile I/O |
| Projector (merger) | 100.5G | 107M | 939 | 2.042 | 2.117 | 49.21 | compute-bound |
| LLM prefill linear | 62.85G | 81.5M | 771 | 1.116 | 1.140 | 56.31 | compute-bound |

## Notes
- Fused SDPA is **8.59x** faster than the naive median (27.835 / 3.239 ms). The timing shows a much faster fused implementation; it does not identify PyTorch's selected SDPA backend.
- L4 is Ada Lovelace (compute capability 8.9) and supports bf16 Tensor Cores. A kernel trace is required before attributing the fused result to FlashAttention specifically.
- Checkpoint dimensions used by the W1 experiments: ViT MLP intermediate size 3420, vocabulary size 151936, and LLM intermediate size 11008. The repository does not retain a cited paper table for a reproducible comparison.
- The benchmark now prints the CUDA device, compute capability, and PyTorch version. Capture those lines with future timings; use an Nsight or PyTorch-profiler kernel trace to support a claim about the fused SDPA backend or its roofline bound.
