# Roofline: which ops are compute- or memory-bound

Supporting detail for the finding in the [main README](../../README.md).

Times four ops from the Qwen2.5-VL forward pass and classifies each as compute-
or memory-bound on an L4.

```bash
python experiments/w01-5-roofline/roofline_benchmark.py
```

The helper warms up 5 times, synchronizes, then takes 20 CUDA-event-timed
repeats and reports median and p95.

## Results

bf16, NVIDIA L4 (compute capability 8.9), PyTorch 2.8.0+cu128, revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`.

**L4: 121 TFLOPS dense bf16 peak, 300 GB/s → ridge point 403 FLOPs/byte.**

| Operation | FLOPs | Bytes | AI | median ms | p95 ms | TFLOPS | % peak | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| Encoder linear (QKV) | 53.8G | 65.9M | 817 | 0.772 | 0.791 | 69.8 | 58% | compute-bound |
| Attention (naive) | 153.5G | 1975M | 77.7 | 27.693 | 28.276 | 5.5 | 5% | memory-bound |
| Attention (fused SDPA) | 153.5G | backend-dependent | n/a | 2.935 | 3.008 | 52.3 | 43% | see note |
| Projector (merger) | 100.5G | 107M | 939 | 2.049 | 2.231 | 49.0 | 41% | compute-bound |
| LLM prefill linear | 62.85G | 81.5M | 771 | 1.111 | 1.132 | 56.6 | 47% | compute-bound |

The three compute-bound ops reach 41–58% of peak — a normal fraction for single
unfused linears. Naive attention reaches 5%, exactly as its AI of 77.7 (far
below the 403 ridge point) predicts.

**Fused SDPA is 9.4× faster than naive** (27.693 → 2.935 ms). It does not
materialize the score matrix, so its HBM traffic depends on backend, tile
sizes, and caching — this worksheet assigns it no AI or verdict. The run also
does not record *which* SDPA backend PyTorch chose, so the speedup cannot be
attributed to FlashAttention without a kernel trace.

## Open: naive attention misses its own roofline by 4.2×

Naive attention is memory-bound, so the roofline predicts
AI × bandwidth = 77.7 × 300 GB/s = **23.3 TFLOPS**. It achieves **5.5** — a
4.2× shortfall. Equivalently, it moves 1,975 MB in 27.693 ms = **71 GB/s, only
24% of L4's 300 GB/s peak**.

So the memory-bound verdict is right but the model is optimistic. Candidate
explanations, none yet tested:

- The byte count assumes one write and one read of the score matrix. Softmax
  likely touches it more than twice, and a 16-head 5476² bf16 tensor is ~960 MB
  — far too large for cache, so every pass is a full HBM round trip.
- Four separate kernels (matmul, scale, softmax, matmul) each re-read their
  input, rather than one fused pass.
- Softmax is not a pure streaming operation: the max and sum reductions mean
  extra passes over the data.

**Next step:** count actual HBM traffic for the naive path with Nsight or
`torch.profiler` memory stats, rather than inferring it. That would also settle
what fused SDPA's real traffic is, which this worksheet currently declines to
estimate.

## Shapes used

From the W1-4 image hook run: 5,476 ViT patches, 1,369 merger output tokens,
1,394 LLM prefill positions. Checkpoint dims: vision width 1,280, ViT MLP
3,420, LLM width 2,048, LLM MLP 11,008, vocab 151,936.

Note these are the *image* shapes. The measured video request is larger — 9,568
patches, 2,392 merged tokens, 2,420 positions — so attention costs more there,
quadratically. The AI values and verdicts do not change.

## Assumptions

- bf16 (2 bytes/element); 1 multiply-add = 2 FLOPs
- Bytes = naive HBM round trip — weights, input, and output each read/written
  once, with no credit for cache or tiling reuse
- Naive attention's score matrix counted as written once, read once; softmax's
  internal traffic folded in. Counting it separately would lower AI further
  without changing the verdict
- GPU peak is the dense (non-sparse) Tensor Core figure from NVIDIA's datasheet

These are theoretical peaks, so achieved-versus-peak is an upper-bound
comparison, not a runtime prediction.
