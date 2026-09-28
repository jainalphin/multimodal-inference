# Qwen2.5-VL-3B: Overall Summary (W1-2 through W1-5)

## What each piece covered
- **W1-4 (shape trace)**: confirmed the real tensor flow through the model on real hardware.
- **W1-5 (roofline)**: explained the L4 compute and memory limits for the measured stages.
- **W1-3 (single-request trace)**: showed those predictions playing out in one real video request.
- **W1-2 (baseline sweep)**: put a cost envelope on clip length × fps across repeated requests.


## W1-4: shape trace findings
- Confirmed on the W4 rerun: `image_grid_thw=(1, 74, 74)` and `pixel_values=(5,476, 1,176)` -> 5,476 ViT tokens -> 32 ViT blocks (28 windowed + 4 full-attention) -> 2×2 merger -> 1,369 image-token positions -> 1,394 total LLM positions (25 non-image text + 1,369 image placeholders). The embedding hook sees all positions: `(1, 1,394, 2,048)`.
- The W4 rerun directly reports these checkpoint values: vision MLP intermediate size 3,420; vocabulary size 151,936; and LLM MLP intermediate size 11,008. The `lm_head` output independently confirms the vocabulary width. The repository does not retain a cited copy of the paper table, so it does not make an unverified checkpoint-versus-paper comparison.

## W1-5: roofline findings
- Encoder linear (QKV), projector (merger), and LLM prefill linear are all **compute-bound** (AI 771-939 FLOPs/Byte) on L4 — large batched GEMMs.
- **Naive** attention is **memory-bound** on L4 (AI 77.7 FLOPs/Byte). Its score matrix is about 960 MB in bf16 and adds about 1.92 GB of HBM traffic for one write and one read.
- Fused SDPA does not materialize that score matrix. Its roofline AI and bound cannot be inferred from the naive I/O model; they require the selected backend and its measured memory traffic.
- In the final W1-5 rerun on an NVIDIA L4 (compute capability 8.9, PyTorch 2.8.0+cu128) with model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, the measured median throughput was 68.32 TFLOPS (QKV), 5.52 TFLOPS (naive attention), 47.39 TFLOPS (fused SDPA), 49.21 TFLOPS (merger), and 56.31 TFLOPS (LLM prefill linear). Fused SDPA was **8.59x** faster than naive attention (3.239 ms vs. 27.835 ms).
- L4 is Ada Lovelace, not Ampere. The timing capture does not identify PyTorch's selected SDPA backend, so it cannot establish that FlashAttention specifically executed.

## W1-3: single-request trace findings (clip_08s.mp4, 2 fps, 2,392 video tokens)
- The supplied trace's clean phase breakdown is: preprocess 24.7%, H2D 0.3%, encoder 24.4%, prefill 10.2%, and decode 40.3%. These values supersede the earlier, unsupported W1-3 percentages.
- In the profiled run, the sum of CUDA kernel durations divided by phase wall time was 53.2% for encoder, 98.6% for prefill, and 38.6% for decode. This is a profiler-run utilization proxy, not clean-run GPU utilization; no clean GPU-busy figures are recorded.
- The trace's largest aggregated kernel names are an `aten::mm` launcher (559.2 ms, 31.8%, 4,496 calls) and an `aten::addmm` launcher (211.8 ms, 12.0%, 3,348 calls). They are aggregated across the whole trace, so the trace alone does not assign all those calls to decode.
- **59.7% of the 3,081.0 ms clean request occurs before the first output token** (about 1,839.1 ms), and encoder is 41.0% of that time. The phase ordering shows that the complete video is encoded before decode begins; it does not by itself establish the cause of vLLM issue #24728.
- Decode takes 1,241.9 ms in the clean request, or **40.1 ms for each of the 31 tokens after the first**. The first generated token is included at the prefill/decode boundary, so reporting this as the cost of all 32 generated tokens would be inaccurate.

The raw W1-3 trace and `run.json` are not checked in. The supplied table therefore supports these timings, but not claims about hardware, exact package versions, clean-run GPU utilization, or comparison with a different run.

## W1-2: baseline sweep findings
- Tokens scale roughly linearly with clip length x fps (2x fps = 2x tokens; 8s -> 30s @ 1fps = 3.75x tokens).
- In the latest rerun, TTFT was 525.4, 1,081.6, 2,155.1, and 4,532.0 ms as the prompt grew from 1,196 to 8,970 video tokens. Relative to the smallest case, that is 2.1x, 4.1x, and 8.6x; prefill grows 2.0x, 4.0x, and 9.4x.
- Decode remained broadly flat at 38.2--43.0 ms/token across all four rows. This rerun does not support a claim that decode becomes slower as the visual prompt grows.
- Every default-cap case fit on the rerun GPU, including 30s @ 2fps with 8,970 video tokens and 8.9 GiB peak allocated memory (including weights). The captured output omits its GPU and package metadata, so it cannot establish why this differs from the earlier saved run.

## Bottom line
- Compute-heavy stages (prefill, merger, encoder linear layers) are efficient and behave close to what the roofline model predicts.
- The two real inefficiencies are structural, not about raw FLOPs:
  1. **Video preprocessing, transfer, encoding, and prefill serialize before the first output token**, consuming 59.7% of the recorded W1-3 request. Streaming or chunked encoding is a relevant optimization direction, but this trace does not attribute the delay to a particular implementation issue.
  2. **The recorded eager decode path takes 40.1 ms for each token after the first.** The trace supports that it is kernel-heavy and that profiling increases its wall time; it does not support a numerical split between GPU weight streaming and host launch gaps.
