# W1-3 comparison: Hugging Face and vLLM

The measurements below are a historical supplied comparison configured with
the same Qwen2.5-VL-3B-Instruct model, 8-second video, 2 FPS sampling rate,
greedy decoding, and a limit of 32 output tokens. The raw traces, run metadata,
and package versions are not checked in, so the report cannot independently
establish the hardware or be treated as comparable to a later W1-3 trace.

| Phase | Hugging Face (ms) | vLLM (ms) | vLLM change |
|---|---:|---:|---:|
| Preprocess | 237.6 | 554.5 | +316.9 |
| H2D / scheduler | 10.3 | 12.2 | +1.9 |
| Vision encoder | 1,315.7 | 966.2 | -349.5 |
| Prefill | 2,106.8 | 1,911.0 | -195.8 |
| Decode | 1,794.7 | 1,676.4 | -118.3 |
| **Total** | **5,465.1** | **5,120.3** | **-344.8** |

## First-token latency

| Metric | Hugging Face | vLLM |
|---|---:|---:|
| Time before first token | 3,670.3 ms | 3,443.8 ms |
| Share of total request | 67.2% | 67.3% |
| Decode time per token after the first | 57.9 ms | 54.1 ms |

vLLM reduced total latency by approximately 6.3% and time to first token by
approximately 6.2%. Both implementations still process the complete video
before decoding. The largest improvement was in the vision encoder; vLLM
preprocessing was slower in this measurement.

This is consistent with both implementations completing video preparation and
encoding before decode. It does not, by itself, identify the cause or reproduce
vLLM issue #24728.

The vLLM trace also contained `kernel_unified_attention`, accounting for
1,710.2 ms, or 48.4% of profiled kernel time. The clean timings above are used
for comparison because profiling adds overhead.
