# Where the time goes in Qwen2.5-VL video inference

Profiling a 3B vision-language model on video, one request at a time, to find
what actually limits latency. NVIDIA L4, bf16, `Qwen2.5-VL-3B-Instruct`.

## The finding

**The GPU is not the bottleneck.** An 8-second clip at 2 FPS (2,392 video
tokens, 32 generated) takes 2,996 ms. Only ~1,750 ms is GPU kernel time, and
most of those kernels are too small to use the hardware.

| Phase | ms | Share | GPU busy | |
|---|---:|---:|---:|---|
| preprocess | 762.3 | 25.4% | **0.0%** | pure CPU — frame decode and resize |
| h2d | 10.4 | 0.3% | 0.0% | negligible |
| encoder | 741.6 | 24.7% | 54.2% | mixed |
| prefill | 310.6 | 10.4% | **98.6%** | saturated |
| decode | 1,171.6 | 39.1% | **40.5%** | launch-bound |

Two host-side limits, at opposite ends of the request:

**Before the first token (60.9% of latency),** preprocessing spends 762 ms
doing *zero* GPU work — video decode and resize on the CPU, blocking the start.

**After it,** decode is the largest phase yet sits 60% idle. 44% of all kernel
time is GEMV — matrix-*vector* — across 7,844 launches, which is what
batch-size-1 decoding produces. Too small to fill the SMs, so the GPU waits
between launches.

Prefill is the control: at 98.6% busy, the same hardware saturates when handed
a large batched matmul. The [roofline](experiments/w01-5-roofline/roofline.md)
agrees — every op that matters is compute-bound (arithmetic intensity 771–939
against a ridge point of 403) and reaches 41–58% of peak. The hardware is fine.
The scheduling is not.

### What that means for optimization

| Target | Approach | Bounds |
|---|---|---:|
| Decode launch overhead | CUDA graphs or a fused decode path | ~39% |
| CPU preprocessing | GPU-side resize, or overlap with setup | ~25% |

Independent and additive, covering ~64% of the request. **Prefill is not worth
touching** — you cannot schedule your way out of 98.6% utilization.

### What this does not show

Batch size 1 only, so the GEMV signature is partly an artifact — at production
batch those become GEMMs. Measured on plain Transformers, not vLLM or SGLang,
which may already fix the preprocessing path. One clip, one run: preprocess
(762.3 ms) and encoder (741.6 ms) are within 3%, so their ordering is not
stable. And naive attention misses its own roofline by 4.2×, which is
[still open](experiments/w01-5-roofline/roofline.md).

## Reproducing

```bash
pip install -U "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate av pandas

python experiments/w01-3-first-trace/first-trace.py       # the trace above
RUN_DIR=runs/<new_run> python experiments/w01-6-bottleneck/w1-6.py
python experiments/w01-5-roofline/roofline_benchmark.py         # roofline
python experiments/w01-4-architecture/qwen2.5vl-3b.py     # tensor shapes
python experiments/w01-2-baseline-video/w1-2-original.py        # clip/FPS sweep
```

Each writes a timestamped `runs/` directory recording config, seed, GPU, and
package versions. `runs/` is gitignored — profiler traces run to hundreds of MB.

## Details

| | |
|---|---|
| [Phase timings and kernels](docs/findings/phase-timings.md) | how the trace was taken, top-5 kernels, full metadata |
| [Bottleneck ranking](docs/findings/bottleneck-ranking.md) | GPU vs non-kernel time per phase |
| [Roofline](experiments/w01-5-roofline/roofline.md) | arithmetic intensity, compute/memory verdicts, open question |
| [Model architecture](docs/architecture.md) | pixels → patches → tokens, measured shapes, token scaling |
| [Clip and FPS scaling](experiments/w01-2-baseline-video/README.md) | how tokens and latency grow with video length |
| [Upstream survey](docs/findings/known_fixes.md) · [contribution log](docs/findings/contrib_notes.md) | what vLLM and SGLang already fixed, and what is open |

Timings in the clip/FPS sweep are fp16, from before the switch to bf16; its
token counts are precision-independent.
