# Contribution notes

Running log for upstream contribution work. The survey of already-merged fixes
and candidate issues lives in [known_fixes.md](known_fixes.md). This file
tracks *my* engagement: what I have read, what I have tried to reproduce, and
what I have actually said in a thread.

Plan tasks that read from this file: **W4-2** (ranked contribution list),
**W4-3** (open an issue with a reproducer), **W6-3** (choose the substantive
contribution).

## Where my own evidence points

Week 1 measured a video request end to end on an L4. The two findings with
upstream relevance:

1. **Pre-first-token time is dominated by CPU preprocessing plus the vision
   encoder** — 60.9% of the request, of which the encoder is 40.6%. The
   762 ms preprocess phase does **zero GPU work**
   ([phase-timings.md](phase-timings.md)). This lines up with vLLM issue
   [#24728](https://github.com/vllm-project/vllm/issues/24728) (long-video CPU
   preprocessing and GPU underuse), the strongest match in the tracker.
2. **Decode is the largest single phase (39.1%) but only 40.5% GPU busy**, and
   44% of all kernel time is GEMV across 7,844 calls
   ([bottleneck-ranking.md](bottleneck-ranking.md)). That is a launch-overhead
   signature rather than an arithmetic limit.

Neither is yet a reproducer against vLLM or SGLang — both numbers come from
plain Transformers. W3-5 sets up both engines; the gap table in W4-1 is what
turns these into a claim about an engine.

## Reproduce queue

Ranked by fit with my setup (Qwen2.5-VL-3B, T4/L4). Details in
[known_fixes.md](known_fixes.md). Check each thread again before starting.

| # | Issue | Why it fits | What to run |
|---:|---|---|---|
| 1 | vLLM [#41343](https://github.com/vllm-project/vllm/issues/41343) fp8_e5m2 KV cache corrupts Qwen2.5-VL | Same silent-corruption check as W2-2 | Frozen eval set with default vs fp8 KV cache, score both with score.py |
| 2 | vLLM [#57740](https://github.com/vllm-project/vllm/issues/57740) `<\|image_pad\|>` in user text | Exact model; CPU-only processor check | Prompt containing the literal token plus one image; compare with transformers [#47217](https://github.com/huggingface/transformers/issues/47217) |
| 3 | SGLang [#40072](https://github.com/sgl-project/sglang/issues/40072) extra CUDA contexts | One GPU is enough | Serve Qwen2.5-VL-3B, send one image, list processes in nvidia-smi |
| 4 | vLLM [#47860](https://github.com/vllm-project/vllm/issues/47860) / SGLang [#38676](https://github.com/sgl-project/sglang/issues/38676) preprocessing metrics | My 762 ms CPU preprocessing finding | Measure pre-engine time in each engine for the W1-3 clip (W4-1) |

## Thread log

Nothing posted upstream yet. No issues opened, no comments made, no PRs.

| Date | Repo | Thread | What I did | Outcome |
|---|---|---|---|---|
| — | — | — | — | — |

## Before engaging on any issue

- Re-check the assignee, recent comments, and any linked branches — the tracker
  snapshot may be stale.
- The two open vLLM PRs in the tracker (#55582, #55583) are already claimed;
  treat them as review-or-benchmark only.
- Have a runnable reproducer and L4 numbers ready before opening anything, per
  W4-3.

## Open questions

- Does the CPU-preprocessing cost measured in Transformers survive in vLLM and
  SGLang, or have their preprocessing paths already addressed it? (W4-1)
- Is the GEMV/launch-overhead decode signature visible inside an engine at
  production batch, or is it an artifact of batch size 1? (W6-4)
