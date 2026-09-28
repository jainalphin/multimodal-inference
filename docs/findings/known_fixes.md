# vLLM vision/video known fixes

As of 2026-09-24, the six-month window starts 2026-03-24.

Recent merged fixes:

- [#54231](https://github.com/vllm-project/vllm/pull/54231) — PyAV video backend removal; merged 2026-08-29.
- [#44205](https://github.com/vllm-project/vllm/pull/44205) — Qwen3-VL EVS device fix; merged 2026-06-04.

Recent open PRs, not completed fixes:

- [#55582](https://github.com/vllm-project/vllm/pull/55582) — faster GLM video frame selection.
- [#46957](https://github.com/vllm-project/vllm/pull/46957) — Qwen3-VL video-wrapper fix.
- [#44543](https://github.com/vllm-project/vllm/pull/44543) — Qwen3-Omni audio/video cache fix.
- [#53708](https://github.com/vllm-project/vllm/pull/53708) — Qwen3-Omni deepstack modality fix.
- [#46836](https://github.com/vllm-project/vllm/pull/46836) — oversized-video protection.
- [#55583](https://github.com/vllm-project/vllm/pull/55583) — skip media decoding on UUID cache hits.

Still-reported problems include long-video CPU preprocessing and GPU underutilization ([#24728](https://github.com/vllm-project/vllm/issues/24728)), multimodal expansion exceeding token limits ([#57038](https://github.com/vllm-project/vllm/issues/57038)), and redundant EPD encoder work ([#55639](https://github.com/vllm-project/vllm/issues/55639)). Check ownership before starting any of them.

## Recent PRs worth studying

These are useful recent core-vLLM PRs if you want to learn the code or offer testing/review help:

- [#55582](https://github.com/vllm-project/vllm/pull/55582) — avoid scanning every source video frame for GLM video backends; a focused CPU-performance change.
- [#55583](https://github.com/vllm-project/vllm/pull/55583) — skip media loading and decoding when a UUID-based multimodal cache hit exists; useful for repeated image/video requests.
- [#53708](https://github.com/vllm-project/vllm/pull/53708) — preserve embedding modality through the Qwen3-Omni deepstack split; a good model-specific multimodal correctness case.
- [#46836](https://github.com/vllm-project/vllm/pull/46836) — reject oversized video inputs before decoder failure; useful for media validation and regression tests.
- [#57441](https://github.com/vllm-project/vllm/pull/57441) — add video support through the Transformers backend; useful for understanding backend integration and feature coverage.

Before offering code, check the current PR status and ask the author whether they need review, a reproduction, benchmarks, or tests. Helping with validation is often the safest first contribution.
