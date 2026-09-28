# SGLang vision/video contribution notes

## Starting point

Use `issue_38167_lightning.ipynb` as the baseline workload. It runs MiniMax-H3 text-to-video through SGLang’s diffusion API and exercises 480p → 768p → 480p on one server process.

Before changing code:

1. Reproduce the sequence on current SGLang.
2. Capture GPU and host memory before and after each request.
3. Save server logs and job status for every request.
4. Check whether the failure is model loading, video decoding, allocator state, cache state, or API behavior.
5. Compare a fresh server against the same server after the 768p request.

## Good first contribution

Turn the notebook sequence into a focused regression or benchmark script. Keep the prompt, seed, duration, resolution, and inference steps fixed. The first version should report:

- startup/load duration;
- peak GPU and host memory;
- preprocessing and generation duration;
- output dimensions and file size;
- success/failure for each request; and
- whether the final 480p request differs from a fresh-server 480p baseline.

This gives maintainers a reproducible performance or memory report before proposing a fix.

## Possible implementation areas

- Mixed-resolution allocation cleanup after completed video jobs.
- Bounded CPU/GPU offload state for large video models.
- Reuse of decoded frames, text embeddings, or vision features where the cache key is correct.
- Better error messages when the requested resolution exceeds the available memory budget.
- Regression tests for repeated video jobs and post-failure cleanup.

## Contribution boundaries

The notebook uses SGLang, not vLLM. Keep SGLang issues, fixes, and measurements in this file and the companion `sglang_known_fixes.md`. Do not present the notebook’s behavior as evidence of a vLLM bug without reproducing it in vLLM separately.
