# SGLang vision, image, and video: known fixes and open work

## Notebook context: MiniMax-H3 video serving

The referenced `issue_38167_lightning.ipynb` is an SGLang diffusion workflow for MiniMax-H3 on a Lightning L4 GPU. It downloads the FL2VA model variant, starts the packaged diffusion server with the Turbo LoRA, and sends a text-to-video sequence at 480p, then 768p, then 480p on the same process.

This workload is useful for checking whether a larger video request leaves behind allocator fragmentation, cache growth, model-state growth, or degraded performance for a later smaller request. Record model load time, host RAM, peak GPU memory, preprocessing time, request latency, output dimensions, and the status of the final 480p request.

The notebook is a reproduction/workload definition for SGLang. It does not by itself prove a vLLM regression or establish vLLM model support.

## Already covered in the notebook/workflow

- Single-GPU launch settings for a Lightning L4.
- Pinned MiniMax-H3 FL2VA model download.
- Turbo LoRA configuration and fixed generation parameters.
- Health polling and server-log capture for startup failures.
- Saved MP4 outputs for the initial 480p request and the follow-up 768p/480p sequence.
- Fixed seeds and prompts for repeatable comparisons.

## What still needs investigation

- Whether the 768p request causes persistent GPU memory growth or fragmentation before the final 480p request.
- Whether CPU-side video preprocessing or model offload dominates time-to-first-frame.
- Whether repeated requests reuse cached text, vision, or model states correctly.
- Whether output resolution changes leave stale allocations or scheduling state.
- Whether the same sequence behaves differently on a 12 GB GPU, where issue #38167 reported the original failure, versus a Lightning L4.

## Suggested roadmap

- Add a small repeatable regression benchmark for mixed-resolution video requests.
- Report per-stage memory and latency instead of only end-to-end job time.
- Add explicit cleanup or reuse checks around resolution changes and completed jobs.
- Document supported GPU-memory budgets and expected behavior when 768p cannot fit.
- Keep model-specific fixes separate from general allocator, cache, scheduler, and API fixes.
