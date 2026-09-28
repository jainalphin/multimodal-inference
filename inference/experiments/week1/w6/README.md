# W1-6

W1-6 is a second pass over the W1-3 profile. Use two inputs from W1-3:

1. the terminal table from `w1-3-hf.py`, especially `clean_ms`, `clean_%`,
   `traced_ms`, and `gpu_busy_%`
2. the saved Perfetto trace, `runs/<your_w1-3_run>/video.trace.json`

The current W1-3 Hugging Face script prints the clean timing table to the
terminal and saves only the raw trace file. It does not save
`results_trace.json`.

Run from the repository root:

```bash
RUN_DIR=runs/<your_w1-3_run> python experiments/week1/w6/w1-6.py
```

The script prints a traced-phase bottleneck table from `video.trace.json`.
Use the W1-3 terminal output for clean request percentages, because profiler
overhead can slow the traced request, especially decode.

Phase meanings match W1-3:

- `preprocess`: video sampling, resizing, chat-template creation, and request
  preparation
- `h2d`: explicit Hugging Face tensor transfer to CUDA
- `sched_h2d`: vLLM scheduling and internal transfer, if using the vLLM trace
- `encoder`: Qwen vision tower
- `prefill`: first language-model pass over the full multimodal prompt
- `decode`: remaining token-by-token generation

The `Other ms` column is phase wall time minus summed kernel time. Treat it as
rough host, synchronization, idle, and overlap time, not a direct CPU launch
measurement.
