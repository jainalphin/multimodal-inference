# Baseline: how tokens and latency scale with clip length

Supporting detail for the finding in the [main README](../../README.md).

Sweeps 2 clips × 2 FPS on an L4 and records tokens, latency, and peak memory.

```bash
python experiments/w01-2-baseline-video/w1-2-original.py
```

Writes `runs/<UTC time>_baseline/` with `results.csv`, `results.json`,
`run.json`, and `pip_freeze.txt`.

## Results

| Clip | FPS | Frames | Video tokens | Prefill ms | Decode ms/tok | TTFT ms | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| clip_08s | 1.0 | 8 | 1,196 | 170.6 | 43.0 | 525.4 | 7.4 |
| clip_08s | 2.0 | 16 | 2,392 | 336.8 | 38.6 | 1,081.6 | 7.6 |
| clip_30s | 1.0 | 30 | 4,485 | 688.7 | 38.2 | 2,155.1 | 8.0 |
| clip_30s | 2.0 | 60 | 8,970 | 1,597.6 | 38.6 | 4,532.0 | 8.9 |

Tokens scale linearly with frames, but 7.5× the tokens costs **9.4× the
prefill** and 8.6× the TTFT — prefill scales worse than linearly because
attention is quadratic in sequence length. Decode stays flat at 38–43 ms/token.
All four fit on a 24 GB L4, so no per-frame cap was needed.

Medians over 5 measured runs after 2 warmups. Peak memory includes weights.

## Frames are fused in pairs

The model's unit of visual cost is a **frame group of 2 frames**, not a frame.
At 364×644 each group is a 26×46 patch grid → 13×23 = **299 tokens per group**,
i.e. 149.5 per actual frame. The script reports both, plus
`video_tok_per_video_s` — tokens per second of clip, an input-cost measure, not
a throughput rate.

## Caveat

These timings are fp16, from before the project standardized on bf16. Token
counts are precision-independent and correct; rerun for bf16 latencies.
