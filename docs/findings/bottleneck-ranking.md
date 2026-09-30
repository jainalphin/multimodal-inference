# Bottleneck ranking: GPU vs host time

Supporting detail for the finding in the [main README](../../README.md).

Second pass over the [W1-3 trace](phase-timings.md). Same request:
`clip_08s.mp4` at 2 FPS, bf16, NVIDIA L4.

## Ranked by clean wall time

| Rank | Phase | Clean ms | Share | Hypothesis |
|---:|---|---:|---:|---|
| 1 | decode | 1,171.6 | 39.1% | Per-token kernel launches, not arithmetic, limit throughput. |
| 2 | preprocess | 762.3 | 25.4% | CPU video decode, frame sampling, and resize. Zero GPU work. |
| 3 | encoder | 741.6 | 24.7% | Vision encoding is serialized before generation. |
| 4 | prefill | 310.6 | 10.4% | Large multimodal prompt GEMMs; already GPU-saturated. |
| 5 | h2d | 10.4 | 0.3% | Negligible. |

Ranks 2 and 3 are within 21 ms (under 3%) — not a stable ordering from one run.
They differ in kind, though, which is what matters: one is CPU, the other GPU.

## GPU vs non-kernel time

From the traced run, splitting each phase's wall time into summed kernel time
and everything else:

| Phase | Wall ms | GPU ms | Other ms | GPU busy |
|---|---:|---:|---:|---:|
| decode | 2,128.5 | 862.3 | 1,266.2 | 40.5% |
| encoder | 1,074.6 | 582.9 | 491.7 | 54.2% |
| preprocess | 601.1 | 0.0 | 601.1 | 0.0% |
| prefill | 309.7 | 305.3 | 4.4 | 98.6% |
| h2d | 10.2 | 0.0 | 10.2 | 0.0% |

Per-phase GPU time sums to 1,750.5 ms against the 1,750.9 ms whole-trace kernel
total, so the attribution is essentially complete.

`Other ms` is wall time minus summed kernel time — rough host, sync, idle, and
overlap time, not a direct CPU-launch measurement. These are *traced* wall
times, so `Other` carries profiler overhead: treat the pattern as real and the
magnitude as an upper bound.

## Three distinct regimes

- **prefill is saturated** (98.6% busy, 4.4 ms non-kernel). Nothing to recover
  from scheduling; only doing less work helps.
- **preprocess is pure CPU** (0.0 ms GPU across 601 ms). Frame decode and
  resize on the host, blocking the request start.
- **decode is launch-bound.** Largest phase, yet 1,266 ms of traced wall time
  is non-kernel and 44% of all kernel time is GEMV across 7,844 launches.

Encoder sits between the extremes at 54.2% busy: real GPU cost plus meaningful
gaps.

## Reproducing

```bash
python experiments/w01-3-first-trace/first-trace.py
RUN_DIR=runs/<the_new_run> python experiments/w01-6-bottleneck/w1-6.py
```

The run behind this page is `runs/20260928T154432Z_trace`.
