import json
import os
from pathlib import Path


# W1-6: second pass over the W1-3 trace.
# Run from repo root:
#   RUN_DIR=runs/<your_w1-3_run> python experiments/week1/w6/w1-6.py

PHASES = ("preprocess", "h2d", "sched_h2d", "encoder", "prefill", "decode")
RUN_DIR = Path(os.environ.get("RUN_DIR", "runs/YOUR_W1-3_RUN_FOLDER"))
TRACE_PATH = RUN_DIR / "video.trace.json"

if not TRACE_PATH.exists():
    raise SystemExit(f"Missing {TRACE_PATH}; set RUN_DIR to the W1-3 folder with video.trace.json.")

trace = json.load(open(TRACE_PATH))
ranges = {}
kernels = []

for event in trace.get("traceEvents", []):
    if event.get("ph") != "X":
        continue
    name = event.get("name")
    if event.get("cat") == "user_annotation" and name in PHASES:
        ranges[name] = (event["ts"], event["ts"] + event["dur"], event["dur"] / 1e3)
    if event.get("cat") == "kernel":
        kernels.append(event)

rows = []
for phase, (start, end, wall_ms) in ranges.items():
    gpu_ms = 0.0
    for kernel in kernels:
        if start <= kernel["ts"] <= end:
            gpu_ms += kernel["dur"] / 1e3
    rows.append((phase, wall_ms, gpu_ms, max(0.0, wall_ms - gpu_ms)))

if not rows:
    raise SystemExit("No W1-3 phase ranges found in video.trace.json.")

hypotheses = {
    "preprocess": "Video decode, resize, or prompt preparation may explain this time.",
    "h2d": "Check whether transfer or synchronization can be reduced.",
    "sched_h2d": "Check whether scheduling or transfer adds avoidable delay.",
    "encoder": "Check the four full-attention vision blocks individually.",
    "prefill": "Inspect the large prompt GEMMs and attention kernels.",
    "decode": "Check whether repeated small kernel launches limit token speed.",
}

total_ms = sum(row[1] for row in rows)
print("\nW1-6 bottlenecks from:", TRACE_PATH)
print("Ranked by traced phase wall time from video.trace.json.")
print()
print(f"{'Rank':>4}  {'Phase':<12} {'Wall ms':>10} {'Share':>8} {'GPU ms':>10} {'Other ms':>10}  Hypothesis")

for rank, (phase, wall_ms, gpu_ms, other_ms) in enumerate(sorted(rows, key=lambda r: r[1], reverse=True), 1):
    print(
        f"{rank:>4}  {phase:<12} {wall_ms:>10.1f} {100 * wall_ms / total_ms:>7.1f}% "
        f"{gpu_ms:>10.1f} {other_ms:>10.1f}  {hypotheses.get(phase, 'Inspect CPU and GPU work separately.')}"
    )

print()
print("Other ms = phase wall time minus summed kernel time.")
print("Treat it as rough host/sync/idle/overlap time, not direct CPU launch time.")
