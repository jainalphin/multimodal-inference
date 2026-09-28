# =============================================================================
# W1-3 vLLM: first trace of ONE video request.
#
# Kaggle setup:
#   1. Enable a T4 GPU and restart the session before running this file.
#   2. Create the clip:
#      !mkdir -p clips
#      !ffmpeg -y -i /kaggle/input/datasets/alphinjain/sample-videos/videos/4.mp4 \
#          -t 8 -an clips/clip_08s.mp4
#   3. Install compatible packages for the Kaggle image/vLLM version.
#
# vLLM counterpart to the Transformers trace script — same clip, FPS, prompt,
# token limit, so results can be compared with that baseline. vLLM owns model
# execution; the hooks only annotate the vision tower and return None so they
# cannot modify model arguments.
#
# Output: printed to console only. video.trace.json (raw profiler trace, for
# https://ui.perfetto.dev) and run.json/results_trace.json (structured data
# for later comparison) are still written — there's no narrative summary file.
# =============================================================================
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Keep the engine in this process so the PyTorch profiler and hooks can see it.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
import vllm
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---- config ----
CONFIG = dict(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    model_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    patch_size=14,
    seed=1234,
    clip_path="clips/clip_08s.mp4",
    video_fps=2.0,
    video_prompt="Describe what happens in this video.",
    # qwen-vl-utils has already resized the decoded frames.  Keep the matching
    # Transformers video-budget policy explicit for vLLM's processor path.
    cap_pixels_per_frame=True,
    new_tokens=32,
    warmup=2,
    out_root="runs",
)

PHASES = ("preprocess", "sched_h2d", "encoder", "prefill", "decode")
cfg = CONFIG


# ---- phase timer ----
class Phases:
    def __init__(self):
        self.ms = {}
        self.cur = None

    def start(self, name):
        torch.cuda.synchronize()
        ctx = record_function(name)
        ctx.__enter__()
        self.cur = (name, ctx, time.perf_counter())

    def stop(self):
        if self.cur is None:
            return
        torch.cuda.synchronize()
        name, ctx, started = self.cur
        self.ms[name] = (time.perf_counter() - started) * 1e3
        ctx.__exit__(None, None, None)
        self.cur = None

    def attach(self, model):
        # Hooks must return None. Returning a tuple makes PyTorch treat it as
        # replacement positional arguments and can pass grid_thw twice.
        def before_visual(module, args):
            self.stop()                 # sched_h2d -> encoder
            self.start("encoder")
            return None

        def after_visual(module, args, output):
            self.stop()                 # encoder -> prefill
            self.start("prefill")
            return None

        model.visual.register_forward_pre_hook(before_visual)
        model.visual.register_forward_hook(after_visual)


# ---- one full request ----
def run_request(llm, processor, phases, video_id):
    engine = llm.llm_engine
    video = {"type": "video", "video": cfg["clip_path"], "fps": float(cfg["video_fps"])}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": cfg["video_prompt"]}]}]
    phases.ms = {}

    phases.start("preprocess")
    _, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=cfg["patch_size"],
        return_video_kwargs=True, return_video_metadata=True,
    )
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    llm.enqueue(
        [{"prompt": text, "multi_modal_data": {"video": videos[0][0]}}],
        SamplingParams(temperature=0.0, max_tokens=cfg["new_tokens"]),
        use_tqdm=False,
        mm_processor_kwargs={**video_kwargs, "do_resize": False,
                             "cap_pixels_per_frame": cfg["cap_pixels_per_frame"]},
    )
    phases.stop()

    phases.start("sched_h2d")
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
        # The vision hook changes sched_h2d -> encoder -> prefill. Once the
        # first engine step completes, remaining steps are token decoding.
        if phases.cur is not None and phases.cur[0] == "prefill":
            phases.stop()
            phases.start("decode")
    phases.stop()

    result = outputs[-1]
    prompt_ids = list(result.prompt_token_ids)
    generated_ids = list(result.outputs[0].token_ids)
    return {
        "ms": dict(phases.ms),
        "input_tokens": len(prompt_ids),
        "video_tokens": prompt_ids.count(video_id),
        "n_generated": len(generated_ids),
    }


# ---- warmup + traced run ----
def profile_request(run, trace_path):
    for _ in range(max(cfg["warmup"], 1)):
        clean = run()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
        record_shapes=True, profile_memory=True,
    ) as prof:
        for _ in range(2):
            traced = run()
            prof.step()
    return clean, traced


# ---- trace analysis ----
def analyze_trace(path):
    ops, kernels, ranges = {}, [], {}
    for event in json.loads(path.read_text()).get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category == "cpu_op":
            external_id = (event.get("args") or {}).get("External id")
            if external_id is not None:
                ops[external_id] = event["name"]
        elif category == "kernel":
            kernels.append(event)
        elif category == "user_annotation" and event["name"] in PHASES:
            ranges[event["name"]] = (event["ts"], event["ts"] + event["dur"])

    aggregate = {}
    for kernel in kernels:
        item = aggregate.setdefault(kernel["name"], {"us": 0.0, "calls": 0, "ops": {}})
        operation = ops.get((kernel.get("args") or {}).get("External id"), "?")
        item["us"] += kernel["dur"]
        item["calls"] += 1
        item["ops"][operation] = item["ops"].get(operation, 0) + 1

    total_us = sum(item["us"] for item in aggregate.values())
    busy = {phase: 0.0 for phase in ranges}
    for kernel in kernels:
        for phase, (start, end) in ranges.items():
            if start <= kernel["ts"] <= end:
                busy[phase] += kernel["dur"]
                break

    top = sorted(aggregate.items(), key=lambda pair: -pair[1]["us"])[:5]
    return {
        "n_kernels": len(kernels),
        "gpu_kernel_ms": total_us / 1e3,
        "busy_ms": {phase: value / 1e3 for phase, value in busy.items()},
        "top_kernels": [
            {
                "kernel": name, "ms": item["us"] / 1e3,
                "pct": 100 * item["us"] / total_us if total_us else 0.0,
                "calls": item["calls"], "op": max(item["ops"], key=item["ops"].get),
            }
            for name, item in top
        ],
    }


# ---- run everything, top to bottom ----
assert torch.cuda.is_available(), "No GPU: enable a Kaggle GPU runtime."
clip = Path(cfg["clip_path"])
if not clip.exists():
    raise FileNotFoundError(f"Video not found: {clip}. Create it with the ffmpeg setup cell in this file.")

run_dir = Path(cfg["out_root"]) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_trace_vllm"
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "run.json").write_text(json.dumps({
    "config": cfg,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "qwen_vl_utils": metadata.version("qwen-vl-utils"),
}, indent=2) + "\n")

processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
llm = LLM(
    model=cfg["model_id"],
    revision=cfg["model_revision"],
    tokenizer_revision=cfg["model_revision"],
    seed=cfg["seed"],
    dtype="half",
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=1,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    limit_mm_per_prompt={"image": 0, "video": 1},
    async_scheduling=False,
)
phases = Phases()
llm.apply_model(phases.attach)

trace_path = run_dir / "video.trace.json"
print("warmup, then one traced request")
clean, traced = profile_request(lambda: run_request(llm, processor, phases, video_id), trace_path)
analysis = analyze_trace(trace_path)
(run_dir / "results_trace.json").write_text(json.dumps(
    {"clean": clean, "traced": traced, **analysis}, indent=2
) + "\n")

# ---- print results ----
c, t = clean["ms"], traced["ms"]
total = sum(c.values())
before = sum(c.get(p, 0.0) for p in ("preprocess", "sched_h2d", "encoder", "prefill"))

print(f"\n{CONFIG['model_id']} (float16, vLLM {vllm.__version__}) on {torch.cuda.get_device_name(0)}")
print(f"{Path(cfg['clip_path']).name} at {cfg['video_fps']:g} fps: {clean['video_tokens']} video tokens, "
      f"{clean['input_tokens']} input tokens, {clean['n_generated']} generated tokens (greedy, seed {cfg['seed']})")
print(f"{analysis['n_kernels']:,} GPU kernels, {analysis['gpu_kernel_ms']:,.1f} ms total kernel time")
print("clean = no profiler; traced = inside the trace (profiler slows decode, so use clean for percentages)\n")

print("phase         clean_ms   clean_%   traced_ms   gpu_busy_ms   gpu_busy_%")
for p in PHASES:
    busy = analysis["busy_ms"].get(p)
    tp = t.get(p)
    busy_pct = f"{100 * busy / tp:.1f}" if busy is not None and tp else "-"
    print(f"{p:<13} {c.get(p, 0):>9,.1f} {100 * c.get(p, 0) / total:>8.1f} "
          f"{tp if tp is not None else 0:>10,.1f} {busy if busy is not None else 0:>12,.1f} {busy_pct:>10}")

print("\ntop-5 kernels by self GPU time")
for i, k in enumerate(analysis["top_kernels"], 1):
    name = k["kernel"] if len(k["kernel"]) <= 90 else k["kernel"][:87] + "..."
    print(f"{i}. {name}")
    print(f"   ms={k['ms']:,.1f}  pct={k['pct']:.1f}%  calls={k['calls']:,}  op={k['op']}")

print(f"\n{before:,.1f} ms = {100 * before / total:.1f}% of the request before first token")
print(f"decode: {c.get('decode', 0) / max(clean['n_generated'] - 1, 1):.1f} ms per token after the first")

print(f"\nRaw trace (open at https://ui.perfetto.dev): {trace_path}")
print(f"All outputs in: {run_dir}")
