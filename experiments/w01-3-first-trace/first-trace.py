# =============================================================================
# W1-3: first trace of ONE video request (the ~8 s clip).
# Qwen2.5-VL-3B-Instruct; target GPU is an NVIDIA L4. One file, nothing else needed.
#
# HOW TO RUN
#   install once:  pip install -U "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate av
#   make the clip: ffmpeg -y -i <source>.mp4 -t 8 -an clips/clip_08s.mp4
#   run:           python experiments/w01-3-first-trace/first-trace.py
#   Takes about 3 minutes. For a quick test set new_tokens=4 in CONFIG.
#
# PRECISION: bf16 throughout. The checked-in phase timings in the README and in
# docs/findings/phase-timings.md were measured in fp16 (an earlier T4 target,
# since T4 has no bf16), so they need a rerun to match this default.
#
# Outputs go to runs/<time>_trace/ :
#   run.json  video.trace.json  results_trace.json  trace_summary.md
# Open video.trace.json at https://ui.perfetto.dev for the screenshot.
# =============================================================================
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
import transformers
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList

# ------------------------------------------------------------------- CONFIG (edit me)
CONFIG = dict(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",   # bigger model? see the doubts above
    model_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    patch_size=14,                   # 14 for Qwen2.5-VL, 16 for Qwen3-VL
    seed=1234,
    clip_path="clips/clip_08s.mp4",
    video_fps=2.0,                   # sampled frames per second
    video_prompt="Describe what happens in this video.",
    # qwen-vl-utils resizes the frames before the processor below.  The
    # processor therefore has do_resize=False; make the matching Transformers
    # video-budget policy explicit to silence its compatibility warning.
    cap_pixels_per_frame=True,
    dtype="bfloat16",                # bf16 is native on L4 and is the policy for all W1 experiments
    new_tokens=32,                   # fewer new tokens = smaller trace file; decode % is for this many tokens
    warmup=2,                        # untraced requests before the traced one (at least 1)
    out_root="runs",
)

PHASES = ("preprocess", "h2d", "encoder", "prefill", "decode")
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None)


# Times one phase at a time. It syncs the GPU at each boundary so a phase covers its own GPU work, and it
# opens a named range so the phase shows up as a bar in the trace. Hooks on the vision tower mark the
# encoder and the start of prefill; the first generated token marks the start of decode.
class Phases:
    def __init__(self, visual):
        self.ms, self.cur = {}, None
        # the hook that runs just before the encoder
        visual.register_forward_pre_hook(lambda module, args: self.start("encoder"))
        visual.register_forward_hook(self.encoder_done)

    def start(self, name):
        torch.cuda.synchronize()
        ctx = record_function(name)
        ctx.__enter__()
        self.cur = (name, ctx, time.perf_counter())

    def stop(self):
        torch.cuda.synchronize()
        name, ctx, t0 = self.cur
        self.ms[name] = (time.perf_counter() - t0) * 1e3
        ctx.__exit__(None, None, None)
        self.cur = None

    def encoder_done(self, module, args, output):
        self.stop()
        # the hook that runs just after the encoder
        self.start("prefill")

    def first_token(self):
        if self.cur is not None and self.cur[0] == "prefill":
            self.stop()
            self.start("decode")


# called once per generated token; the first call marks the end of prefill
class Marker(LogitsProcessor):
    def __init__(self, phases):
        self.phases = phases

    def __call__(self, input_ids, scores):
        self.phases.first_token()
        return scores


# one full request: decode + resize the frames (CPU), copy to GPU, encoder, prefill, decode
def run_request(model, processor, ph, cfg):
    video = {"type": "video", "video": cfg["clip_path"], "fps": float(cfg["video_fps"])}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": cfg["video_prompt"]}]}]
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    ph.ms = {}
    with torch.inference_mode():
        ph.start("preprocess")            # ---- STEP 1: PREPROCESS (CPU) ----
        _, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=cfg["patch_size"], return_video_kwargs=True, return_video_metadata=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vids, metas = zip(*videos)
        inputs = processor(text=[text], videos=list(vids), video_metadata=list(metas), padding=True,
                           return_tensors="pt", do_resize=False,
                           cap_pixels_per_frame=cfg["cap_pixels_per_frame"], **video_kwargs)
        ph.stop()
        ph.start("h2d")
        inputs = inputs.to("cuda")
        ph.stop()
        n_in = inputs["input_ids"].shape[1]
        video_tokens = int((inputs["input_ids"] == video_id).sum())
        out = model.generate(**inputs, max_new_tokens=cfg["new_tokens"],
                             logits_processor=LogitsProcessorList([Marker(ph)]), **GREEDY)
        ph.stop()
    return {"ms": dict(ph.ms), "input_tokens": int(n_in), "video_tokens": video_tokens,
            "n_generated": int(out.shape[1] - n_in)}


# untraced warmup requests (the last one is the "clean" result), then one traced request
def profile_request(run, trace_path, cfg):
    for _ in range(max(cfg["warmup"], 1)):
        clean = run()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
                 on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
                 record_shapes=True, profile_memory=True) as prof:
        for _ in range(2):
            traced = run()
            prof.step()
    return clean, traced


# reads the exported trace: top-5 kernels by GPU time, the op that launched each, GPU busy time per phase
def analyze_trace(path):
    ops, kernels, ranges = {}, [], {}
    for e in json.load(open(path)).get("traceEvents", []):
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "cpu_op":
            eid = (e.get("args") or {}).get("External id")
            if eid is not None:
                ops[eid] = e["name"]
        elif e.get("cat") == "kernel":
            kernels.append(e)
        elif e.get("cat") == "user_annotation" and e["name"] in PHASES:
            ranges[e["name"]] = (e["ts"], e["ts"] + e["dur"])
    agg = {}
    for k in kernels:
        a = agg.setdefault(k["name"], {"us": 0.0, "calls": 0, "ops": {}})
        op = ops.get((k.get("args") or {}).get("External id"), "?")
        a["us"] += k["dur"]
        a["calls"] += 1
        a["ops"][op] = a["ops"].get(op, 0) + 1
    total_us = sum(a["us"] for a in agg.values())
    busy = {p: 0.0 for p in ranges}
    for k in kernels:
        for p, (start, end) in ranges.items():
            if start <= k["ts"] <= end:
                busy[p] += k["dur"]
                break
    top = sorted(agg.items(), key=lambda kv: -kv[1]["us"])[:5]
    return {"n_kernels": len(kernels), "gpu_kernel_ms": total_us / 1e3,
            "busy_ms": {p: v / 1e3 for p, v in busy.items()},
            "top_kernels": [{"kernel": n, "ms": a["us"] / 1e3, "pct": 100 * a["us"] / total_us,
                             "calls": a["calls"], "op": max(a["ops"], key=a["ops"].get)} for n, a in top]}


def fmt(x):
    return "–" if x is None else f"{x:,.1f}"


def build_summary(cfg, clean, traced, a):
    c, t = clean["ms"], traced["ms"]
    total = sum(c.values())
    L = [f"# First trace: {cfg['model_id']} ({cfg['dtype']}, sdpa) on {torch.cuda.get_device_name(0)}", "",
         f"- {Path(cfg['clip_path']).name} at {cfg['video_fps']:g} fps: {clean['video_tokens']} video tokens, "
         f"{clean['input_tokens']} input tokens, {clean['n_generated']} generated tokens (greedy, seed {cfg['seed']})",
         f"- {a['n_kernels']:,} GPU kernels, {a['gpu_kernel_ms']:,.1f} ms total kernel time",
         "- `clean` = no profiler; `traced` = inside the trace (the profiler slows decode, so use clean for percentages)",
         "", "## 1. Where the time goes", "",
         "| phase | clean ms | clean % | traced ms | GPU busy ms | GPU busy % of traced phase |", "|---|---|---|---|---|---|"]
    for p in PHASES:
        busy = a["busy_ms"].get(p)
        L.append(f"| {p} | {fmt(c[p])} | {fmt(100 * c[p] / total)} | {fmt(t[p])} | {fmt(busy)} | "
                 f"{fmt(100 * busy / t[p]) if busy is not None else '–'} |")
    L += ["", "## 2. Top-5 kernels by self GPU time", "",
          "| # | kernel | GPU ms | % of kernel time | calls | op |", "|---|---|---|---|---|---|"]
    for i, k in enumerate(a["top_kernels"], 1):
        name = k["kernel"] if len(k["kernel"]) <= 90 else k["kernel"][:87] + "..."
        L.append(f"| {i} | `{name}` | {fmt(k['ms'])} | {fmt(k['pct'])} | {k['calls']:,} | `{k['op']}` |")
    before = sum(c[p] for p in ("preprocess", "h2d", "encoder", "prefill"))
    L += ["", "## 3. First change to try and why", "", "TODO: write after reading the two tables above.", "",
          "## Time before the first token (clean run)", "",
          f"- {fmt(before)} ms = {fmt(100 * before / total)}% of the request; the encoder is "
          f"{fmt(100 * c['encoder'] / before)}% of that",
          f"- decode: {fmt(c['decode'] / max(clean['n_generated'] - 1, 1))} ms per token after the first", "",
          "TODO: compare this pre-first-token delay with a chunked or streamed video path.", ""]
    return "\n".join(L)


def main(cfg=None):
    cfg = dict(CONFIG if cfg is None else cfg)
    assert torch.cuda.is_available(), "No CUDA GPU visible; this script targets an NVIDIA L4."
    if not Path(cfg["clip_path"]).exists():
        raise FileNotFoundError(f"{cfg['clip_path']} not found: make it with the ffmpeg line at the top of this file")
    torch.manual_seed(cfg["seed"])
    run_dir = Path(cfg["out_root"]) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_trace"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(
        {"config": cfg, "gpu": torch.cuda.get_device_name(0),
         "compute_capability": list(torch.cuda.get_device_capability(0)),
         "torch": torch.__version__,
         "transformers": transformers.__version__, "qwen_vl_utils": metadata.version("qwen-vl-utils")},
        indent=2) + "\n")

    processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], dtype=getattr(torch, cfg["dtype"]),
        attn_implementation="sdpa").to("cuda").eval()
    print(f"model loaded, {torch.cuda.memory_allocated() // 2**20} MiB resident")
    ph = Phases(model.model.visual)

    trace_path = run_dir / "video.trace.json"
    print("warmup, then one traced request")
    clean, traced = profile_request(lambda: run_request(model, processor, ph, cfg), trace_path, cfg)
    analysis = analyze_trace(trace_path)

    (run_dir / "results_trace.json").write_text(json.dumps({"clean": clean, "traced": traced, **analysis}, indent=2) + "\n")
    summary = build_summary(cfg, clean, traced, analysis)
    (run_dir / "trace_summary.md").write_text(summary)
    print("\n" + summary)
    print(f"\nAll outputs in: {run_dir}")


if __name__ == "__main__":
    main()
