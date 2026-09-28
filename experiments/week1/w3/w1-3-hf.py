# =============================================================================
# W1-3: first trace of ONE video request (the ~8 s clip).
# Qwen2.5-VL-3B-Instruct fp16; intended for a Kaggle T4. One file, nothing else needed.
#
# HOW TO RUN (Kaggle)
#   cell 1 (fresh session):  !pip install -q -U "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate av
#   cell 2 (once per session): make the 8 s clip
#     !mkdir -p clips
#     !ffmpeg -y -i /kaggle/input/datasets/alphinjain/sample-videos/videos/4.mp4 -t 8 -an clips/clip_08s.mp4
#   cell 3: paste this whole file and run it (about 3 minutes). For a quick test set new_tokens=4 in CONFIG.
#
# Output: printed to console only (no trace_summary.md file). video.trace.json is still written,
# since that's the raw profiler trace you open at https://ui.perfetto.dev.
# =============================================================================
import json
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from transformers import AutoModelForImageTextToText, AutoProcessor, LogitsProcessor, LogitsProcessorList

# ---- config ----
CONFIG = dict(
    model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    model_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
    patch_size=14,                   # 14 for Qwen2.5-VL, 16 for Qwen3-VL
    clip_path="clips/clip_08s.mp4",
    video_fps=2.0,                   # sampled frames per second
    video_prompt="Describe what happens in this video.",
    # Make the Transformers 5.21 video-budget policy explicit.  qwen-vl-utils
    # resizes first below, so the processor is intentionally called with
    # do_resize=False; this flag records the matching policy and silences its
    # compatibility warning without resizing those already-prepared frames.
    cap_pixels_per_frame=True,
    new_tokens=32,                   # fewer new tokens = smaller trace file
    warmup=2,                        # untraced requests before the traced one (at least 1)
    out_dir="runs",
)

PHASES = ("preprocess", "h2d", "encoder", "prefill", "decode")
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
cfg = CONFIG


# ---- phase timer ----
# Times one phase at a time and draws it as a named bar in the trace. It syncs the GPU at every boundary so
# each phase covers its own GPU work. Hooks on the vision encoder mark "encoder" and the start of "prefill".
class Phases:
    def __init__(self, visual):
        self.ms, self.cur = {}, None
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

    def encoder_done(self, module, args, output):
        self.stop()
        self.start("prefill")


# ---- decode-start marker ----
# called once per generated token; the first call means prefill is over and decode starts
class Marker(LogitsProcessor):
    def __init__(self, ph):
        self.ph, self.first = ph, True

    def __call__(self, input_ids, scores):
        if self.first:
            self.first = False
            self.ph.stop()
            self.ph.start("decode")
        return scores


# ---- one full request ----
# open and resize the frames (CPU), copy to GPU, then encoder, prefill and decode
def run_request(model, processor, ph):
    video = {"type": "video", "video": cfg["clip_path"], "fps": float(cfg["video_fps"])}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": cfg["video_prompt"]}]}]
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    ph.ms = {}

    with torch.inference_mode():
        ph.start("preprocess")
        _, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=cfg["patch_size"], return_video_kwargs=True, return_video_metadata=True
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vids, metas = zip(*videos)
        inputs = processor(
            text=[text], videos=list(vids), video_metadata=list(metas), padding=True,
            return_tensors="pt", do_resize=False,
            cap_pixels_per_frame=cfg["cap_pixels_per_frame"], **video_kwargs
        )
        ph.stop()

        ph.start("h2d")
        inputs = inputs.to("cuda")
        ph.stop()

        n_in = inputs["input_ids"].shape[1]
        video_tokens = int((inputs["input_ids"] == video_id).sum())
        out = model.generate(
            **inputs, max_new_tokens=cfg["new_tokens"],
            logits_processor=LogitsProcessorList([Marker(ph)]), **GREEDY
        )
        ph.stop()

    return {"ms": dict(ph.ms), "video_tokens": video_tokens, "n_generated": int(out.shape[1] - n_in)}


# ---- trace analysis ----
# reads the trace file: top-5 kernels by GPU time, the op that launched each, GPU busy time per phase
def analyze_trace(path):
    ops, kernels, ranges = {}, [], {}

    for e in json.load(open(path))["traceEvents"]:
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

    by_name, busy = {}, dict.fromkeys(PHASES, 0.0)
    for k in kernels:
        row = by_name.setdefault(k["name"], {"ms": 0.0, "calls": 0, "ops": {}})
        op = ops.get((k.get("args") or {}).get("External id"), "?")
        row["ms"] += k["dur"] / 1e3
        row["calls"] += 1
        row["ops"][op] = row["ops"].get(op, 0) + 1
        for phase, (start, end) in ranges.items():
            if start <= k["ts"] <= end:
                busy[phase] += k["dur"] / 1e3

    total = sum(r["ms"] for r in by_name.values())
    top = sorted(by_name.items(), key=lambda kv: -kv[1]["ms"])[:5]

    return {
        "busy_ms": busy,
        "top_kernels": [
            {
                "kernel": n, "ms": r["ms"], "pct": 100 * r["ms"] / total,
                "calls": r["calls"], "op": max(r["ops"], key=r["ops"].get)
            }
            for n, r in top
        ],
    }


# ---- run everything, top to bottom ----
if not Path(cfg["clip_path"]).exists():
    raise FileNotFoundError(f"{cfg['clip_path']} not found: make it with the ffmpeg line at the top of this file")

run_dir = Path(cfg["out_dir"]) / (time.strftime("%Y%m%d_%H%M%S") + "_trace")
run_dir.mkdir(parents=True)
trace_path = run_dir / "video.trace.json"

processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg["model_revision"])
model = AutoModelForImageTextToText.from_pretrained(
    cfg["model_id"], revision=cfg["model_revision"], dtype=torch.float16, attn_implementation="sdpa"
).to("cuda").eval()
ph = Phases(model.model.visual)

print("warmup, then one traced request")
for _ in range(cfg["warmup"]):
    clean = run_request(model, processor, ph)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    record_shapes=True, profile_memory=True
) as prof:
    for _ in range(2):
        traced = run_request(model, processor, ph)
        prof.step()

a = analyze_trace(trace_path)
c, t = clean["ms"], traced["ms"]
total = sum(c.values())
before = sum(c[p] for p in PHASES[:4])

print(f"\n{Path(cfg['clip_path']).name} at {cfg['video_fps']:g} fps: "
      f"{clean['video_tokens']} video tokens, {clean['n_generated']} generated tokens")
print("clean = no profiler; traced = inside the trace (profiler slows decode, so use clean for percentages)\n")

print("phase        clean_ms   clean_%   traced_ms   gpu_busy_%")
for p in PHASES:
    busy = f"{100 * a['busy_ms'][p] / t[p]:.1f}" if p in PHASES[2:] else "-"
    print(f"{p:<12} {c[p]:>9,.1f} {100 * c[p] / total:>8.1f} {t[p]:>10,.1f} {busy:>10}")

print("\ntop-5 kernels by GPU time")
for i, k in enumerate(a["top_kernels"], 1):
    print(f"{i}. {k['kernel'][:80]}")
    print(f"   ms={k['ms']:,.1f}  pct={k['pct']:.1f}%  calls={k['calls']:,}  op={k['op']}")

print(f"\n{before:,.1f} ms = {100 * before / total:.1f}% of the request before first token; "
      f"encoder is {100 * c['encoder'] / before:.1f}% of that")
print(f"decode: {c['decode'] / max(clean['n_generated'] - 1, 1):.1f} ms per token after the first")

print(f"\nRaw trace (open at https://ui.perfetto.dev): {trace_path}")
