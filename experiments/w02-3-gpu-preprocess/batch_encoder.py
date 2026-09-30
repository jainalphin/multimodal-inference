# W2-4: Batch the vision encoder across frames
#
# What changes (one factor only):
#   Baseline: frames go through the vision encoder a few at a time (however the model packs them)
#   This run: stack all N frames into one (N, C, H, W) batch → one encoder forward call
#
# Sweep N = 1, 4, 8, 16 — record encoder latency per frame and peak memory at each.
# If GEMV turns into GEMM in the profile, the batch size is big enough to fill the tensor cores.
#
# This is a video experiment — uses regression_video (36 MVBench clips), not the image set.
# Quality is scored against the frozen fp32 baseline from w02-0-baseline.
#
# Kaggle setup:
#   Accelerator: GPU T4 x1  (fp16 ~7.5 GiB fits on one T4)
#   Internet: on
#   Dataset 1: your w2-eval upload  →  EVAL_DIR below
#   Dataset 2: your frozen fp32 baseline  →  BASELINE below
#
# !pip install -q "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate av

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

# ------------------------------------------------------------------ settings (edit me)
EVAL_DIR = Path("/kaggle/input/datasets/alphinjain/w2-eval/eval")
BASELINE = Path("/kaggle/input/datasets/alphinjain/w2-baseline/fp32/video.jsonl")  # frozen fp32 video predictions
OUT_DIR  = Path("/kaggle/working/runs/w02-4-batch-encoder")
DTYPE    = torch.float16   # T4 has no native bf16
MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"
GREEDY   = dict(do_sample=False, temperature=None, top_p=None, top_k=None)

# Batch sizes to sweep. Each is one separate run over the full video set.
BATCH_SIZES = [1, 4, 8, 16]

# %%
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=MODEL_REV, torch_dtype=DTYPE,
    device_map="auto", attn_implementation="sdpa").eval()

DEVICE = model.device
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(f"weights: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

# ------------------------------------------------------------------ encoder timing hook
# Records GPU time for every vision encoder forward call using CUDA events.
# Attached once; cleared before each request.
_enc_events = []

def _enc_pre(module, args):
    pair = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    pair[0].record()
    _enc_events.append(pair)

def _enc_post(module, args, output):
    _enc_events[-1][1].record()

model.model.visual.register_forward_pre_hook(_enc_pre)
model.model.visual.register_forward_hook(_enc_post)


def encoder_ms():
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in _enc_events)


# ------------------------------------------------------------------ ask
# encoder_batch_size controls how many frames the vision tower sees per forward call.
# The model's default packs all frames together; setting a smaller batch splits them.

def ask(content, max_new_tokens, encoder_batch_size):
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=14, return_video_kwargs=True, return_video_metadata=True)
    videos, metas = zip(*videos)
    inputs = processor(text=[text], videos=list(videos), video_metadata=list(metas),
                       do_resize=False, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(DEVICE)
    n_frames = inputs["pixel_values_videos"].shape[0]

    _enc_events.clear()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        if encoder_batch_size >= n_frames:
            # one shot — all frames in a single encoder call (the model default)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)
        else:
            # split pixel_values into chunks, encode each chunk, concatenate, then generate
            chunks = inputs["pixel_values_videos"].split(encoder_batch_size, dim=0)
            grid = inputs["video_grid_thw"]   # [n_frame_groups, 3]
            encoded_chunks = []
            for chunk in chunks:
                # encode this batch of frames through the vision tower only
                enc_out = model.model.visual(chunk)   # [chunk_n_patches, hidden]
                encoded_chunks.append(enc_out)
            # replace pixel_values with the pre-encoded features so generate doesn't re-encode
            inputs["image_features"] = torch.cat(encoded_chunks, dim=0)
            del inputs["pixel_values_videos"]
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)

    enc_ms = encoder_ms()
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    answer = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del inputs, out
    torch.cuda.empty_cache()
    return answer, enc_ms, n_frames, peak_gib


# ------------------------------------------------------------------ sweep over batch sizes

s = json.loads((EVAL_DIR / "sets" / "regression_video.json").read_text())
v = s["vision"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

sweep_timing = {}   # batch_size → list of (enc_ms, n_frames)

def score(set_name, predictions, baseline=None):
    src = (EVAL_DIR / "score.py").read_text()
    settings = (f'EVAL_DIR = Path("{EVAL_DIR}")\nSET_NAME = "{set_name}"\n'
                f'PREDICTIONS = "{predictions}"\nBASELINE = {repr(str(baseline)) if baseline else None}\n')
    src = re.sub(r"^EVAL_DIR = .*?^BASELINE = [^\n]*\n", settings, src, flags=re.M | re.S)
    exec(src, {})

for batch_size in BATCH_SIZES:
    out_path = OUT_DIR / f"batch{batch_size}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    timings = []

    print(f"\n{'='*60}")
    print(f"batch_size={batch_size}")
    print(f"{'='*60}")

    with open(out_path, "a") as f:
        for k, it in enumerate(s["items"], 1):
            if it["id"] in done:
                continue
            video = {"type": "video", "video": str(EVAL_DIR / it["path"]),
                     "fps": v["fps"], "max_frames": v["max_frames"],
                     "max_pixels": v["max_pixels_per_frame"]}
            if "start_s" in it:
                video["video_start"], video["video_end"] = it["start_s"], it["end_s"]
            options = "\n".join(f"({'ABCDEFGH'[i]}) {c}" for i, c in enumerate(it["candidates"]))
            content = [video, {"type": "text", "text": s["prompt_template"].format(
                question=it["question"], options=options)}]

            try:
                t0 = time.perf_counter()
                output, enc_ms, n_frames, peak_gib = ask(content, s["generation"]["max_new_tokens"], batch_size)
                dt = time.perf_counter() - t0
                timings.append((enc_ms, n_frames, peak_gib, dt))
                f.write(json.dumps({"id": it["id"], "output": output}) + "\n")
                f.flush()
                print(f"{k:>2}/{len(s['items'])}  enc={enc_ms:.0f}ms  {n_frames}frames  "
                      f"peak={peak_gib:.1f}GiB  total={dt:.1f}s  {output!r:<8}  gt={it['answer_letter']}")
            except torch.cuda.OutOfMemoryError:
                print(f"{k:>2}/{len(s['items'])}  OOM at batch_size={batch_size}")
                gc.collect()
                torch.cuda.empty_cache()

    sweep_timing[batch_size] = timings

# ------------------------------------------------------------------ timing summary across batch sizes

print("\n\n--- encoder batch sweep summary ---")
print(f"{'batch':>6}  {'enc ms/frame':>13}  {'peak GiB':>9}  {'n items':>7}")
for bs, rows in sweep_timing.items():
    if not rows:
        print(f"{bs:>6}  {'OOM':>13}  {'—':>9}  {0:>7}")
        continue
    enc_per_frame = [enc / n for enc, n, _, _ in rows]
    peaks = [p for _, _, p, _ in rows]
    print(f"{bs:>6}  {np.mean(enc_per_frame):>13.1f}  {np.mean(peaks):>9.2f}  {len(rows):>7}")

# ------------------------------------------------------------------ quality for each batch size

print("\n--- quality vs fp32 ground truth ---")
for batch_size in BATCH_SIZES:
    out_path = OUT_DIR / f"batch{batch_size}.jsonl"
    if not out_path.exists():
        continue
    print(f"\nbatch_size={batch_size}")
    score("regression_video", out_path, BASELINE)

# ------------------------------------------------------------------ provenance

(OUT_DIR / "run.json").write_text(json.dumps({
    "experiment": "w02-4-batch-encoder",
    "change": "vision encoder batched across frames, sweep N=[1,4,8,16]",
    "model": MODEL_ID, "revision": MODEL_REV,
    "dtype": str(DTYPE), "device_map": "auto", "attn": "sdpa",
    "batch_sizes": BATCH_SIZES,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch": torch.__version__, "transformers": transformers.__version__,
}, indent=2))
