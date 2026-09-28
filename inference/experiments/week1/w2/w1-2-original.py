# =============================================================================
# W1-2 video baseline: Qwen2.5-VL-3B-Instruct, fp16. One file: code, results, notes.
#
# RUN
#     Cell 1 (fresh session):
#         !pip install -q -U "transformers==5.17.0" "qwen-vl-utils>=0.0.14" accelerate av
#     Cell 2: paste this whole file and run (duration depends on the GPU and clips).
#     Output folder: runs/<UTC time>_baseline/  (results.csv, results.json, run.json, pip_freeze.txt)
#
# HOW IT WORKS
#     Each row is ONE request: all sampled frames of the clip go through the vision encoder and the LLM
#     together in a single forward pass (nothing is streamed).
#         frames           = duration x fps, rounded down to an even number (library limits 4 to 768), evenly spaced
#         tokens per frame = (frame height / 28) x (frame width / 28)          one token = 28x28 px
#         video tokens     = (frames / 2) x tokens per frame                   frames are fused in pairs
#         example: 30 s at 1 fps, frames resized to 364x644 -> 30 frames = 15 pairs x (13 x 23 = 299) = 4,485 tokens
#     A per-frame token cap (CAPS) shrinks the frames. For each clip x fps the caps are tried in order, and only
#     a CUDA out-of-memory error moves on to the next, smaller cap.
#
# COLUMNS (only the non-obvious ones; times are medians in ms)
#     cap_tokens_per_frame  cap used (<NA> = library default, 768)     token_check_ok  count matches the formula
#     preprocess_ms         CPU decode + sample + resize + tokenize, plus the copy to the GPU
#     encoder_ms            GPU time inside the vision tower            prefill_ms      ttft_ms - encoder_ms
#     ttft_ms               time to first token = encoder + prefill     peak_gib        includes the ~7.1 GiB of weights
#
# LATEST SUPPLIED RERUN RESULTS
#     The captured console output does not include GPU, package, warmup, repeat, or weights metadata, so do not
#     carry the historical metadata below forward. All four default-cap cases fit, including 30 s @ 2 fps.
#
#     clip      fps  cap      frames  frame_hw  tok/frame  video_tokens  tok/s
#     clip_08s  1.0  default       8  364x644         299         1,196    150
#     clip_08s  2.0  default      16  364x644         299         2,392    299
#     clip_30s  1.0  default      30  364x644         299         4,485    149
#     clip_30s  2.0  default      60  364x644         299         8,970    299
#
#     clip      fps  prefill  decode/tok      ttft  peak_gib  (timings are medians; peak includes weights)
#     clip_08s  1.0    170.6        43.0     525.4       7.4
#     clip_08s  2.0    336.8        38.6   1,081.6       7.6
#     clip_30s  1.0    688.7        38.2   2,155.1       8.0
#     clip_30s  2.0  1,597.6        38.6   4,532.0       8.9
#
# WHAT THEY SHOW
#     - Tokens grow linearly with frames: 2x fps = 2x tokens; 8 s -> 30 s at 1 fps = 3.75x.
#     - Versus 8 s @ 1 fps, TTFT is x2.1, x4.1, and x8.6; prefill is x2.0, x4.0, and x9.4.
#     - Decode is broadly flat at 38.2--43.0 ms/token; this rerun does not show a prompt-length slowdown.
#     - Peak allocated memory rises from 7.4 to 8.9 GiB, and every default-cap case fits on the rerun GPU.
#     - Row 4 is now a like-for-like doubling of row 3: it has 2x frames and 2x video tokens at the same resolution.
#
# STILL OPEN
#     - Image rows (skipped for now). p95 needs 20+ repeats (W1-5).
#     - Confirm the two clips are the real ones (SHA-256 is in run.json).
#     - Save the run folder (Save Version or download): session files vanish when the session ends.
#
# NOTES
#     - Record the package versions from each run's `run.json` before comparing measurements. The supplied rerun
#       transcript does not include them.
# =============================================================================

import gc
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from importlib import metadata

import pandas as pd
import torch
import transformers
from IPython.display import display   # Jupyter/Kaggle table display (works in a script too)
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText


# ======================================================================================
# 1. SETTINGS (edit me)
# ======================================================================================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
CLIPS = ["clips/clip_08s.mp4", "clips/clip_30s.mp4"]   # clips used by the checked-in sweep configuration
FPS_LIST = [1.0, 2.0]                     # frame sampling rates to test (frames given = duration x fps)
CAPS = [None, 256, 192, 128, 64]          # max tokens PER FRAME, tried in order (None = library default)
PROMPT = "Describe what happens in this video."
MAX_NEW_TOKENS = 64
WARMUP = 2                                # runs per case that are thrown away
REPEATS = 5                               # runs per case that are measured; report the median
SEED = 1234

# Qwen2.5-VL token math (see the header). Change only if you change the model family.
PATCH = 14                                # the vision tower cuts each frame into 14x14 px patches
MERGE = 2                                 # 2x2 neighbouring patches are merged into one LLM token
TOKEN_PX = PATCH * MERGE                  # so one token covers 28x28 px of one frame
FRAMES_PER_GROUP = 2                      # consecutive frames are fused in pairs
MIN_TOKENS_PER_FRAME = 128                # library default lower bound for frame size

OUT_DIR = "runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "Turn on a CUDA GPU in the notebook settings"
torch.manual_seed(SEED)
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| transformers", transformers.__version__)


# ======================================================================================
# 2. LOAD MODEL
# ======================================================================================
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16, attn_implementation="sdpa",   # fp16 + SDPA compatibility configuration
).to("cuda:0").eval()

weights_gib = torch.cuda.memory_allocated() / 2**30
video_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")   # placeholder id for video tokens
print(f"weights on GPU: {weights_gib:.2f} GiB")

# Time the vision tower with CUDA events. These two functions are just PyTorch hook callbacks:
# every forward call of the vision tower appends one [start, end] event pair to `enc`.
enc = []


def enc_start(module, args):
    enc.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
    enc[-1][0].record()


def enc_end(module, args, output):
    enc[-1][1].record()


model.model.visual.register_forward_pre_hook(enc_start)
model.model.visual.register_forward_hook(enc_end)

greedy = dict(do_sample=False, temperature=None, top_p=None, top_k=None)


# ======================================================================================
# 3. MEASURE ONE CASE  (the one helper function in this script)
# ======================================================================================
# Runs one (clip, fps, per-frame token cap) case WARMUP+REPEATS times and returns one result row.
# Raises torch.cuda.OutOfMemoryError if the case does not fit on the GPU.
def measure(clip, fps, cap):
    # What we ask the video reader for. A cap limits how big each frame may be.
    video = {"type": "video", "video": clip, "fps": fps}
    if cap:
        video["max_pixels"] = cap * TOKEN_PX**2
        video["min_pixels"] = min(cap, MIN_TOKENS_PER_FRAME) * TOKEN_PX**2
    messages = [{"role": "user", "content": [video, {"type": "text", "text": PROMPT}]}]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    runs = []
    with torch.inference_mode():
        for i in range(WARMUP + REPEATS):

            # --- 1) CPU preprocessing: decode, sample frames, resize, tokenize, copy to GPU
            t0 = time.perf_counter()
            _, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=PATCH, return_video_kwargs=True, return_video_metadata=True)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            video_frames, video_meta = zip(*videos)   # one entry per video (we send exactly one; ALL its frames go in together)
            inputs = processor(
                text=[text], videos=list(video_frames), video_metadata=list(video_meta),
                padding=True, return_tensors="pt", do_resize=False,   # already resized above
                **video_kwargs
            )
            t, h, w = inputs["video_grid_thw"][0].tolist()   # frame groups, height and width in 14 px patches
            n_input = inputs["input_ids"].shape[1]
            video_tokens = int((inputs["input_ids"] == video_pad_id).sum())
            inputs = inputs.to("cuda:0")
            torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - t0) * 1000

            # --- 2) time to first token = vision encoder + prefill + 1 sampled token
            enc.clear()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**inputs, max_new_tokens=1, **greedy)
            torch.cuda.synchronize()
            ttft_ms = (time.perf_counter() - t0) * 1000
            encoder_ms = sum(start.elapsed_time(end) for start, end in enc)

            # --- 3) full generation
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **greedy)
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000
            n_new = out.shape[1] - n_input
            output = processor.batch_decode(out[:, n_input:], skip_special_tokens=True)[0]

            if i >= WARMUP:
                runs.append(dict(
                    preprocess_ms=preprocess_ms,
                    encoder_ms=encoder_ms,
                    prefill_ms=ttft_ms - encoder_ms,
                    ttft_ms=ttft_ms,
                    decode_ms_per_token=(total_ms - ttft_ms) / max(n_new - 1, 1),
                ))
            del inputs, out

    # What the video reader says about the clip and the frames it picked (same on every run).
    frame_indices = [int(idx) for idx in video_meta[0]["frames_indices"]]      # which source frames were used
    duration_s = video_meta[0]["total_num_frames"] / video_meta[0]["fps"]      # source frames / source fps
    frames_sampled = len(frame_indices)
    tokens_per_frame = h * w // MERGE**2

    medians = {name: statistics.median(run[name] for run in runs) for name in runs[0]}
    return dict(
        # what the model receives
        duration_s=duration_s,
        frames_sampled=frames_sampled,
        frame_groups=t,
        sampled_fps=frames_sampled / duration_s,
        frame_hw=f"{h * PATCH}x{w * PATCH}",
        tokens_per_frame=tokens_per_frame,
        video_tokens=video_tokens,
        video_tok_per_s=video_tokens / duration_s,
        token_check_ok=(frames_sampled == t * FRAMES_PER_GROUP) and (video_tokens == t * tokens_per_frame),
        frame_indices=frame_indices,
        # speed and memory
        **medians,
        new_tokens=n_new,
        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        output=output,
        runs=runs,
    )


# ======================================================================================
# 4. RUN EVERY CLIP x FPS, LOWERING THE CAP ONLY WHEN THE GPU RUNS OUT OF MEMORY
# ======================================================================================
rows = []
for clip in CLIPS:
    for fps in FPS_LIST:
        for cap in CAPS:
            name = os.path.basename(clip)
            print(f"{name} fps={fps:g} cap={cap or 'default'} ...", end=" ", flush=True)

            try:
                result = dict(status="ok", **measure(clip, fps, cap))
            except torch.cuda.OutOfMemoryError:
                result = dict(status="oom")
            gc.collect()
            torch.cuda.empty_cache()
            print(result["status"])

            rows.append(dict(clip=name, fps_requested=fps, cap_tokens_per_frame=cap, **result))
            json.dump(rows, open(f"{OUT_DIR}/results.json", "w"), indent=2)   # saved as we go

            if result["status"] == "ok":
                break   # first cap that fits ends this clip/FPS ladder


# ======================================================================================
# 5. SHOW RESULTS
# ======================================================================================
df = pd.DataFrame(rows)

# An OOM row leaves blanks, which would turn whole-number columns into floats (8 -> 8.0). Keep them whole.
for col in ["cap_tokens_per_frame", "frames_sampled", "frame_groups", "tokens_per_frame", "video_tokens"]:
    if col in df:
        df[col] = df[col].astype("Int64")

case = ["clip", "fps_requested", "cap_tokens_per_frame", "status"]

print("\nEach row = ONE request: every frame in `frames_sampled` goes through the vision encoder "
      "and the LLM together.")

print("\nTABLE 1: what the model receives")
display(df.reindex(columns=case + [
    "duration_s", "frames_sampled", "frame_groups", "sampled_fps", "frame_hw",
    "tokens_per_frame", "video_tokens", "video_tok_per_s", "token_check_ok",
]).round(1))

print("TABLE 2: speed (ms, medians) and peak GPU memory (GiB, includes weights)")
display(df.reindex(columns=case + [
    "preprocess_ms", "encoder_ms", "prefill_ms", "decode_ms_per_token", "ttft_ms", "peak_gib",
]).round(1))

print("Does it fit on this GPU?")
for (clip, fps), attempts in df.groupby(["clip", "fps_requested"], sort=False):
    last = attempts.iloc[-1]   # the attempt that ended the ladder: ok, or smallest cap that still OOMed
    cap = last["cap_tokens_per_frame"]
    cap_text = "default cap" if pd.isna(cap) else f"{int(cap)} tokens/frame"
    print(f"  {clip} @ {fps:g} fps:", f"fits ({cap_text})" if last["status"] == "ok" else "does NOT fit at any cap tried")


# ======================================================================================
# 6. SAVE
# ======================================================================================
run_info = dict(
    model=MODEL_ID, requested_revision=MODEL_REVISION, loaded_revision=getattr(model.config, "_commit_hash", None),
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
    transformers=transformers.__version__, qwen_vl_utils=metadata.version("qwen-vl-utils"),
    seed=SEED, fps_list=FPS_LIST, caps=CAPS, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS,
    warmup=WARMUP, repeats=REPEATS, weights_gib=round(weights_gib, 2),
    token_math=dict(patch=PATCH, merge=MERGE, token_px=TOKEN_PX, frames_per_group=FRAMES_PER_GROUP),
    clip_sha256={clip: hashlib.sha256(open(clip, "rb").read()).hexdigest() for clip in CLIPS},
)
json.dump(run_info, open(f"{OUT_DIR}/run.json", "w"), indent=2)
df.drop(columns=["runs", "frame_indices"], errors="ignore").to_csv(f"{OUT_DIR}/results.csv", index=False)
open(f"{OUT_DIR}/pip_freeze.txt", "w").write(
    "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())))
print("\nSaved to", OUT_DIR)
