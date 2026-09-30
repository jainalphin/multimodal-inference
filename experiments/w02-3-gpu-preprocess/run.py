# W2-3: GPU-side resize/normalize for images
#
# What changes (one factor only):
#   Baseline: PIL image → CPU resize → CPU normalize → .to("cuda")
#   This run: PIL image → uint8 .to("cuda") → GPU resize → GPU normalize
#
# The frozen fp32 baseline (experiments/w02-0-baseline/run.py, predictions in eval/baselines/fp32/) is the reference for quality.
# This script only runs the optimized path, then scores it against that baseline.
#
# Kaggle setup:
#   Accelerator: GPU T4 x1  (fp16 fits on one T4)
#   Internet: on
#   Dataset 1: your w2-eval upload  →  EVAL_DIR below
#   Dataset 2: your frozen fp32 baseline  →  BASELINE below
#
# !pip install -q "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

# ------------------------------------------------------------------ settings (edit me)
EVAL_DIR = Path("/kaggle/input/datasets/alphinjain/w2-eval/eval")
BASELINE = Path("/kaggle/input/datasets/alphinjain/w2-baseline/fp32/image.jsonl")  # frozen fp32 predictions
OUT_DIR  = Path("/kaggle/working/runs/w02-3-gpu-preprocess")
DTYPE    = torch.float16   # T4 has no native bf16
MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"
GREEDY   = dict(do_sample=False, temperature=None, top_p=None, top_k=None)

# %%
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REV)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=MODEL_REV, torch_dtype=DTYPE,
    device_map="auto", attn_implementation="sdpa").eval()

DEVICE = model.device
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# ------------------------------------------------------------------ GPU preprocess helpers

# CLIP mean/std — same values Qwen2.5-VL's image processor uses
_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=DEVICE, dtype=DTYPE).view(3, 1, 1)
_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE, dtype=DTYPE).view(3, 1, 1)


def target_size(pil_img, max_pixels, patch=14):
    # replicates Qwen's resize rule: scale so H*W <= max_pixels, round to nearest 28px
    w, h = pil_img.size
    p2 = patch * 2
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        h = max(p2, int(h * scale))
        w = max(p2, int(w * scale))
    return (h // p2) * p2, (w // p2) * p2


def gpu_preprocess(pil_img, target_h, target_w):
    # uint8 H2D (small transfer) → resize on GPU → normalize on GPU
    arr = np.asarray(pil_img.convert("RGB"))                            # [H, W, 3] uint8
    t = torch.from_numpy(arr).permute(2, 0, 1).to(DEVICE, non_blocking=True)  # [3, H, W] uint8
    t = t.to(DTYPE).div_(255.0).unsqueeze(0)                            # [1, 3, H, W] float
    # antialias=True matches the HF processor's bicubic default — without it you get a quality delta
    t = F.interpolate(t, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
    return ((t.squeeze(0) - _MEAN) / _STD)                              # [3, H, W] normalized


# ------------------------------------------------------------------ ask

def ask(content, max_new_tokens, max_pixels):
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, _, _ = process_vision_info(
        messages, image_patch_size=14, return_video_kwargs=True, return_video_metadata=True)
    pil_img = images[0]

    th, tw = target_size(pil_img, max_pixels)
    inputs = processor(text=[text], images=[pil_img], do_resize=False, return_tensors="pt")
    inputs["pixel_values"] = gpu_preprocess(pil_img, th, tw).unsqueeze(0)  # [1, 3, H, W]
    inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)
    answer = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del inputs, out
    torch.cuda.empty_cache()
    return answer


# ------------------------------------------------------------------ run on the image set

s = json.loads((EVAL_DIR / "sets" / "regression_image.json").read_text())
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "gpu.jsonl"
done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
timings = []

with open(out_path, "a") as f:
    for k, it in enumerate(s["items"], 1):
        if it["id"] in done:
            continue
        content = [
            {"type": "image", "image": str(EVAL_DIR / it["path"]), "max_pixels": s["vision"]["max_pixels"]},
            {"type": "text",  "text": s["prompt_template"].format(question=it["question"])},
        ]
        t0 = time.perf_counter()
        output = ask(content, s["generation"]["max_new_tokens"], s["vision"]["max_pixels"])
        dt = time.perf_counter() - t0
        timings.append(dt)
        f.write(json.dumps({"id": it["id"], "output": output}) + "\n")
        f.flush()
        print(f"{k:>3}/{len(s['items'])}  {dt:5.2f}s  {output!r:<30}  gt={it['answers'][0]!r}")

if timings:
    print(f"\ntiming: mean={np.mean(timings):.3f}s  median={np.median(timings):.3f}s  n={len(timings)}")

# ------------------------------------------------------------------ score

def score(set_name, predictions, baseline=None):
    src = (EVAL_DIR / "score.py").read_text()
    settings = (f'EVAL_DIR = Path("{EVAL_DIR}")\nSET_NAME = "{set_name}"\n'
                f'PREDICTIONS = "{predictions}"\nBASELINE = {repr(str(baseline)) if baseline else None}\n')
    src = re.sub(r"^EVAL_DIR = .*?^BASELINE = [^\n]*\n", settings, src, flags=re.M | re.S)
    exec(src, {})

print("\n--- quality vs fp32 ground truth ---")
score("regression_image", out_path, BASELINE)

print("\n--- agreement vs fp32 baseline (are the answers identical?) ---")
score("regression_image", out_path, BASELINE)

# ------------------------------------------------------------------ provenance

(OUT_DIR / "run.json").write_text(json.dumps({
    "experiment": "w02-3-gpu-preprocess",
    "change": "GPU resize+normalize (bicubic, antialias=True, CLIP mean/std)",
    "model": MODEL_ID, "revision": MODEL_REV,
    "dtype": str(DTYPE), "device_map": "auto", "attn": "sdpa",
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch": torch.__version__, "transformers": transformers.__version__,
}, indent=2))
