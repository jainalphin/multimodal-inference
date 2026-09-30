# # W2-0: fp32 baseline on the frozen regression sets
#
# Runs Qwen2.5-VL-3B-Instruct (the project model, pinned revision) in fp32 on both regression sets, writes one {"id", "output"} line per item, then scores them.
#
# Kaggle setup
# - Accelerator: GPU T4 x2. fp32 weights are ~14 GiB, which does not fit on one T4; device_map="auto" splits the model across both.
# - Internet: on (to download the model).
# - Add your uploaded dataset. Its eval/ folder is EVAL_DIR below.
#
# Expected time: roughly 30–60 min for the 200 images and 15–30 min for the 36 videos. Both loops resume: if the session dies, rerun the cell and it skips the items already written.
#
# For a later run (fp16, an optimization), change only DTYPE and OUT_DIR.

# %%
# Run this once in its own cell first:
# !pip install -q "transformers>=5.21,<5.22" "qwen-vl-utils>=0.0.14" accelerate av

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less memory lost to fragmentation; set before torch
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"   # decodes only the sampled frames; torchvision decodes the whole clip into RAM

import json
import re
import time
from pathlib import Path

import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

# ------------------------------------------------------------------ settings (edit me)
EVAL_DIR = Path("/kaggle/input/datasets/alphinjain/w2-eval/eval")   # the uploaded eval/ folder (check the path in the Data panel)
OUT_DIR = Path("/kaggle/working/runs/fp32")     # predictions go here
DTYPE = torch.float32                           # the baseline. T4 has no native bf16, so later runs use float16

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"   # same pin as W1 and MANIFEST.json
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# ## Load the model

# %%
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
# On 2x T4, fp32 (~14 GiB) is split over both GPUs. GPU 0 also runs the vision encoder, whose fp32 attention
# needs a few GiB per page, so it gets only 7 GiB of weights and GPU 1 takes the rest.
# On one big GPU (L4 24 GB, A10G, ...) or for fp16 (~7.5 GiB), everything goes on GPU 0.
split = DTYPE == torch.float32 and torch.cuda.device_count() > 1
max_memory = {0: "7GiB", 1: "13GiB"} if split else None
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE, device_map="auto", attn_implementation="sdpa",
    max_memory=max_memory).eval()

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB of weights")

# ## One request
#
# content is the user message: one image or video entry plus the question text.
# qwen-vl-utils loads and resizes the image or samples the video frames using the limits in the entry
# (max_pixels, fps, ...). The processor therefore gets do_resize=False, so nothing is resized twice.

# %%
def ask(content, max_new_tokens):
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=14, return_video_kwargs=True, return_video_metadata=True)

    if videos:
        videos, metas = zip(*videos)
        inputs = processor(text=[text], videos=list(videos), video_metadata=list(metas),
                           do_resize=False, return_tensors="pt", **video_kwargs)
    else:
        inputs = processor(text=[text], images=images, do_resize=False, return_tensors="pt")

    inputs = inputs.to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **GREEDY)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]   # drop the prompt, keep only the answer
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    del inputs, out
    torch.cuda.empty_cache()   # hand this item's activations back before the next (differently sized) one
    return answer

# ## Image set: 200 DocVQA items
#
# The prompt, pixel limit and token limit all come from the set file, so every run uses exactly the same settings.

# %%
s = json.loads((EVAL_DIR / "sets" / "regression_image.json").read_text())
out_path = OUT_DIR / "image.jsonl"
done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()

with open(out_path, "a") as f:
    for k, it in enumerate(s["items"], 1):
        if it["id"] in done:
            continue
        content = [
            {"type": "image", "image": str(EVAL_DIR / it["path"]), "max_pixels": s["vision"]["max_pixels"]},
            {"type": "text", "text": s["prompt_template"].format(question=it["question"])},
        ]
        t0 = time.perf_counter()
        output = ask(content, s["generation"]["max_new_tokens"])
        f.write(json.dumps({"id": it["id"], "output": output}) + "\n")
        f.flush()   # written item by item, so a crash loses at most one answer
        print(f"{k:>3}/{len(s['items'])}  {time.perf_counter() - t0:5.1f}s  {output!r:<30}  gt={it['answers'][0]!r}")

# ## Video set: 36 MVBench items
#
# Options are listed as (A) ..., (B) ... in the original MVBench order. Items with start_s/end_s use only that segment.

# %%
s = json.loads((EVAL_DIR / "sets" / "regression_video.json").read_text())
v = s["vision"]
out_path = OUT_DIR / "video.jsonl"
done = {json.loads(line)["id"] for line in open(out_path)} if out_path.exists() else set()

with open(out_path, "a") as f:
    for k, it in enumerate(s["items"], 1):
        if it["id"] in done:
            continue
        video = {"type": "video", "video": str(EVAL_DIR / it["path"]),
                 "fps": v["fps"], "max_frames": v["max_frames"], "max_pixels": v["max_pixels_per_frame"]}
        if "start_s" in it:
            video["video_start"], video["video_end"] = it["start_s"], it["end_s"]
        options = "\n".join(f"({'ABCDEFGH'[i]}) {c}" for i, c in enumerate(it["candidates"]))
        content = [video, {"type": "text", "text": s["prompt_template"].format(question=it["question"], options=options)}]

        t0 = time.perf_counter()
        output = ask(content, s["generation"]["max_new_tokens"])
        f.write(json.dumps({"id": it["id"], "output": output}) + "\n")
        f.flush()
        print(f"{k:>2}/{len(s['items'])}  {time.perf_counter() - t0:5.1f}s  {output!r:<12}  gt={it['answer_letter']}  {it['task']}")

# ## Record what produced these predictions

# %%
(OUT_DIR / "run.json").write_text(json.dumps({
    "model": MODEL_ID, "revision": MODEL_REVISION, "dtype": str(DTYPE), "device_map": "auto", "attn": "sdpa", "video_reader": os.environ["FORCE_QWENVL_VIDEO_READER"],
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch": torch.__version__, "transformers": transformers.__version__,
    "manifest": json.loads((EVAL_DIR / "sets" / "MANIFEST.json").read_text())["sets"],
}, indent=2))

# ## Score
#
# This runs eval/score.py with its four settings filled in, so the scoring code is the same file you read.
# For the baseline BASELINE is None. For a later run, pass the fp32 file as baseline.
#
# Sanity check for fp32: DocVQA ANLS about 80–90 (a little under the published number, because pages are
# downscaled) and MVBench accuracy about 50–70. A number far outside that means the prompt building or the
# answer parsing is wrong. Look at the printed outputs above.

# %%
def score(set_name, predictions, baseline=None):
    src = (EVAL_DIR / "score.py").read_text()
    settings = (f'EVAL_DIR = Path("{EVAL_DIR}")\nSET_NAME = "{set_name}"\n'
                f'PREDICTIONS = "{predictions}"\nBASELINE = {repr(str(baseline)) if baseline else None}\n')
    src = re.sub(r"^EVAL_DIR = .*?^BASELINE = [^\n]*\n", settings, src, flags=re.M | re.S)
    exec(src, {})

score("regression_image", OUT_DIR / "image.jsonl")
score("regression_video", OUT_DIR / "video.jsonl")

# ## Freeze the baseline
#
# Download /kaggle/working/runs/fp32/ from the notebook's Output panel: image.jsonl, video.jsonl, run.json and the
# .score.json files. Keep them as the frozen fp32 baseline. The simplest way is to make them a second private Kaggle dataset,
# so later notebooks can attach it and pass it as baseline=.


"""
Conclusion: fp32 baseline (1x NVIDIA L4, 2026-09-30)

- DocVQA (image set: questions about scanned document pages), 200 items: 94.76 ANLS, 95% CI [91.68, 97.34]
- MVBench (video set: multiple-choice questions about short clips), 36 items: 61.11% accuracy, 95% CI [44.44, 77.78]
- Both are close to Qwen2.5-VL-3B's published numbers (DocVQA 93.9, MVBench 67.0) and inside the CIs,
  so the prompts, image and video loading, and scoring work.
- Pages were shrunk to at most ~1 million pixels (~1,280 image tokens) to fit memory. Accuracy stayed at the
  published level, so the shrinking did not hurt the model's ability to read the documents.
- The video CI is wide (+-17 points): use the video set for "answers unchanged", not for accuracy claims.
- Checks: all 36 video answers are a single letter, none of the 200 image answers is empty,
  and the set hashes match the manifest.
- Video was read with decord. torchvision decoded whole clips into RAM and ran out of memory on the
  63-76 s 1080p clips. All later runs must use decord too.
- Frozen in eval/baselines/fp32/. Every later optimization (W2-2 fp16 onward) is scored against these files.

regression_image  n=200
  ANLS                    94.76   95% CI [91.68, 97.34]
regression_video  n=36
  accuracy                61.11   95% CI [44.44, 77.78]
"""
