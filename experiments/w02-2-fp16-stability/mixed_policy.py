# W2-2: test the mixed-precision fix.
#
# Policy: everything in fp16, except visual.blocks.31 and visual.merger in fp32.
#   - blocks.31 is the layer that overflowed (66,995 > fp16 max 65,504)
#   - the merger must be fp32 too: block 31's output (~67,000) would overflow again if cast back to fp16
#     before merger.ln_q normalizes it. The merger's output is small, so it goes back to fp16 for the LLM.
#
# Pass = 0 NaN/Inf, docvqa_50818 answers "28.00" again, agreement with fp32 stays at 99-100%.
# fp32 reference: the fp32.jsonl that task.py saved (same input pipeline), so fp32 is not rerun here.
#
# Run twice: SET_NAME = "regression_image", then "regression_video".
import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"   # same video reader as the fp32 baseline
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ------------------------------------------------------------------ settings (edit me)
EVAL_DIR = Path("/teamspace/studios/this_studio/data/eval")
SET_NAME = "regression_image"       # then "regression_video"
RUN_DIR = Path("/teamspace/studios/this_studio/runs/w2-2") / SET_NAME   # where task.py saved fp32.jsonl
FP32_BLOCKS = [31]                  # vision blocks kept in fp32

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"

processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)

# ---------- the set (same as task.py) ----------
eval_set = json.loads((EVAL_DIR / "sets" / f"{SET_NAME}.json").read_text())
v = eval_set["vision"]
regression_set = []
for it in eval_set["items"]:
    path = str(EVAL_DIR / it["path"])
    if SET_NAME == "regression_image":
        entry = {"type": "image", "image": path, "max_pixels": v["max_pixels"]}
        prompt = eval_set["prompt_template"].format(question=it["question"])
    else:
        entry = {"type": "video", "video": path, "fps": v["fps"],
                 "max_frames": v["max_frames"], "max_pixels": v["max_pixels_per_frame"]}
        if "start_s" in it:
            entry["video_start"], entry["video_end"] = it["start_s"], it["end_s"]
        options = "\n".join(f"({'ABCDEFGH'[i]}) {c}" for i, c in enumerate(it["candidates"]))
        prompt = eval_set["prompt_template"].format(question=it["question"], options=options)
    regression_set.append((entry, prompt))


def build_inputs(entry, question, device, dtype):
    messages = [{"role": "user", "content": [entry, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=14, return_video_kwargs=True, return_video_metadata=True)
    if videos:
        videos, metas = zip(*videos)
        inputs = processor(text=[text], videos=list(videos), video_metadata=list(metas),
                           do_resize=False, return_tensors="pt", **video_kwargs)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(dtype)
    else:
        inputs = processor(text=[text], images=images, do_resize=False, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs.to(device)


# ---------- the mixed model ----------
def to_fp32(x):
    """Cast floating tensors (also inside tuples, e.g. the rotary cos/sin) to fp32."""
    if torch.is_tensor(x) and x.is_floating_point():
        return x.float()
    if isinstance(x, tuple):
        return tuple(to_fp32(t) for t in x)
    return x


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, revision=model_revision, torch_dtype=torch.float16).to("cuda").eval()
visual = model.model.visual

for b in FP32_BLOCKS:
    visual.blocks[b].float()
    # the block receives fp16 from block 30: cast its inputs to fp32
    visual.blocks[b].register_forward_pre_hook(
        lambda m, args, kwargs: (to_fp32(args), {k: to_fp32(x) for k, x in kwargs.items()}), with_kwargs=True)
visual.merger.float()   # receives block 31's fp32 output directly
# the merger's output is small (~80), so hand it back to the LLM in fp16
visual.merger.register_forward_hook(lambda m, args, out: out.half())

# ---------- per-layer absmax + NaN/Inf, for every layer (same columns as task.py's stability CSV) ----------
stats = {}


def check(name):
    def hook(module, args, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if not (torch.is_tensor(t) and t.is_floating_point()):
            return
        s = stats.setdefault(name, {"absmax": 0.0, "nan": 0, "inf": 0})
        s["nan"] += torch.isnan(t).sum().item()
        s["inf"] += torch.isinf(t).sum().item()
        a = t.abs().float().masked_fill_(~torch.isfinite(t), 0)   # non-finite values count as 0 for absmax
        s["absmax"] = max(s["absmax"], a.max().item())
    return hook


for name, m in model.named_modules():
    if len(list(m.children())) == 0:
        m.register_forward_hook(check(name))

# ---------- generate ----------
answers, times = [], []
for k, (entry, q) in enumerate(regression_set, 1):
    inputs = build_inputs(entry, q, model.device, torch.float16)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        g = model.generate(**inputs, max_new_tokens=eval_set["generation"]["max_new_tokens"], do_sample=False)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
    answers.append(processor.decode(g[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    del inputs, g
    torch.cuda.empty_cache()
    print(f"{k}/{len(regression_set)}", end="\r")

with open(RUN_DIR / "mixed.jsonl", "w") as f:
    for it, a in zip(eval_set["items"], answers):
        f.write(json.dumps({"id": it["id"], "output": a}) + "\n")

# ---------- results ----------
fp32 = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(RUN_DIR / "fp32.jsonl")}
print(f"\n============ MIXED POLICY ({SET_NAME}): fp16 + blocks {FP32_BLOCKS} and merger in fp32 ============")
import pandas as pd
df = pd.DataFrame([{"module": n, **v} for n, v in stats.items()])
df.to_csv(RUN_DIR / "stability_mixed.csv", index=False)   # compare with stability_per_module.csv (the fp16 run)
bad = df[(df.nan > 0) | (df.inf > 0)]
print(f"NaN/Inf: {len(bad)} of {len(df)} modules", bad.module.head(5).tolist())
peak = df.loc[df.module == f"model.visual.blocks.{FP32_BLOCKS[-1]}.mlp.down_proj", "absmax"].item()
print(f"blocks.{FP32_BLOCKS[-1]}.mlp.down_proj peak: {peak:,.0f}  (now fp32, so no 65,504 limit)")
print(f"mean time per item: {sum(times) / len(times):.2f} s")
changed = [(it["id"], fp32[it["id"]], a) for it, a in zip(eval_set["items"], answers) if a != fp32[it["id"]]]
print(f"changed vs fp32: {len(changed)} of {len(answers)}")
for i, a32, am in changed:
    print(f"  {i}: fp32 {a32!r} -> mixed {am!r}")

# score.py with its four settings filled in: accuracy, paired difference and agreement vs fp32, with CIs
src = (EVAL_DIR / "score.py").read_text()
src = re.sub(r"^EVAL_DIR = .*?^BASELINE = [^\n]*\n",
             f'EVAL_DIR = Path("{EVAL_DIR}")\nSET_NAME = "{SET_NAME}"\n'
             f'PREDICTIONS = "{RUN_DIR / "mixed.jsonl"}"\nBASELINE = "{RUN_DIR / "fp32.jsonl"}"\n',
             src, flags=re.M | re.S)
exec(src, {})


"""
RESULTS: mixed policy (fp16, but visual.blocks.31 + visual.merger in fp32), 1x L4, decord

============ IMAGE (200 DocVQA pages): PASS ============

- NaN/Inf: 0 modules (plain fp16 had NaN in 404 modules).
- blocks.31.mlp.down_proj peak: 66,864. Above fp16's 65,504, but held in fp32 now, so no overflow.
- docvqa_50818: answers "28.00" again (plain fp16 gave "!!!!!!!!").
- Changed vs fp32: 1 of 200, only docvqa_46238 ("Grady, 1995" -> "6471"): fp16 rounding, not fixed by this policy.
- ANLS 94.26 vs fp32 94.76 (plain fp16: 93.76). Difference -0.50 [-1.50, 0.00], not significant.
- Agreement with fp32: 99.5% (plain fp16: 99.0%).
- Time: 2.96 s per page.

Output:
NaN/Inf: 0 of 626 modules []
blocks.31.mlp.down_proj peak: 66,864  (now fp32, so no 65,504 limit)
mean time per item: 2.96 s
changed vs fp32: 1 of 200
  docvqa_46238: fp32 'Grady, 1995' -> mixed '6471'
regression_image  n=200
  ANLS                    94.26   95% CI [91.05, 97.02]
  baseline ANLS           94.76
  difference (paired)     -0.50   95% CI [-1.50, +0.00]   includes 0: no significant change
  agreement with fp32     99.50   95% CI [98.50, 100.00]   (1 of 200 answers changed)

============ VIDEO (36 MVBench clips): PASS ============

- NaN/Inf: 0 modules.
- blocks.31.mlp.down_proj peak: 57,008, held in fp32 (plain fp16 came within 13% of the limit here).
- Changed vs fp32: 0 of 36. Accuracy 61.11% in both. Agreement 100%.
- Time: 2.66 s per clip.

Output:
NaN/Inf: 0 of 626 modules []
blocks.31.mlp.down_proj peak: 57,008  (now fp32, so no 65,504 limit)
mean time per item: 2.66 s
changed vs fp32: 0 of 36
regression_video  n=36
  accuracy                61.11   95% CI [44.44, 77.78]
  baseline accuracy       61.11
  difference (paired)     +0.00   95% CI [+0.00, +0.00]   includes 0: no significant change
  agreement with fp32    100.00   95% CI [100.00, 100.00]   (0 of 36 answers changed)

============ CONCLUSION ============

The mixed policy fixes the fp16 overflow on both sets: 0 NaN/Inf, the broken page answers correctly again,
and the only remaining difference from fp32 is 1 ordinary rounding flip in 236 items.
"""
