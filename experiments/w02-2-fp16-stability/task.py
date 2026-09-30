"""
The question:

Can Qwen2.5-VL run in fp16 instead of fp32 without breaking? fp16 uses half the memory and runs faster, but it can't store numbers above 65,504.

"""



import os
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"   # same video reader as the fp32 baseline (experiments/w02-0-baseline/run.py)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less memory lost to fragmentation; set before torch

import gc
import json
import re
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"   # same pin as eval/sets/MANIFEST.json
FP16_MAX = 65504.0
FP16_TINY = 6.1e-5  # smallest normal fp16 value

processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)

# ---------- regression set: the frozen W2-0 sets ----------
EVAL_DIR = Path("/kaggle/input/datasets/alphinjain/w2-eval/eval")   # the uploaded eval/ folder
SET_NAME = "regression_image"       # run once with this, then once with "regression_video"
# full block outputs are kept on CPU for the drift table: ~1.3 GB per page, ~4 GB per clip, per precision
N_DRIFT = 5 if SET_NAME == "regression_image" else 2
# everything this run produces is saved here, so a result never needs a rerun to be re-read or re-scored
OUT_DIR = Path("/teamspace/studios/this_studio/runs/w2-2") / SET_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

eval_set = json.loads((EVAL_DIR / "sets" / f"{SET_NAME}.json").read_text())
v = eval_set["vision"]
regression_set = []   # (vision entry, prompt); the entry says what to load and with which limits
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


# same input building as experiments/w02-0-baseline/run.py: qwen-vl-utils loads + resizes, the processor does not resize again
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


# ---------- hooks ----------
def is_block(m):
    return type(m).__name__.endswith(("VisionBlock", "DecoderLayer"))


def first_tensor(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out if torch.is_tensor(out) and out.is_floating_point() else None


def attach_hooks(model, stats, current):
    handles = []
    for name, m in model.named_modules():
        leaf = len(list(m.children())) == 0
        block = is_block(m)
        if not (leaf or block):
            continue

        def hook(module, args, out, name=name, block=block):
            t = first_tensor(out)
            if t is None:
                return
            s = stats.setdefault(name, {"absmax": 0.0, "nan": 0, "inf": 0, "tiny": 0, "n": 0})
            # measure in chunks of 16M values: the logits alone are GBs, and full-size temporaries ran out of GPU memory
            for c in t.detach().flatten().split(16_000_000):
                c = c.float()                          # measure in fp32 so the measuring cannot overflow
                s["nan"] += torch.isnan(c).sum().item()
                s["inf"] += torch.isinf(c).sum().item()
                a = c.abs().masked_fill_(~torch.isfinite(c), 0)   # non-finite values count as 0 for absmax/tiny
                s["absmax"] = max(s["absmax"], a.max().item())
                s["tiny"] += ((a > 0) & (a < FP16_TINY)).sum().item()
                s["n"] += c.numel()
            if block:
                current[name] = t.detach().cpu().float()   # copy to CPU first, so no extra GPU copy

        handles.append(m.register_forward_hook(hook))
    return handles


# ---------- one full pass over the regression set ----------
def run_set(dtype, device):
    if device == "auto":   # fp32 does not fit one T4: split it across both GPUs
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=model_revision, torch_dtype=dtype, device_map="auto",
            # 2 GPUs: leave GPU 0 room for the vision encoder's activations. 1 GPU: put everything on it.
            max_memory={0: "7GiB", 1: "13GiB"} if torch.cuda.device_count() > 1 else None).eval()   # leave GPU 0 room for the vision encoder's activations
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=model_revision, torch_dtype=dtype).to(device).eval()
    device = model.device   # where the inputs go (the first GPU when split)
    stats, current = {}, {}
    handles = attach_hooks(model, stats, current)

    last_logits, top1, blocks = [], [], []
    for i, (entry, q) in enumerate(regression_set):
        inputs = build_inputs(entry, q, device, dtype)
        current.clear()
        with torch.no_grad():
            out = model(**inputs)
        last_logits.append(out.logits[0, -1].float().cpu())
        top1.append(out.logits[0].argmax(-1).cpu())
        # full block outputs are large, so keep them only for the first N_DRIFT items
        blocks.append(dict(current) if i < N_DRIFT else {})
        del inputs, out
        torch.cuda.empty_cache()   # free this item's activations before the next one

    for h in handles:
        h.remove()

    gens = []
    for entry, q in regression_set:
        inputs = build_inputs(entry, q, device, dtype)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=eval_set["generation"]["max_new_tokens"], do_sample=False)
        gens.append(g[0, inputs["input_ids"].shape[1]:].cpu())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"stats": stats, "last_logits": last_logits, "top1": top1, "blocks": blocks, "gens": gens}


# # ---------- run both ----------
# # fp32 split across Kaggle's 2x T4 (it does not fit on one 15 GB T4), fp16 on one GPU
r32 = run_set(torch.float32, "auto")
# torch.save(r32, "r32.pt")
r16 = run_set(torch.float16, "cuda")
# torch.save(r16, "r16.pt")


# ---------- table 1: activation ranges and NaN/Inf per module ----------
rows = []
for name, s16 in r16["stats"].items():
    s32 = r32["stats"].get(name)
    if s32 is None:
        continue
    rows.append({
        "module": name,
        "kind": re.sub(r"\.\d+(?=\.|$)", ".*", name),
        "absmax_fp32": s32["absmax"],
        "absmax_fp16": s16["absmax"],
        "headroom": FP16_MAX / max(s16["absmax"], 1e-12),
        "nan_fp16": s16["nan"],
        "inf_fp16": s16["inf"],
        "tiny_frac_fp16": s16["tiny"] / max(s16["n"], 1),
    })
df = pd.DataFrame(rows)

df.to_csv(OUT_DIR / "stability_per_module.csv", index=False)   # all modules, in the order they run

# Kept in run order (vision encoder -> LLM -> lm_head), not sorted by count: the FIRST row is where NaN/Inf
# starts, the rest are layers that only received it. Sorting by count would put lm_head, the last victim, on top.
bad = df[(df.nan_fp16 > 0) | (df.inf_fp16 > 0)]
print(f"\nModules with NaN or Inf in fp16: {len(bad)} of {len(df)} (first 5 in run order; full table in the CSV)")
print(bad[["module", "nan_fp16", "inf_fp16"]].head(5).to_string(index=False))


# ---------- table 2: drift per block (vision blocks and LLM layers) ----------
drift = []
for i in range(len(regression_set)):
    for name, a in r32["blocks"][i].items():
        b = r16["blocks"][i][name]
        d = (a - b).abs()
        drift.append({
            "sample": i,
            "block": name,
            "max_diff": d.max().item(),
            "mean_diff": d.mean().item(),
            "rel_max": (d.max() / a.abs().max().clamp_min(1e-12)).item(),
        })
drift = pd.DataFrame(drift)
drift.to_csv(OUT_DIR / "block_drift.csv", index=False)


# ---------- end-to-end agreement: print only the samples whose generated answer changed ----------
print(f"============\nChanged answers ({SET_NAME}):")
n_same = 0
for i in range(len(regression_set)):
    if torch.equal(r32["gens"][i], r16["gens"][i]):
        n_same += 1
        continue
    has_nan = torch.isnan(r16["last_logits"][i]).any().item()
    a32 = processor.decode(r32["gens"][i], skip_special_tokens=True)
    a16 = processor.decode(r16["gens"][i], skip_special_tokens=True)
    print(f"  {eval_set['items'][i]['id']}: fp32 {a32!r} -> fp16 {a16[:20]!r}  ({'NaN' if has_nan else 'rounding'})")
print(f"identical: {n_same}/{len(regression_set)}")


# ---------- accuracy vs ground truth + agreement with fp32, with bootstrap CIs (eval/score.py) ----------
# the answers are saved too, so score.py can re-score them later without running the model
for tag, r in [("fp32", r32), ("fp16", r16)]:
    with open(OUT_DIR / f"{tag}.jsonl", "w") as f:
        for it, g in zip(eval_set["items"], r["gens"]):
            f.write(json.dumps({"id": it["id"], "output": processor.decode(g, skip_special_tokens=True).strip()}) + "\n")
src = (EVAL_DIR / "score.py").read_text()
src = re.sub(r"^EVAL_DIR = .*?^BASELINE = [^\n]*\n",
             f'EVAL_DIR = Path("{EVAL_DIR}")\nSET_NAME = "{SET_NAME}"\n'
             f'PREDICTIONS = "{OUT_DIR / "fp16.jsonl"}"\nBASELINE = "{OUT_DIR / "fp32.jsonl"}"\n',
             src, flags=re.M | re.S)
exec(src, {})


# short summary; the full tables are in the CSVs
w = df.nsmallest(1, "headroom").iloc[0]
d = drift.nlargest(1, "rel_max").iloc[0]
print("============")
print(f"closest to overflow: {w.module}  (fp32 max {w.absmax_fp32:,.0f} vs fp16 max 65,504, inf {w.inf_fp16})")
print(f"largest drift:       {d.block}  (rel_max {d.rel_max:.1%}, sample {d['sample']})")
print(f"full tables:         {OUT_DIR}/stability_per_module.csv, block_drift.csv")

"""
ANSWER:
Almost, but not safely. fp16 gave the same answer as fp32 on 198 of 200 pages. On one page, it broke completely.


RESULTS: fp16 vs fp32 (1x L4, decord)

============ IMAGE (200 DocVQA pages) ============

Conclusion: UNSTABLE. fp16 breaks on 1 page in 200 because one layer overflows.

What happened (on page docvqa_50818):
1. visual.blocks.31.mlp.down_proj (last vision block) outputs 66,995 in fp32.
2. fp16 can store at most 65,504, so 3 values become inf.
3. The next layer (merger.ln_q) normalizes them: inf * 0 = NaN.
4. 3 of the page's ~1,280 image tokens are now NaN.
5. The LLM's attention mixes those tokens into every word, so everything after is NaN.
6. NaN scores -> the model outputs token 0 ("!") -> "!!!!!!!!" instead of "28.00".

Answers that changed (2 of 200):
- docvqa_50818: "28.00" -> "!!!!!!!!"        the overflow above (a bug)
- docvqa_46238: "Grady, 1995" -> "6471"      small fp16 rounding tipped a close call (fp32 was right; not a bug)

Scores:
- ANLS: fp16 93.76 vs fp32 94.76. Difference -1.00, CI [-2.50, 0.00]: not statistically significant.
- Agreement: 198/200 (99%) identical answers.
- The average hides the bug: one garbage answer barely moves a 200-page average.

Rest of the model:
- No other layer overflows. Next closest: blocks.28-30 at ~14,000 (4.7x below the fp16 limit).
- The language model is fine in fp16. All its NaNs came from the vision encoder.

Fix to test:
- T4: run blocks.31 + merger in fp32, the rest in fp16. The merger must be fp32 too, or block 31's output
  (~67,000) overflows again when converted back to fp16.
- L4: use bf16. It has the same range as fp32, so 66,995 cannot overflow.

Full tables: runs/w2-2/regression_image/stability_per_module.csv, block_drift.csv

============ VIDEO: regression_video (36 MVBench clips) ============

Conclusion: STABLE on these 36 clips, but with little margin.

- No NaN or Inf anywhere (0 of 694 modules).
- Answers: 36/36 identical to fp32. Accuracy 61.11% in both.
- Closest to overflow: the same layer, visual.blocks.31.mlp.down_proj, at 56,875 in fp32.
  That is only 1.15x below the fp16 limit (65,504): about 13% more and it would overflow like the image case.
- Next closest: blocks.28-30 at ~13,000 (5x headroom).
- Largest drift: LLM layers.35 (18% rel_max) and visual.blocks.31 (11%); no answer changed.

Image vs video, same layer:
- image: blocks.31 reached 66,995 -> overflow on 1 of 200 pages
- video: blocks.31 reached 56,875 -> no overflow on 36 clips, but close
- So video is not safe either: a different clip could cross the limit. The same fix applies to both.

Full tables: runs/w2-2/regression_video/stability_per_module.csv, block_drift.csv


============ MIXED POLICY FIX (mixed_policy.py: fp16, but blocks.31 + merger in fp32) ============

Image (200 pages): PASS.
- NaN/Inf: 0 (fp16 had 404 modules with NaN).
- blocks.31.mlp.down_proj peak 66,864, now held in fp32.
- docvqa_50818 answers "28.00" again.
- Changed vs fp32: 1 of 200, only the rounding flip (docvqa_46238), as expected.
- ANLS 94.26 vs fp32 94.76 (fp16 was 93.76). Difference -0.50 [-1.50, 0.00], not significant. Agreement 99.5%.
- Time: 2.96 s per page.

Video (36 clips): PASS.
- NaN/Inf: 0. blocks.31 peak 57,008, held in fp32.
- Changed vs fp32: 0 of 36. Accuracy 61.11% in both. Agreement 100%.
- Time: 2.66 s per clip.


============ OVERALL W2-2 CONCLUSION ============

- fp16 is unstable in ONE layer: visual.blocks.31.mlp.down_proj (last vision block).
- Image: it overflows on real pages (1/200 answers destroyed). Video: it came within 13% of overflowing.
- Everything else, including the whole language model, is stable in fp16.
- Fix: the mixed policy (blocks.31 + merger in fp32, rest fp16) PASSES on both sets: 0 NaN/Inf,
  docvqa_50818 answers "28.00" again, agreement 99.5% (image) and 100% (video).
- W2-2 done.
"""
