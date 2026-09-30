import gc
import re
import requests
import torch
import pandas as pd
from PIL import Image
import open_clip
import numpy as np
from PIL import Image, ImageDraw


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
base = Image.open(requests.get(url, stream=True).raw).convert("RGB")

FP16_MAX = 65504.0
FP16_TINY = 6.1e-5

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)


def checkerboard(size=448, cell=14):
    idx = np.arange(size) // cell
    board = ((idx[:, None] + idx[None, :]) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([board] * 3, axis=-1))


def stripes(size=448):
    col = (np.arange(size) % 2 * 255).astype(np.uint8)
    board = np.tile(col, (size, 1))
    return Image.fromarray(np.stack([board] * 3, axis=-1))


def noise(size=448, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def solid(color, size=448):
    return Image.new("RGB", (size, size), color)


def text_page(w=896, h=672):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i in range(30):
        d.text((10, 10 + i * 20), f"Line {i}: the quick brown fox jumps over the lazy dog 0123456789", fill="black")
    return img


# `base` is the COCO image already loaded in the main script
stress_set = [
    (checkerboard(), "Describe this image."),
    (stripes(), "Describe this image."),
    (noise(), "Describe this image."),
    (solid((0, 0, 0)), "Describe this image."),
    (solid((255, 255, 255)), "Describe this image."),
    (text_page(), "Read the text in this image."),
    (base.resize((1120, 840)), "Describe this image in detail."),
]



import open_clip


def is_block(m):
    return type(m).__name__.endswith(("Block", "ResidualAttentionBlock"))


def first_tensor(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out if torch.is_tensor(out) and out.is_floating_point() else None


def attach_hooks(model, stats, current):
    """
    This gives you the natural hook points:
        • patch_embed: image pixels become patch representations
        • each Block: one complete Transformer stage
        • norm: final visual representation before projection
        • projection/head: representation becomes the CLIP embedding
        The general rule is: Hook semantic stages, not every container.
            For a Transformer, blocks are usually the best diagnostic boundary.
            For a CNN, use major stages or residual blocks.
            For an encoder-decoder, hook the encoder, decoder blocks, and final projection.
            For an MLP, hook meaningful layer groups.

    The internal hooks answer:
        Where did fp16 begin to differ from fp32?
    The final output answers:
        Did that difference change the model’s decision?
    """
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
            f = t.detach().float()  # measure in fp32 so the measuring cannot overflow
            a = f[torch.isfinite(f)].abs()
            s = stats.setdefault(name, {"absmax": 0.0, "nan": 0, "inf": 0, "tiny": 0, "n": 0})
            s["nan"] += torch.isnan(f).sum().item()
            s["inf"] += torch.isinf(f).sum().item()
            if a.numel():
                s["absmax"] = max(s["absmax"], a.max().item())
            s["tiny"] += ((a > 0) & (a < FP16_TINY)).sum().item()
            s["n"] += f.numel()
            if block:
                current[name] = f.cpu()

        handles.append(m.register_forward_hook(hook))
    return handles

# ---------- one full pass over the regression set ----------
# ---------- regression set (replace with your W1 set if you have one) ----------
regression_set = [
    (base, "Describe this image."),
    (base.resize((448, 448)), "How many animals are there?"),
    (base.resize((336, 504)), "What colors do you see?"),
    (base.resize((896, 672)), "Describe this image in detail."),
]

tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')



def run_set(dtype, device):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    model = model.to(device=device, dtype=dtype).eval()

    stats, current, image_embeddings, text_embeddings, similarities = {}, {}, [], [], []
    handles = attach_hooks(model, stats, current)
    candidates = ["a shirt", "a shoe", "a dress", "a jacket", "a pair of pants"]
    text_tokens = tokenizer(candidates).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens, normalize=True).float().cpu()

    last_logits, top1, blocks = [], [], []
    for img, question in stress_set:
        current.clear()

        image_tensor = preprocess_val(img).unsqueeze(0)
        image_tensor = image_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor, normalize=True)

        image_features_cpu = image_features.float().cpu()
        scores = image_features_cpu @ text_features.T
        predicted_index = scores.argmax(dim=-1).item()


        image_embeddings.append(image_features_cpu)
        text_embeddings.append(text_features)
        similarities.append({
            "scores": scores,
            "top1_index": predicted_index,
            "top1_text": candidates[predicted_index],
            "question": question,
        })

        blocks.append(dict(current))

    for h in handles:
        h.remove()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"stats": stats, "blocks": blocks, "image_embeddings": image_embeddings, "text_embeddings": text_embeddings, "similarities": similarities}


r32 = run_set(torch.float32, device)
r16 = run_set(torch.float16, device)

for i in range(len(stress_set)):
    fp32 = r32["similarities"][i]
    fp16 = r16["similarities"][i]

    print(
        f"sample {i}: "
        f"fp32={fp32['top1_text']!r}, "
        f"fp16={fp16['top1_text']!r}, "
        f"same={fp32['top1_text'] == fp16['top1_text']}"
    )

# ---------- activation ranges ----------

rows = [
    {
        "module": name,
        "absmax_fp32": s32["absmax"],
        "absmax_fp16": s16["absmax"],
        "headroom": FP16_MAX / max(s16["absmax"], 1e-12),
        "nan_fp16": s16["nan"],
        "inf_fp16": s16["inf"],
        "tiny_frac_fp16": s16["tiny"] / max(s16["n"], 1),
    }
    for name, s16 in r16["stats"].items()
    if (s32 := r32["stats"].get(name)) is not None
]

df = pd.DataFrame(rows)

cols = ["module", "absmax_fp32", "absmax_fp16", "headroom", "nan_fp16","inf_fp16"]


# ---------- block drift ----------

# ---------- activation ranges ----------

rows = [
    {
        "module": name,
        "absmax_fp32": s32["absmax"],
        "absmax_fp16": s16["absmax"],
        "headroom": FP16_MAX / max(s16["absmax"], 1e-12),
        "nan_fp16": s16["nan"],
        "inf_fp16": s16["inf"],
        "tiny_frac_fp16": s16["tiny"] / max(s16["n"], 1),
    }
    for name, s16 in r16["stats"].items()
    if (s32 := r32["stats"].get(name)) is not None
]

df = pd.DataFrame(rows)

cols = ["module", "absmax_fp32", "absmax_fp16", "headroom", "nan_fp16","inf_fp16"]

print("\nTop 5 modules with most fp16 headroom:")
print(df.nlargest(5, "headroom")[cols].to_string(index=False))


# ---------- block drift ----------

drift = pd.DataFrame([
    {
        "sample": sample,
        "block": name,
        "max_diff": diff.max().item(),
        "p99_diff": torch.quantile(diff, 0.99).item(),
        "p95_diff": torch.quantile(diff, 0.95).item(),
        "mean_diff": diff.mean().item(),
        "rel_max": (diff.max() /fp32.abs().max().clamp_min(1e-12) ).item(),
    }
    for sample, (fp32_blocks, fp16_blocks)
    in enumerate(zip(r32["blocks"], r16["blocks"]))
    for name, fp32 in fp32_blocks.items()
    for fp16 in [fp16_blocks[name]]
    for diff in [(fp32 - fp16).abs()]
])

print("============\nTop 5 modules closest to fp16 overflow:")
print(
    df.nsmallest(5, "headroom")[cols]
    .to_string(index=False)
)

print("============\nTop 5 modules with most fp16 headroom:")
print(
    df.nlargest(5, "headroom")[cols]
    .to_string(index=False)
)

print("============\nTop 5 highest block drift:")
print(
    drift.nlargest(5, "rel_max")
    .to_string(index=False)
)

print("============\nEnd-to-end top-1 agreement:")

same_count = 0

for i, (fp32_result, fp16_result) in enumerate(
    zip(r32["similarities"], r16["similarities"])
):
    same = fp32_result["top1_index"] == fp16_result["top1_index"]
    same_count += same

    print(
        f"sample {i}: "
        f"fp32={fp32_result['top1_text']!r}, "
        f"fp16={fp16_result['top1_text']!r}, "
        f"same={same}"
    )

print(
    f"agreement: {same_count}/{len(r32['similarities'])}"
)

"""

Conclusion: the fp16 MarQoS run is numerically stable on this test set in L4.
• FP16 produced no NaN or Inf values.
• The riskiest module had approximately 70× headroom before FP16 overflow.
• The largest relative block drift was 1.39%.
• The worst p99 drift was 0.0104
• The corresponding mean drift was only 0.0019, so most values changed very little.
• Drift was highest in later visual blocks, especially blocks 5–7, on samples 4 and 5.
• FP32 and FP16 produced the same top-1 result for all samples: 7/7 agreement.
• Overall conclusion: FP16 passed the numerical-stability test for this MarQoS test set.
• This conclusion applies only to the seven tested inputs and five candidate text labels.


============
Top 5 modules closest to fp16 overflow:
                       module  absmax_fp32  absmax_fp16  headroom  nan_fp16  inf_fp16
text.transformer.resblocks.10   932.135315        932.0 70.283262         0         0
 text.transformer.resblocks.9   929.900940        930.0 70.434409         0         0
text.transformer.resblocks.11   925.621338        925.5 70.776877         0         0
 text.transformer.resblocks.8   923.742126        924.0 70.891775         0         0
 text.transformer.resblocks.7   920.782654        921.0 71.122693         0         0
============
Top 5 modules with most fp16 headroom:
                              module  absmax_fp32  absmax_fp16      headroom  nan_fp16  inf_fp16
            visual.trunk.attn_pool.q     0.572737     0.572754 114366.745098         0         0
       visual.trunk.attn_pool.q_norm     0.572737     0.572754 114366.745098         0         0
   text.transformer.resblocks.4.ls_1     0.879562     0.879883  74446.277469         0         0
     visual.trunk.blocks.8.attn.proj     1.025576     1.025391  63881.996190         0         0
visual.trunk.blocks.8.attn.proj_drop     1.025576     1.025391  63881.996190         0         0
============
Top 5 highest block drift:
 sample                 block  max_diff  p99_diff  p95_diff  mean_diff  rel_max
      4 visual.trunk.blocks.5  2.104031  0.007720  0.003789   0.001354 0.013915
      4 visual.trunk.blocks.6  2.175133  0.009061  0.004495   0.001562 0.012040
      5 visual.trunk.blocks.6  1.957893  0.006645  0.003027   0.001011 0.012037
      5 visual.trunk.blocks.7  2.041740  0.010153  0.004230   0.001347 0.011616
      4 visual.trunk.blocks.7  2.234543  0.010447  0.005434   0.001874 0.011525
============
End-to-end top-1 agreement:
sample 0: fp32='a shoe', fp16='a shoe', same=True
sample 1: fp32='a shirt', fp16='a shirt', same=True
sample 2: fp32='a shirt', fp16='a shirt', same=True
sample 3: fp32='a shirt', fp16='a shirt', same=True
sample 4: fp32='a shirt', fp16='a shirt', same=True
sample 5: fp32='a shoe', fp16='a shoe', same=True
sample 6: fp32='a pair of pants', fp16='a pair of pants', same=True
agreement: 7/7

"""