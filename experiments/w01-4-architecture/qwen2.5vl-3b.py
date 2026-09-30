# =============================================================================
# W1-4: tensor trace of ONE VIDEO request through Qwen2.5-VL-3B-Instruct.
#
# Forward hooks record the shape at every stage the plan names: processor
# output (preprocessing), patch embed, each vision block, the merger, the LLM
# embedding, the first LLM block, and the lm_head.  The printed shapes are the
# evidence behind docs/architecture.md.
#
# Matches the W1-3 trace configuration so the shapes line up with the measured
# timings: same clip, same FPS, same revision, bf16.
#
# RUN (from the repository root):
#     python experiments/w01-4-architecture/qwen2.5vl-3b.py
# Writes runs/<UTC time>_w4/run.json with every shape and the checkpoint config.
# =============================================================================
import json
import os
from datetime import datetime, timezone
from importlib import metadata

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# Hugging Face commit resolved on 2026-09-28. Keep this fixed so future W1-4
# runs use the same model files and processor configuration.
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"

# W1-3 configuration, so these shapes describe the request that was profiled.
CLIP_PATH = "experiments/w01-2-baseline-video/clip_08s.mp4"
VIDEO_FPS = 2.0
PATCH = 14                 # the vision tower cuts each frame into 14x14 px patches
FRAMES_PER_GROUP = 2       # consecutive frames are fused in pairs
PROMPT = "Describe what happens in this video."
DTYPE = torch.bfloat16     # L4 supports bf16 natively; all W1 experiments use it
SEED = 1234

RUN_DIR = os.path.join("runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_w4"))
os.makedirs(RUN_DIR, exist_ok=True)

if not os.path.exists(CLIP_PATH):
    raise FileNotFoundError(f"{CLIP_PATH} not found; run from the repository root.")

torch.manual_seed(SEED)

# Allocator state before loading, so the reported peak is attributable.
print(f"{torch.cuda.memory_allocated() / 1e9:.1f} GB allocated before load")
print(f"{torch.cuda.memory_reserved() / 1e9:.1f} GB reserved before load")


# 1. Load the vision-language model and the matching processor.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, dtype=DTYPE, device_map="cuda"
)
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model.eval()


# 2. Register forward hooks before inference so intermediate tensor shapes and
# numerical issues can be inspected during the forward pass.
shapes = {}

def make_hook(name):
    """Record the output shape and report any NaN/Inf values."""
    def hook(module, inp, out):
        # Some modules return tuples; the first item is the tensor we inspect.
        t = out[0] if isinstance(out, tuple) else out
        shapes[name] = tuple(t.shape)
        if torch.isnan(t).any().item() or torch.isinf(t).any().item():
            print(f"WARNING {name}: nan or inf in output")
    return hook

# Capture selected vision and language-model stages. Hooks on every vision
# block make it possible to identify where a shape or numerical problem starts.
model.model.visual.patch_embed.register_forward_hook(make_hook("patch_embed"))
for i, blk in enumerate(model.model.visual.blocks):
    blk.register_forward_hook(make_hook(f"vit_block_{i}"))  # full attention at {7,15,23,31}

model.model.visual.merger.register_forward_hook(make_hook("merger"))
model.model.language_model.embed_tokens.register_forward_hook(make_hook("llm_embed"))
model.model.language_model.layers[0].register_forward_hook(make_hook("llm_block_0"))
model.lm_head.register_forward_hook(make_hook("lm_head"))


from qwen_vl_utils import process_vision_info

# 3. Build the video request, exactly as W1-3 does.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": CLIP_PATH, "fps": VIDEO_FPS},
            {"type": "text", "text": PROMPT},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# qwen-vl-utils samples and resizes the frames, so the processor gets
# do_resize=False. return_video_kwargs carries the sampling metadata forward.
_, videos, video_kwargs = process_vision_info(
    messages, image_patch_size=PATCH, return_video_kwargs=True, return_video_metadata=True
)
video_frames, video_meta = zip(*videos)

inputs = processor(
    text=[text], videos=list(video_frames), video_metadata=list(video_meta),
    padding=True, return_tensors="pt", do_resize=False,
    cap_pixels_per_frame=True, **video_kwargs
).to("cuda")


# 4. PREPROCESSING OUTPUT — the first stage the plan asks for.  This is the
# processor's output, before any model module runs, so it cannot come from a
# module forward hook; it is read directly off the batch.
shapes["preprocess_pixel_values"] = tuple(inputs["pixel_values_videos"].shape)
t_grid, h_grid, w_grid = inputs["video_grid_thw"][0].tolist()

print("\n--- preprocessing output (stage 0) ---")
print("pixel_values_videos:", tuple(inputs["pixel_values_videos"].shape))
print("video_grid_thw:", inputs["video_grid_thw"].tolist())
print(f"  {t_grid} frame groups x {h_grid} x {w_grid} patches of {PATCH}px")
print(f"  frames sampled: {t_grid * FRAMES_PER_GROUP} at {VIDEO_FPS:g} fps")
print(f"  frame size: {h_grid * PATCH}x{w_grid * PATCH} px")

merge = model.config.vision_config.spatial_merge_size
tokens_per_group = h_grid * w_grid // merge**2
print(f"  tokens per frame group: {tokens_per_group}"
      f"  (per actual frame: {tokens_per_group / FRAMES_PER_GROUP})")
print(f"  expected video tokens: {t_grid * tokens_per_group}")

# Keep architecture claims tied to the loaded checkpoint rather than to a
# paper/configuration remembered separately.
vision_config = model.config.vision_config
text_config = model.config.text_config
print("\n--- checkpoint configuration ---")
print("vision depth:", vision_config.depth)
print("vision full-attention blocks:", vision_config.fullatt_block_indexes)
print("vision spatial merge size:", vision_config.spatial_merge_size)
print("vision MLP intermediate size:", vision_config.intermediate_size)
print("LLM hidden size / MLP intermediate size / vocab size:",
      text_config.hidden_size, text_config.intermediate_size, text_config.vocab_size)

# 5. One diagnostic forward pass; no gradients needed.
with torch.no_grad():
    outputs = model(**inputs)

print("\n--- hook shapes, in execution order ---")
print(f"{'preprocess_pixel_values':<26}", shapes["preprocess_pixel_values"])
for name, shape in shapes.items():
    if name != "preprocess_pixel_values":
        print(f"{name:<26}", shape)

# `input_ids` already contains one video-placeholder token for each merged
# visual token.  Report the split as well as the total, so the text-token
# figure is a measured value rather than an inferred one.
video_token_id = model.config.video_token_id
video_token_count = int((inputs["input_ids"] == video_token_id).sum())
total_token_count = inputs["input_ids"].shape[1]
merger_rows = shapes["merger"][0]

print("\n--- token accounting ---")
print("input_ids:", tuple(inputs["input_ids"].shape))
print("video placeholder tokens:", video_token_count)
print("non-video text tokens:", total_token_count - video_token_count)
print("merger output rows:", merger_rows)
# The merger emits exactly one vector per video token; if these disagree, the
# shape story in docs/architecture.md is wrong.
print("merger rows == video tokens:", merger_rows == video_token_count)
print("video tokens == groups x tokens/group:",
      video_token_count == t_grid * tokens_per_group)

print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (includes weights)")

# 6. Preserve the evidence needed to reproduce the documented W1-4 shapes.
run_info = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_ID,
    "requested_revision": MODEL_REVISION,
    "loaded_revision": getattr(model.config, "_commit_hash", None),
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "dtype": str(DTYPE),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "qwen_vl_utils": metadata.version("qwen-vl-utils"),
    "input": {
        "clip": CLIP_PATH,
        "requested_fps": VIDEO_FPS,
        "seed": SEED,
        "frame_groups": t_grid,
        "frames_sampled": t_grid * FRAMES_PER_GROUP,
        "frame_hw": [h_grid * PATCH, w_grid * PATCH],
        "video_grid_thw": inputs["video_grid_thw"].tolist(),
        "pixel_values_videos_shape": list(inputs["pixel_values_videos"].shape),
        "tokens_per_group": tokens_per_group,
        "tokens_per_frame": tokens_per_group / FRAMES_PER_GROUP,
        "input_ids_shape": list(inputs["input_ids"].shape),
        "video_placeholder_tokens": video_token_count,
        "non_video_text_tokens": total_token_count - video_token_count,
    },
    "checks": {
        "merger_rows_equal_video_tokens": merger_rows == video_token_count,
        "video_tokens_match_grid": video_token_count == t_grid * tokens_per_group,
    },
    "hook_shapes": {name: list(shape) for name, shape in shapes.items()},
    "checkpoint_config": {
        "vision_depth": vision_config.depth,
        "full_attention_blocks": list(vision_config.fullatt_block_indexes),
        "spatial_merge_size": vision_config.spatial_merge_size,
        "vision_mlp_intermediate_size": vision_config.intermediate_size,
        "llm_hidden_size": text_config.hidden_size,
        "llm_mlp_intermediate_size": text_config.intermediate_size,
        "vocab_size": text_config.vocab_size,
    },
    "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
}
run_info_path = os.path.join(RUN_DIR, "run.json")
with open(run_info_path, "w") as f:
    json.dump(run_info, f, indent=2)
print("saved run metadata:", run_info_path)
