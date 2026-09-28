import json
import os
import random
from datetime import datetime, timezone
from importlib import metadata

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# Hugging Face commit resolved on 2026-09-28. Keep this fixed so future W1-4
# runs use the same model files and processor configuration.
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
RUN_DIR = os.path.join("runs", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_w4"))
os.makedirs(RUN_DIR, exist_ok=True)


# Optional CUDA cleanup commands are kept here for interactive experiments.
# They can be uncommented between runs when a previous model/output still
# holds GPU memory.
# import torch, gc

# # del model          # or del outputs, del inputs, whatever's holding GPU memory
# # del outputs
# # del model, outputs, inputs
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()

# Show the CUDA allocator state before loading the model.
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

def make_test_image(width, height, seed=0):
    """Create a deterministic image containing random outlined rectangles."""
    random.seed(seed)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # random shapes so it's not a blank image (avoids degenerate edge cases)
    for _ in range(20):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], outline=color, width=3)
    return img

# Generate a few image sizes for quick input-size experiments.
img_small  = make_test_image(448, 448)
img_medium = make_test_image(1036, 1036)
img_large  = make_test_image(2000, 1500)


# 1. Load the vision-language model and the matching processor.
# bfloat16 reduces GPU memory use, while device_map places the model on CUDA.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16, device_map="cuda"
)
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model.eval()


# 2. Register forward hooks before inference so intermediate tensor shapes and
# numerical issues can be inspected during the forward pass.
shapes = {}

def make_hook(name):
    """Build a hook that records output shape and reports NaN/Inf values."""
    def hook(module, inp, out):
        # Some modules return tuples; the first item is the tensor we inspect.
        t = out[0] if isinstance(out, tuple) else out
        shapes[name] = tuple(t.shape)
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        if has_nan or has_inf:
            print(f"⚠️ {name}: nan={has_nan}, inf={has_inf}")
    return hook
# Capture selected vision and language-model stages. Hooks on every vision
# block make it possible to identify where a shape or numerical problem starts.
model.model.visual.patch_embed.register_forward_hook(make_hook("patch_embed"))
for i, blk in enumerate(model.model.visual.blocks):
    blk.register_forward_hook(make_hook(f"vit_block_{i}"))  # tag windowed vs full using indices {7,15,23,31}

model.model.visual.merger.register_forward_hook(make_hook("merger"))
model.model.language_model.embed_tokens.register_forward_hook(make_hook("llm_embed"))
model.model.language_model.layers[0].register_forward_hook(make_hook("llm_block_0"))
model.lm_head.register_forward_hook(make_hook("lm_head"))


from qwen_vl_utils import process_vision_info

# Build the multimodal chat message. The image is a PIL image and the text is
# the instruction that will be sent to the model.
img_medium = make_test_image(1024, 1024)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_medium},  # your PIL image from earlier
            {"type": "text", "text": "Describe this image."}
        ]
    }
]
# Convert the message into the model's expected prompt and image/video inputs.
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# Tokenize the prompt and preprocess visual inputs, then move tensors to CUDA.
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")


# Inspect the visual tensor layout and temporal-height-width grid metadata.
print("pixel_values:", inputs["pixel_values"].shape)
print("image_grid_thw:", inputs["image_grid_thw"])

# Keep architecture claims in the accompanying notes tied to the loaded
# checkpoint rather than to a paper/configuration remembered separately.
vision_config = model.config.vision_config
text_config = model.config.text_config
print("vision depth:", vision_config.depth)
print("vision full-attention blocks:", vision_config.fullatt_block_indexes)
print("vision spatial merge size:", vision_config.spatial_merge_size)
print("vision MLP intermediate size:", vision_config.intermediate_size)
print("LLM hidden size / MLP intermediate size / vocab size:",
      text_config.hidden_size, text_config.intermediate_size, text_config.vocab_size)

# No gradients are needed for this diagnostic forward pass.
with torch.no_grad():
    outputs = model(**inputs)

# Print the shapes collected by the hooks in registration/execution order.
for name, shape in shapes.items():
    print(name, shape)

# `input_ids` already contains one image-placeholder token for each merged
# visual token.  Report the split as well as the total, so the documented
# 25-text-token figure is a measured value rather than an inferred one.
image_token_id = model.config.image_token_id
image_token_count = int((inputs["input_ids"] == image_token_id).sum())
total_token_count = inputs["input_ids"].shape[1]
print("text+image token count:", inputs["input_ids"].shape)
print("image placeholder tokens:", image_token_count)
print("non-image text tokens:", total_token_count - image_token_count)


# Print the complete model output for additional debugging when needed.
print(outputs)

# Preserve the evidence needed to reproduce the documented W1-4 shapes.
run_info = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model": MODEL_ID,
    "requested_revision": MODEL_REVISION,
    "loaded_revision": getattr(model.config, "_commit_hash", None),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "qwen_vl_utils": metadata.version("qwen-vl-utils"),
    "input": {
        "source_image_hw": [img_medium.height, img_medium.width],
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "input_ids_shape": list(inputs["input_ids"].shape),
        "image_placeholder_tokens": image_token_count,
        "non_image_text_tokens": total_token_count - image_token_count,
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
}
run_info_path = os.path.join(RUN_DIR, "run.json")
with open(run_info_path, "w") as f:
    json.dump(run_info, f, indent=2)
print("saved run metadata:", run_info_path)
