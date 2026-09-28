Qwen 2.5 VL 3B

Checkpoint revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`.
The reported W4 rerun ran on an NVIDIA L4 in Lightning AI Compute.

> W4 rerun (`1024×1024` generated image): `pixel_values` is `(5476, 1176)`, `image_grid_thw` is `(1, 74, 74)`, and `input_ids` is `(1, 1394)`. The 2×2 merger maps 5,476 vision patches to 1,369 visual-token positions, leaving 25 non-image text positions (`1394 − 1369`).

> “Your shape” contains only tensors printed by the W4 hook trace. “Not captured” means the row describes an interface or operation but the script did not record that tensor; it is not presented as a measurement.

For latency, memory, and end-to-end performance results, see the [W1 overall findings](../../../docs/findings/w1-vlm-overall-summary.md).

## Vision side

| # | Component | Likely module/class name | What it does | Predicted shape | Your shape |
|---|---|---|---|---|---|
| 1 | Raw image input | — (before processor) | original pixel data | (H, W, 3) | (1024, 1024, 3) |
| 2 | Preprocessing / resize | `Qwen2_5_VLImageProcessor` | resizes and normalizes; each flattened vision input row contains two RGB 14×14 patches | `pixel_values`: (patches, 1176); `image_grid_thw`: (T, H, W) | `pixel_values`: (5476, 1176); `image_grid_thw`: (1, 74, 74) |
| 3 | Patch embedding | `Qwen2_5_VisionPatchEmbed` | maps each packed pair of temporal 14×14 RGB patches (1,176 values) to one 1,280-dim vision vector | (5476, 1280) | (5476, 1280) |
| 4 | Rotary position calc | often a function, not a hookable module (e.g. `rot_pos_emb`) | computes vision positional rotations | n/a (not captured by this hook set) | not captured |
| 5a | ViT block — windowed (pick one, e.g. block 0) | `model.visual.blocks[0]` | norm → windowed self-attn → residual → norm → SwiGLU MLP → residual | in/out: (5476, 1280) |(5476, 1280) |
| 5b | ViT block — full attention (pick block 7) | `model.visual.blocks[7]` | same structure, but full self-attn over all patches | in/out: (5476, 1280) |(5476, 1280) |
| 6 | Merger / projector | `model.visual.merger` (a `Qwen2_5_VLPatchMerger`-type class) | groups 2×2 patches, concatenates, 2-layer MLP projects to LLM width | (5476, 1280) → (1369, hidden_size) | (1369,2048) |

## Bridge (vision → language)

| # | Component | Likely module | What it does | Predicted shape | Your shape |
|---|---|---|---|---|---|
| 7 | Initial LLM embedding lookup | `model.model.language_model.embed_tokens` | embeds all 1,394 input IDs, including 1,369 image-placeholder IDs | (batch, sequence, hidden_size) | (1, 1394, 2048) |
| 8 | Visual embedding scatter | inside the top-level `forward()` | replaces the 1,369 image-placeholder embeddings with merger output at the same sequence positions | (1, 1394, 2048) | shape-preserving; not separately hooked |
| 9 | MRoPE position ids | a function like `get_rope_index` | computes the (t, h, w) position IDs for every token, text and visual | n/a — log the position_ids tensor shape, e.g. (3, sequence) | not captured |

## Language side

| # | Component | Likely module | What it does | Predicted shape | Your shape |
|---|---|---|---|---|---|
| 10 | First LLM decoder layer | `model.model.language_model.layers[0]` | norm → GQA self-attn (causal) → residual → norm → SwiGLU MLP → residual | (batch, sequence, hidden_size) | (1, 1394, 2048) |
| 11 | Final norm | `model.model.language_model.norm` | last RMSNorm before the output head | (batch, sequence, hidden_size) | not captured |
| 12 | LM head | `model.lm_head` | projects to vocabulary logits | (batch, sequence, vocab) | (1, 1394, 151936) |

The W4 rerun confirms 32 vision blocks, numbered 0–31. Its checkpoint configuration reports full-attention blocks `[7, 15, 23, 31]`; the other 28 are windowed. It also reports spatial merge size 2, vision MLP intermediate size 3,420, LLM hidden size 2,048, LLM MLP intermediate size 11,008, and vocabulary size 151,936.
