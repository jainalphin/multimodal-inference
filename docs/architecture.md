# Qwen2.5-VL video path

Supporting detail for the finding in the [main README](../README.md).

```text
video frames
  → sample and resize
  → 3D patch embedding
  → vision encoder (32 blocks)
  → 2×2 visual-token merger
  → scatter visual tokens into the text sequence
  → language-model prefill and decode
```

## Measured shapes

`clip_08s.mp4` at 2 FPS, bf16 on L4, forward hooks.
`video_grid_thw = [[8, 26, 46]]` — 8 frame groups of 26×46 patches.

| Stage | Shape | Where the numbers come from |
|---|---|---|
| Processor output | `(9568, 1176)` | 9568 = 8 × 26 × 46 patch rows; 1176 = 3 ch × 2 temporal × 14 × 14 |
| Patch embedding | `(9568, 1280)` | 1280 = vision width |
| Each of 32 vision blocks | `(9568, 1280)` | shape-preserving; full attention at 7, 15, 23, 31 |
| 2×2 merger | `(2392, 2048)` | 2392 = 9568 / 2²; merger input 5120 = 1280 × 2²; 2048 = LLM width |
| LLM embedding / block 0 | `(1, 2420, 2048)` | 2420 = 2392 visual + 28 text |
| LM head | `(1, 2420, 151936)` | 151936 = vocab size |

2,392 video tokens + 28 text = 2,420. Merger rows equal the video-token count,
and 8 groups × 299 = 2,392. Peak 8.8 GiB including weights.

**`llm_embed` fires before `patch_embed`.** Not a bug: the LLM embeds the whole
`input_ids` sequence, placeholders included, before the vision tower runs; the
merger output is then scattered into those positions. So the visual and text
paths are not a simple sequential pipeline.

## Token count vs resolution

Visual tokens are `(H/14 × W/14) / 2²` = `(H × W) / 784` — **quadratic in side
length**. Doubling resolution quadruples visual tokens, and prefill grows
faster still since attention is quadratic in sequence length.

| Input | Patch grid | ViT patches | Visual tokens |
|---|---|---:|---:|
| 448×448 | 32×32 | 1,024 | 256 |
| 1036×1036 | 74×74 | 5,476 | 1,369 |
| 2044×2044 | 146×146 | 21,316 | 5,329 |

For video the unit is a **frame group of 2 fused frames**: at 364×644 that is a
26×46 patch grid → 299 merged positions per group, i.e. 149.5 per actual frame.

## Checkpoint

- `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Vision: 32 blocks, width 1,280, full attention at 7/15/23/31, spatial merge 2×2
- Language: width 2,048, MLP 11,008, vocab 151,936

## Reproducing

```bash
python experiments/w01-4-architecture/qwen2.5vl-3b.py
```

Matches the W1-3 configuration and hooks every stage above. Writes
`runs/<UTC time>_w4/run.json` with shapes, token accounting, and consistency
checks. `runs/` is gitignored.
