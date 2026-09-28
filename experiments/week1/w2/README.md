# W1-2: Qwen2.5-VL video baseline

## Latest rerun results

| Clip | FPS | Cap | Frames | Tokens/frame | Video tokens | Tokens/s | Prefill (ms) | Decode (ms/token) | TTFT (ms) | Peak GPU GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clip_08s | 1.0 | default | 8 | 299 | 1,196 | 149.5 | 170.6 | 43.0 | 525.4 | 7.4 |
| clip_08s | 2.0 | default | 16 | 299 | 2,392 | 299.0 | 336.8 | 38.6 | 1,081.6 | 7.6 |
| clip_30s | 1.0 | default | 30 | 299 | 4,485 | 149.4 | 688.7 | 38.2 | 2,155.1 | 8.0 |
| clip_30s | 2.0 | default | 60 | 299 | 8,970 | 298.7 | 1,597.6 | 38.6 | 4,532.0 | 8.9 |

All four default-cap requests fit on the GPU used for this rerun. The captured
output did not include its GPU or package-version metadata.
