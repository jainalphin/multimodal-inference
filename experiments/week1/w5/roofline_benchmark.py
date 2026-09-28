import torch
import torch.nn.functional as F
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration

# ---- 1. timing helper ----
def time_op(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    t = np.array(times_ms)
    return {
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ms": float(np.mean(t)),
        "std_ms": float(np.std(t)),
    }

def achieved_tflops(flops, result):
    return flops / (result["median_ms"] / 1000) / 1e12

# ---- L4 spec: dense (non-sparse) FP16/BF16 Tensor Core peak, official NVIDIA datasheet ----
GPU_SPECS = {
    "L4": {"peak_tflops": 121, "bandwidth_gbs": 300},
}

def ridge_point(peak_tflops, bandwidth_gbs):
    return (peak_tflops * 1e12) / (bandwidth_gbs * 1e9)

def verdict(ai, gpu_name):
    rp = ridge_point(GPU_SPECS[gpu_name]["peak_tflops"], GPU_SPECS[gpu_name]["bandwidth_gbs"])
    return ("compute-bound" if ai > rp else "memory-bound"), rp

# ---- 2. load model (skip if already loaded in your session) ----
print("CUDA device:", torch.cuda.get_device_name())
print("CUDA compute capability:", torch.cuda.get_device_capability())
print("PyTorch:", torch.__version__)
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"
print("Model revision:", model_revision)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, revision=model_revision, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

# ---- 3. define the four ops, using your own N values from the last task ----

# a) encoder linear (QKV projection), N=5476
qkv = model.model.visual.blocks[0].attn.qkv
x_a = torch.randn(5476, 1280, device="cuda", dtype=torch.bfloat16)
fn_a = lambda: qkv(x_a)
flops_a = 2 * 5476 * 1280 * 3840
bytes_a = (1280 * 3840 * 2) + (5476 * 1280 * 2) + (5476 * 3840 * 2)
ai_a = flops_a / bytes_a

# b) encoder attention, N=5476 — naive vs fused
N, heads, hd = 5476, 16, 80
q = torch.randn(1, heads, N, hd, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, heads, N, hd, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, heads, N, hd, device="cuda", dtype=torch.bfloat16)

def fn_b_naive():
    scores = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)

fn_b_fused = lambda: F.scaled_dot_product_attention(q, k, v)
flops_b = 4 * (N ** 2) * 1280
qkv_bytes = 3 * N * 1280 * 2
score_bytes = heads * (N ** 2) * 2 * 2  # naive score tensor: write + read, bf16
out_bytes = N * 1280 * 2
bytes_b = qkv_bytes + score_bytes + out_bytes
ai_b = flops_b / bytes_b

# c) projector (merger), N=1369 — merger.forward expects the raw
# un-grouped ViT output (5476, 1280); it groups 2x2 patches internally
print(model.model.visual.merger)
x_c = torch.randn(5476, 1280, device="cuda", dtype=torch.bfloat16)
fn_c = lambda: model.model.visual.merger(x_c)
flops_c = 2 * 1369 * 5120 * 5120 + 2 * 1369 * 5120 * 2048
bytes_c = (5120*5120*2) + (1369*5120*2) + (1369*5120*2) + (5120*2048*2) + (1369*2048*2)
ai_c = flops_c / bytes_c

# d) LLM prefill linear (gate_proj), N=1394
# NOTE: intermediate_size is 11008 in the checkpoint configuration used here.
gate = model.model.language_model.layers[0].mlp.gate_proj
x_d = torch.randn(1394, 2048, device="cuda", dtype=torch.bfloat16)
fn_d = lambda: gate(x_d)
flops_d = 2 * 1394 * 2048 * 11008
bytes_d = (2048*11008*2) + (1394*2048*2) + (1394*11008*2)
ai_d = flops_d / bytes_d

# ---- 4. run everything ----
with torch.no_grad():
    r_a = time_op(fn_a)
    r_b_naive = time_op(fn_b_naive)
    r_b_fused = time_op(fn_b_fused)
    r_c = time_op(fn_c)
    r_d = time_op(fn_d)

print("Encoder linear (QKV):   ", r_a, "-> TFLOPS:", achieved_tflops(flops_a, r_a))
print("Attention (naive):      ", r_b_naive, "-> TFLOPS:", achieved_tflops(flops_b, r_b_naive))
print("Attention (fused SDPA): ", r_b_fused, "-> TFLOPS:", achieved_tflops(flops_b, r_b_fused))
print("Projector (merger):     ", r_c, "-> TFLOPS:", achieved_tflops(flops_c, r_c))
print("LLM prefill linear:     ", r_d, "-> TFLOPS:", achieved_tflops(flops_d, r_d))

# ---- 5. roofline verdicts: memory-bound vs compute-bound on L4 ----
# ai_b includes a materialized attention-score tensor.  That is valid for fn_b_naive,
# whose score tensor is written and then read by softmax, but it is not the I/O model
# for fused SDPA.  The SDPA backend and its tile/cache traffic must be profiled before
# assigning it a roofline verdict.
ops = [
    ("Encoder linear (QKV)",   ai_a, True),
    ("Attention (naive)",      ai_b, True),
    ("Attention (fused SDPA)", None, False),
    ("Projector (merger)",     ai_c, True),
    ("LLM prefill linear",     ai_d, True),
]

print()
print(f"{'Operation':<24} {'AI':>8} {'L4 verdict':>16}")
for name, ai, can_classify in ops:
    if not can_classify:
        print(f"{name:<24} {'n/a':>8} {'profile I/O':>16}")
        continue
    v_l4, _ = verdict(ai, "L4")
    print(f"{name:<24} {ai:>8.1f} {v_l4:>16}")
