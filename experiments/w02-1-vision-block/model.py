import copy
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # [D]
        self.variance_epsilon = eps

    def forward(self, hidden_states):  # [N, D] -> [N, D]
        input_dtype = hidden_states.dtype
        # so hidden_state ** 2 dont overflow in float16
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [N, heads, head_dim]; cos, sin: [N, head_dim]
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype

    q, k = q.to(torch.float32), k.to(torch.float32)
    cos = cos.unsqueeze(1).to(torch.float32)  # [N, 1, head_dim]
    sin = sin.unsqueeze(1).to(torch.float32)

    q_embedding = q * cos + rotate_half(q) * sin
    k_embedding = k * cos + rotate_half(k) * sin
    return q_embedding.to(orig_q_dtype), k_embedding.to(orig_k_dtype)


class Attn(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.scaling = self.head_dim ** -0.5

    def forward(self, hidden_states, position_embeddings, cu_seqlens):
        seq_len = hidden_states.shape[0]

        # [N, D] -> [N, 3*D] -> [N, 3, heads, head_dim] -> 3 x [N, heads, head_dim]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # v is not rotated

        # [N, heads, head_dim] -> [1, heads, N, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # block-diagonal mask: tokens attend only inside their own segment
        seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        seg_id = torch.repeat_interleave(
            torch.arange(len(seg_lens), device=cu_seqlens.device), seg_lens
        )
        mask = seg_id[:, None] == seg_id[None, :]  # [N, N] bool, True = allowed

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size, mlp_hidden):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden, bias=True)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden, bias=True)
        self.down_proj = nn.Linear(mlp_hidden, hidden_size, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attn(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size, mlp_hidden)

    def forward(self, x, position_embeddings, cu_seqlens):
        x = x + self.attn(self.norm1(x), position_embeddings, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------- load model, build my block ----------
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
model.eval()

visual = model.model.visual.to("cuda")  # older transformers: model.visual
device = next(visual.parameters()).device
vc = model.config.vision_config
processor = AutoProcessor.from_pretrained(model_id)

mine = Block(vc.hidden_size, vc.num_heads, vc.intermediate_size).to(device).eval()


# ---------- capture reference inputs and compare ----------
def diff(a, b, name):
    d = (a.float() - b.float()).abs()
    print(f"{name}: max {d.max().item():.3e}  mean {d.mean().item():.3e}")


def run(img, block_idx):
    inp = processor(images=img, text="<|vision_start|><|image_pad|><|vision_end|>", return_tensors="pt")
    ref = visual.blocks[block_idx]
    cap = {}

    def hook(module, args, kwargs, output):
        print("block input shape:", args[0].shape)
        cap["x"] = args[0].detach()
        cap["kw"] = kwargs
        cap["out"] = output.detach()

    h = ref.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        visual(inp["pixel_values"].to(device, torch.float32),
               grid_thw=inp["image_grid_thw"].to(device))
    h.remove()

    x = cap["x"]
    pos = cap["kw"]["position_embeddings"]
    cu = cap["kw"]["cu_seqlens"]

    mine.load_state_dict(ref.state_dict())  # same architecture, pretrained weights of this block
    with torch.no_grad():
        out = mine(x, pos, cu)

    d = (out - cap["out"]).abs()
    print(f"block {block_idx} | N={x.shape[0]} | segments={len(cu) - 1} | cu[-1]={cu[-1].item()} "
          f"| max {d.max().item():.3e} mean {d.mean().item():.3e}")
    return ref, x, pos, cu, cap["out"]


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# fp32: three images (full-attention block 7), plus windowed block 0
run(img, 7)
run(img.resize((448, 448)), 7)
run(img.resize((336, 504)), 7)
run(img, 0)

# piece-by-piece on block 7
ref, x, pos, cu, ref_out = run(img, 7)
with torch.no_grad():
    n1_ref = ref.norm1(x)
    diff(mine.norm1(x), n1_ref, "norm1")

    a_ref = ref.attn(n1_ref, cu_seqlens=cu, position_embeddings=pos)
    diff(mine.attn(n1_ref, pos, cu), a_ref, "attn")

    x1 = x + a_ref
    n2_ref = ref.norm2(x1)
    diff(mine.norm2(x1), n2_ref, "norm2")

    diff(mine.mlp(n2_ref), ref.mlp(n2_ref), "mlp")

# fp16
ref16 = copy.deepcopy(ref).half()
mine16 = copy.deepcopy(mine).half()
x16 = x.half()
pos16 = (pos[0].half(), pos[1].half())
with torch.no_grad():
    r16 = ref16(x16, cu_seqlens=cu, position_embeddings=pos16)
    m16 = mine16(x16, pos16, cu)
diff(m16, r16, "fp16 mine vs fp16 ref")
diff(m16, ref_out, "fp16 mine vs fp32 ref")
print("relative error vs fp32 ref:", ((m16.float() - ref_out).abs().max() / ref_out.abs().max()).item())