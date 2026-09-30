# W2-2: fp16 stability notes

## Which layers need fp32?

The rule is: **keep fp32 from the layer that overflows until the values fit back under 65,504.** Where that happens depends on where the big value lives.

### Case 1: the big value is in the residual stream, so fp32 to the end

This is our case. Each vision block works like this:

```
x = x + attn(norm1(x))     ← the block adds its result to x
x = x + mlp(norm2(x))      ← x is the "residual stream", passed on to the next block
```

The norms only shrink what goes **into** attention and the MLP. `x` itself is never shrunk between blocks. So if `x` becomes 67,000 in block 15, nothing forces it back down. A later block could cancel it, but in our data the peaks mostly grew block to block (87% of blocks on images, 97% on video).

So if a middle block, say block 15, overflowed in the residual stream, you'd most likely need **blocks 15–31 and the merger in fp32** (check with the hooks), because the first thing that shrinks `x` is `merger.ln_q`. Converting back to fp16 anywhere before that would overflow again.

### Case 2: the big value is only inside the block, so just that block in fp32

Sometimes a value overflows inside a block, but the block's final output is small. For example:
- the MLP's hidden layer (3,420 wide) spikes, but `down_proj` brings it back down
- attention scores overflow before the softmax

The block's output `x` then fits in fp16, so you can convert back right after it:

```
block 14 (fp16) → block 15 (fp32) → back to fp16 → blocks 16–31 (fp16)
```

### How to tell which case you're in

Look at the `absmax` column in `stability_per_module.csv`:
- **The block's own output** (the row for `visual.blocks.15`): if it's above 65,504, you're in case 1 and need fp32 to the end.
- **If only an internal layer is above 65,504** (like `blocks.15.mlp.act_fn`) but the block's output is fine, you're in case 2 and only that block needs fp32.

In our results, the row for `visual.blocks.31` itself is 66,995, so the block's own output overflows. That's case 1, and it's why the merger is in fp32 too.

### A cheaper alternative for case 1

If a middle block overflowed and blocks 15–31 in fp32 was too slow, the standard trick is to **keep only the residual stream `x` in fp32**. Attention and the MLP stay fp16, and just the additions (`x = x + …`) run in fp32. Large models are commonly trained this way ("fp32 residual"). It's cheaper, because almost all the computation is in attention and the MLP, but it needs a modified block rather than a `.float()` call. It only helps if the big value builds up in the residual stream. If a layer's own output is too big for fp16, like our `down_proj` at 66,861, that layer still overflows. For block 31 at the end, `.float()` on one block is simpler and good enough.

## What the hooks can and can't see

The hooks are attached to more than the blocks. In `task.py`, `attach_hooks` hooks every **leaf module** (the smallest named layers) plus every block:

```python
leaf = len(list(m.children())) == 0      # a layer with no sub-layers: Linear, activation, norm
block = is_block(m)                      # a whole VisionBlock / DecoderLayer
if not (leaf or block): continue
```

So each block gets one row for **the block's output** and one row for **each piece inside it**. For block 31, `stability_per_module.csv` has:

```
model.visual.blocks.31                  ← the block's output (the residual stream x)
model.visual.blocks.31.norm1            ← inside: the norm before attention
model.visual.blocks.31.attn.qkv         ← inside: the q/k/v projection
model.visual.blocks.31.attn.proj        ← inside: the attention output projection
model.visual.blocks.31.norm2            ← inside: the norm before the MLP
model.visual.blocks.31.mlp.gate_proj    ← inside
model.visual.blocks.31.mlp.act_fn       ← inside
model.visual.blocks.31.mlp.up_proj      ← inside
model.visual.blocks.31.mlp.down_proj    ← inside: this is where 66,861 appeared
```

Our table showed both kinds:

```
model.visual.blocks.31                66,995   ← the block's output
model.visual.blocks.31.mlp.down_proj  66,861   ← the internal layer
```

That's how we know it's case 1: the block's output row is above 65,504 too, not just the internal layer.

What the hooks **can't** see are calculations that aren't separate layers:
- **The residual addition** `x = x + mlp(...)`. It's plain `+` in the code, not a module, but its result is exactly the block's output row, so you still see it.
- **The MLP multiply** `act_fn(gate) * up`. Not a module; its result goes straight into `down_proj`, and the hook only sees `down_proj`'s output, not its input.
- **Attention scores and the softmax** inside `F.scaled_dot_product_attention`. It's one function call, so the intermediate scores are invisible; the hook only sees `attn.proj`'s output.

So an overflow in one of those hidden steps would only show up as NaN or Inf in the next visible layer, without its own row. For block 31 that didn't happen: the overflow is in `down_proj`'s own output, which the hooks capture directly.
