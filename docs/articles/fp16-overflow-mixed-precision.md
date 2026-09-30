# Why Qwen2.5-VL outputs "!!!!!!!!" in fp16

I showed a vision-language model a scanned packing slip from 1982 ([Academic Press, one copy of *Banksias, Vol. 1*](https://www.industrydocuments.ucsf.edu/docs/mskw0228)) and asked it a simple question: "What is the weight of the shipment?"

The answer is printed right there on the slip, `WGT- 28.00`. In fp32 the model said:

```
28.00
```

Same model, same image, same question, in fp16:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Thirty-two exclamation marks. Not a wrong number, not a hallucinated one. Just the model screaming.

On the other 199 images it never screamed. This post is about why it lost its mind on exactly one of them, how three numbers out of millions caused it, and how running a tiny slice of the model in fp32 fixed it.

## The setup: why bother with fp16

I'm optimizing [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), a model that looks at images and video and answers questions about them.

Models normally run in [fp32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format): 32-bit numbers, precise and with a huge range. [fp16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) uses half the bits, so the model takes half the memory and runs a lot faster on GPU [tensor cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/). The catch is that fp16 is small in two ways:

- **It's less precise:** numbers get rounded more coarsely.
- **It has a ceiling:** the biggest number fp16 can hold is 65,504. Go above it and the number becomes `inf`, infinity. That's called an overflow.

There's also [bf16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), with fp16's size but fp32's range. I stuck with fp16 because it's still widely used, and I wanted to see whether this model survives it.

## The test: don't trust vibes

Eyeballing a few answers doesn't prove a precision change is safe, so I set up a proper test:

- **Real questions with known answers:** 200 document questions ([DocVQA](https://www.docvqa.org/)) and 36 video questions ([MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)), the same set for every run.
- **An fp32 reference.** The model's fp32 answers to all of them, saved once.
- **Two scores:**
  - how close the answers are to the correct ones ([ANLS](https://arxiv.org/abs/1905.13648) for documents, which gives partial credit for near-miss spellings; plain accuracy for video)
  - how many answers are the same as fp32

  Each comes with a confidence interval ([bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))): resample the questions 10,000 times and see how much the score wobbles just from which questions happened to be picked.
- **Hooks on every layer.** A [hook](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) is a small function PyTorch calls every time a layer runs. Mine recorded each layer's largest value and counted any `inf` or `NaN`.

Everything ran on one [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/) on [Lightning AI](https://lightning.ai/). Decoding was [greedy](https://huggingface.co/docs/transformers/generation_strategies), meaning always pick the most likely next word, so the same input always gives the same output. A rerun reproduced every number exactly.

## The results looked great

- **DocVQA score:** fp32 94.76, fp16 93.76
- **Same answer as fp32:** 198 of 200

On paper, fp16 looked fine: the drop in score wasn't statistically significant, and 99% of answers matched fp32. But when I looked at the two answers that changed, one of them was the screaming.

## Following the numbers

This is where the hooks paid off. They recorded the largest value in every layer, and only one layer went over fp16's limit: `visual.blocks.31.mlp.down_proj`, part of the MLP in the last block of the vision encoder.

Largest value in fp32, against fp16's ceiling of 65,504:

- `blocks.31.mlp.down_proj`: 66,861
- `blocks.31` (the block's output): 66,995
- `blocks.28–30` (next largest): about 14,000

On that one image, three values in this layer went past 65,504. Three numbers, out of millions. fp16 couldn't hold them, so they became infinity.

Why the last block? A vision transformer passes a running total, the [residual stream](https://transformer-circuits.pub/2021/framework/index.html), from block to block. Each block adds its result to the total, and no norm ever rescales it along the way. So the values creep up: about 34 in block 0, about 900 by block 15, about 14,000 through blocks 19–30. Then block 31 jumps almost 5×, to 66,995, straight through the ceiling. A few oversized values like this in late layers, sometimes called massive activations, have been reported in other transformers too ([Sun et al., 2024](https://arxiv.org/abs/2402.17762)). The same block-31 overflow has also been reported for this model running in Ollama ([MaxusAI/ollama#216](https://github.com/MaxusAI/ollama/issues/216)), where it shows up as `?` repeated instead of `!`.

## How three numbers ruin an entire answer

Three infinities in one layer sound harmless. They aren't, because infinity turns into something worse, and the hooks show every step of it. The counts add up exactly:

1. **3 `inf`** in `blocks.31.mlp.down_proj`: three values over the ceiling.
2. **3 `NaN`** in `merger.ln_q`. This is a [normalization](https://arxiv.org/abs/1910.07467): it divides each 1,280-value row by the row's size. With infinity in the row, the size is infinity too. The infinite entry becomes `inf / inf`, which isn't a number at all: `NaN`, ["not a number."](https://en.wikipedia.org/wiki/NaN) Every other value in that row becomes `finite / inf = 0`. Either way, the row is destroyed.
3. **15,360 `NaN`** in the merger. It glues 4 image patches into one token (4 × 1,280 = 5,120 values), so 3 bad values ruin 3 whole tokens.
4. **6,144 `NaN`** at the merger's output. That's 3 × 2,048: exactly 3 of the image's tokens are now garbage.
5. **Millions of `NaN`** in the language model. [Attention](https://arxiv.org/abs/1706.03762), the mechanism that lets every token look at every other token, mixes those 3 poisoned tokens into every token after them.
6. **The final output.** Every score is `NaN`, so there's nothing to choose between, and the model falls back to token 0. In Qwen's vocabulary, token 0 is `!`.

Three numbers too big for fp16 → three poisoned image tokens → attention spreads the poison everywhere → a model that can only say `!`.

## The average didn't even blink

That broken image moved the score by half a point out of 100. The confidence interval of the difference just touched zero, so by the statistics nothing happened.

But nothing about that image was "slightly worse." It was no answer at all. On a real service, that's 1 request in 200 coming back as a wall of exclamation marks. **Averages measure quality. They don't catch crashes.** You have to look for NaN, or at least at which answers changed.

## The fix: give one block a bigger box

The fix is mixed precision: keep most of the model in fp16, and run only the fragile part in fp32. PyTorch's own [`torch.autocast`](https://docs.pytorch.org/docs/stable/amp.html) does this for known-risky operations like softmax. Here the fragile part was found by measuring.

The fp32 part can't be just the one layer, though. It has to last until the numbers are small again. Block 31's output is about 67,000; convert it back to fp16 right after the block and it overflows all over again. The first thing that shrinks it is the normalization at the start of the merger. So the fp32 part is block 31 plus the merger:

```
blocks 0–30     fp16
   ↓ convert to fp32
block 31        fp32   ← 67,000 fits here
merger          fp32   ← shrinks everything; its output peaks at about 50
   ↓ convert back to fp16
language model  fp16
```

In code, that's the model loaded in fp16 and changed in place:

```python
visual = model.model.visual

visual.blocks[31].float()                              # block 31 in fp32
visual.blocks[31].register_forward_pre_hook(           # its fp16 inputs -> fp32
    lambda m, args, kwargs: (to_fp32(args), {k: to_fp32(x) for k, x in kwargs.items()}),
    with_kwargs=True)

visual.merger.float()                                  # merger in fp32 too
visual.merger.register_forward_hook(lambda m, args, out: out.half())   # small again -> fp16
```

A [pre-hook](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook) runs just before a layer and can change its inputs; a forward hook runs just after and can replace its output. The two hooks convert numbers at the edges of the fp32 part, and `to_fp32` converts the hidden states and the position embeddings.

## Did it work?

```
                        fp32      plain fp16    mixed
Layers with NaN/Inf     0         404 of 694    0
down_proj peak          66,861    overflow      66,864 (no problem)
The screaming image     28.00     !!!!!!!!      28.00
DocVQA score            94.76     93.76         94.26
Same answer as fp32     -         198 / 200     199 / 200
```

The fix worked: no NaN anywhere, and the packing slip is answered correctly again. Block 31 still produces the same large value; fp32 simply has room for it.

On the 36 video clips, plain fp16 never overflowed, but block 31 peaked at 56,875, only 13% under the ceiling. With the fix, video also had zero NaN and 36 of 36 answers identical to fp32.

## The one that's still wrong

One answer still differs from fp32, on a different image:

- **Question:** "What is printed against the serial number 3 in the 2nd column?"
- **fp32:** `Grady, 1995` (correct)
- **fp16, and mixed:** `6471` (incorrect)

No NaN here. The image is a table of references, and the row for serial number 3 reads `3 | Grady, 1995 | 6471`. fp16 found the right row but read the 3rd column instead of the 2nd ([see the original document](https://www.industrydocuments.ucsf.edu/docs/yscw0217), page 60). It was probably a close call between the two, and fp16's rounding tipped it the other way. The fix doesn't bring this one back: the mixed run also answers `6471`. I'll look at recovering it in a future post.

## What if a block in the middle had overflowed?

Here the overflow was in the last block, so the fp32 part was short. If it had been a block in the middle, say block 15, the rule stays the same: **keep fp32 from the layer that overflows until the numbers fit back under 65,504.** Where that happens depends on where the big number lives.

**Case 1: the big number is in the running total, so fp32 all the way to the end.**

Each block works like this:

```
x = x + attn(norm1(x))     ← the block adds its result to x
x = x + mlp(norm2(x))      ← x is the running total, passed on to the next block
```

The norms only rescale what goes *into* attention and the MLP; `x` itself is never normalized between blocks, only added to. So if `x` hit 67,000 in block 15, no norm would bring it back down before the merger. A later block could cancel it out, but in this model the peaks mostly kept growing. If `x` stays above 65,504, you'd need fp32 from block 15 all the way through the merger, and the hooks will show you whether it does.

```
block 14 (fp16) → blocks 15–31 (fp32) → merger (fp32) → back to fp16 → language model (fp16)
                  x stays ~67,000        shrinks x
```

**Case 2: the big number is only inside the block, so just that block in fp32.**

Sometimes a value spikes inside a block but the block's final output is small. For example, the MLP's hidden layer spikes but `down_proj` brings it back down. Then the block's output fits in fp16, and you can convert back right after it:

```
block 14 (fp16) → block 15 (fp32) → back to fp16 → blocks 16–31 (fp16)
```

**How to tell which case you're in:** the hooks record both the block's own output and every layer inside it. If the block's output is above 65,504, it's case 1. If only a layer inside it is, it's case 2. Here, block 31's own output was 66,995, so it was case 1, and that's why the merger had to be in fp32 too.

And on a GPU with bf16 (A100, L4, H100), bf16 avoids this overflow entirely, because its range matches fp32.

## The small print

- **Scope:** one model (Qwen2.5-VL-3B-Instruct) on an NVIDIA L4.
- **Coverage:** 200 document images and 36 video clips. The fix passed on both.
- **Not a guarantee.** The fix protects the layer that overflowed on these inputs. Blocks 28–30 sit 4.7× under the ceiling, so a much more extreme image could still get them.
- **Speed cost:** not measured yet. The fix puts one of 32 vision blocks plus the merger in fp32.

## Takeaways

1. **Look at what changed, not just the score.** 99% agreement and "no significant change" were hiding a model that screamed.
2. **Hook every layer.** Counting `inf` and `NaN` layer by layer turned "fp16 is weird sometimes" into "these three numbers, in this layer, on this image."
3. **Keep the fp32 part small, but finish the job.** Protect the layer that overflows *and* everything after it until the numbers are small again.

---

## References

- **Model:** [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) by the Qwen team, Alibaba Cloud. [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923), Bai et al., 2025.
- **Document questions:** [DocVQA](https://www.docvqa.org/). [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398), Mathew, Karatzas and Jawahar, WACV 2021. Document images from the [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/). Copy used: [lmms-lab-encoder/DocVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA).
- **Video questions:** [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) by OpenGVLab. [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/abs/2311.17005), Li et al., CVPR 2024.
- **Prompts and metric conventions:** [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- **Related findings:** [Massive Activations in Large Language Models](https://arxiv.org/abs/2402.17762), Sun, Chen, Kolter and Liu, 2024. The same overflow in Ollama: [MaxusAI/ollama#216](https://github.com/MaxusAI/ollama/issues/216).
- **Software:** [PyTorch](https://pytorch.org/), [Hugging Face transformers](https://github.com/huggingface/transformers), [qwen-vl-utils](https://pypi.org/project/qwen-vl-utils/), [decord](https://github.com/dmlc/decord).
- **Compute:** one NVIDIA L4 GPU on [Lightning AI](https://lightning.ai/).

## Yes, I used AI

I used [Claude](https://www.anthropic.com/claude) while working through this: to write code, explain things I didn't know, and draft this post. I'm sharing what I learned so I (and you) don't have to spend AI tokens rediscovering the same thing. Every number here comes from my own runs.
