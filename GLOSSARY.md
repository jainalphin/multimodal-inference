# Glossary

Plain meanings of the terms used in this project, each with an example from our own runs.

## Precision

- **fp32 (float32):** 32-bit numbers. Precise, the reference.
  - Example: the fp32 baseline, with all 236 answers saved.
- **fp16 (float16):** 16-bit numbers. Half the memory and faster, but the largest value is only 65,504.
  - Example: a value of 70,000 becomes `inf` in fp16.
- **bf16 (bfloat16):** 16-bit with the same range as fp32 but less precision. The T4 has no native support, so we use fp16 there.
- **fp16 vs bf16: why fp16 is still used**
  - Older GPUs don't have fast bf16. The T4 (Kaggle's free tier) is one, so fp16 is the only fast option there.
  - fp16 is more precise. bf16 pays for its big range with fewer precision bits, so it rounds more coarsely and causes more small answer flips.
  - So people often pick fp16 even on GPUs that support bf16, like the L4 (a common cloud inference GPU: Google Cloud G2, AWS G6).
  - Trade-off: fp16 is more precise but has the 65,504 ceiling (it overflowed in W2-2). bf16 can't overflow there but rounds more.
- **dtype:** a tensor's number format.
  - Example: `DTYPE = torch.float32`.

## Hardware and memory

- **T4:** a 16 GB GPU on Kaggle. fp32 Qwen (~14 GiB of weights) does not fit on one.
- **L4:** a 24 GB GPU. fp32 fits, which is where the baseline ran.
- **device_map="auto":** splits the model's layers across GPUs, filling GPU 0 first and then GPU 1. Only one GPU works at a time.
  - Example: on 2×T4, the vision encoder plus about 11 LLM layers go on GPU 0 and the rest on GPU 1.
- **max_memory:** how many GiB of weights each GPU may hold.
  - Example: `{0: "7GiB", 1: "13GiB"}` leaves GPU 0 room for the vision encoder's work.
- **OOM (out of memory):** there are two kinds.
  - **GPU OOM:** `torch.OutOfMemoryError: CUDA out of memory`. Too much on the GPU.
  - **CPU OOM:** `killed`, with no traceback. The machine ran out of RAM. Example: torchvision decoding a 76 s 1080p clip.
- **Fragmentation:** free memory is split into pieces too small to use.
  - Example: "4.34 GiB reserved but unallocated". Fixed with `expandable_segments:True`.

## Model inputs

- **Token:** the unit the model reads. Text words and image patches both become tokens.
- **Visual tokens:** one token per 28×28-pixel area of an image.
  - Example: a 1008×756 image gives 36×27 = 972 tokens.
- **max_pixels:** a cap on image size. Bigger images are shrunk to fit.
  - Example: DocVQA pages are capped at about 1M pixels (about 1,280 tokens). 190 of 200 pages were shrunk.
- **fps (frames per second):** how many video frames are sampled per second.
  - Example: 1 fps on a 20 s clip gives 20 frames. The cap is 32 frames.
- **Video reader:** the library that decodes video frames.
  - **torchvision:** decodes the whole clip into RAM, so a CPU OOM on long clips.
  - **decord:** decodes only the sampled frames. We must use decord for every run so the frames match.
- **Prompt template:** the fixed text wrapped around each question.
  - Example: `{question}` + "Answer the question using a single word or phrase."

## Model outputs

- **Greedy decoding:** always pick the most likely next token. Same input always gives the same answer, so runs are comparable.
- **Logits:** the model's raw score for every possible next token (152k of them) at each position.
  - Example: for a 4,000-token clip in fp32, the logits take about 2.5 GB.
- **Top-1:** the highest-scoring token.
  - Example: "top-1 match over all positions 0.9715" means fp16 picked the same token as fp32 at 97% of positions.

## The evaluation sets

- **Ground truth:** the known correct answer.
  - Example: `docvqa_237` has the answer "Maria Shulleeta".
- **Debug set:** 22 inputs with no answers. For traces and timing only.
- **Regression set:** fixed questions with ground truth, used to check that a change didn't break quality.
  - Image: 200 DocVQA questions.
  - Video: 36 MVBench questions.
- **Human subset:** 30 items I grade by hand, for when outputs change but scores don't show whether that matters.
- **Blind grading:** grading outputs labelled A/B without knowing which run made them, so I'm not biased.
- **DocVQA:** questions about scanned document pages (forms, letters, tables).
  - Example: "Who was writing this letter to Dr. Richard Carchman?"
- **MVBench:** multiple-choice questions about short video clips that need several frames to answer.
  - Example: "Which way is the cylinder moving? (A) left (B) right …"
- **lmms-eval:** a standard evaluation toolkit for multimodal models. We borrowed its prompts and metrics.

## Freezing

- **Freeze:** never change a set or baseline after creating it, so every run is tested on exactly the same thing.
- **Seed:** the starting number for random sampling. The same seed gives the same sample every time.
  - Example: seed `20260928` always picks the same 200 DocVQA questions.
- **sha256 / hash:** a fingerprint of a file. Change one byte and the hash changes completely.
  - Example: `score.py` refuses to run if a set's hash doesn't match the manifest.
- **Manifest:** the file listing every set's hash and the seed.
- **Baseline:** the reference run everything is compared against.
  - Example: fp32 scores 94.76 on DocVQA and 61.11 on MVBench, frozen in `eval/baselines/fp32/`.
- **JSONL:** a text file with one JSON object per line.
  - Example: `{"id": "docvqa_237", "output": "Maria Shulleeta"}`

## Scoring

- **Accuracy:** the share of questions answered correctly.
  - Example: 22 of 36 video questions right = 61.11%.
- **How "right" is defined for video (MVBench): exact letter match. It is *not* a distance metric.**
  - Each question stores the correct option letter as `answer_letter`.
  - `letter()` finds the first standalone letter A–H in the model's output.
  - Right = 1 if the two letters are equal, otherwise 0: `float(letter(out) == it["answer_letter"])`.

    | Model output | `answer_letter` | `letter()` finds | Result |
    |---|---|---|---|
    | `B` | B | B | right, 1 |
    | `(B)` | B | B | right, 1 |
    | `B. the cup` | B | B | right, 1 |
    | `C` | B | C | wrong, 0 |
    | `the cup` | B | none | wrong, 0 |

  - There's no partial credit: "C" is as wrong as no answer.
  - Weakness: a sentence like "A man picks up…" is read as option A.
- **How "right" is defined for images (DocVQA): ANLS. This one *is* a distance metric**, built on Levenshtein distance (below).
  - Each question gets 0 to 1, not just right or wrong. Near-miss spellings get partial credit.
  - The DocVQA score is the average of these per-question values.
- **In the bootstrap:** each resample averages the same per-question values. For video they are only ever 1 or 0; for DocVQA they range from 0 to 1.
- **Levenshtein distance:** the fewest single-character edits (insert, delete, replace) to turn one string into another.
  - Example: "Shuleeta" to "Shulleeta" is 1 edit (insert an "l").
- **ANLS:** DocVQA's metric. It gives partial credit for near-miss spellings.
  - Score = 1 − edits ÷ length of the longer string, and 0 if that's below 0.5.
  - "Maria Shuleeta" vs "Maria Shulleeta" scores 0.93.
  - "0.26" vs "0.28" scores 0.75.
  - "cat" vs "dog" scores 0.
- **Agreement with fp32:** the share of questions where the new run gave the same answer as fp32, whether right or wrong.
  - Example: "25 of 200 answers changed" means 87.5% agreement.
- **changed_ids:** the ids of the questions whose answers changed. These are the ones to look at by hand.

## Statistics

- **CI (confidence interval):** the range the true score probably falls in.
  - Example: video is 61.11% with a CI of [44.44, 77.78]. With only 36 questions, the true score could be anywhere from about 44% to 78%.
- **95% (confidence level):** the CI is built to contain the true score 95% of the time. It's a standard choice, not a measurement.
  - It comes from `np.percentile(means, 2.5)` and `(means, 97.5)`: cut 2.5% off each end, leaving the middle 95%.
- **Bootstrap:** getting a CI by resampling your own questions.
  - Draw 36 of your 36 questions at random, with repeats allowed. Score that sample. Repeat 10,000 times.
  - Sort the 10,000 scores. The 250th lowest and the 250th highest are the CI.
- **Why the video CI is wide:** each question is worth 2.8 points (100 ÷ 36), so a few lucky or unlucky questions move the score a lot.
  - Example: DocVQA has 200 questions and a CI of about ±3. Video has 36 questions and about ±17.
- **Paired difference:** the new score minus the fp32 score, computed on the *same* resampled questions each time.
  - Both runs face the same easy and hard questions, so the difficulty cancels out. This CI is much narrower than comparing two separate CIs.
- **Significant change:** the CI of the paired difference does *not* include 0.
  - Example: −1.04, CI [−1.61, −0.56]. Everything is below 0, so there's a real drop.
  - Example: +2.78, CI [−22.22, +27.78]. It includes 0, so it's just noise.

## fp16 stability (W2-2)

- **Hook:** a function PyTorch calls after a layer runs, used to inspect its output without changing the model.
- **absmax:** the largest absolute value a layer produced.
  - Example: `visual.blocks.31` reached 33,312.
- **Headroom:** how far from the fp16 limit, as 65,504 ÷ absmax.
  - Example: 65,504 ÷ 33,312 = 1.97×. Values could only double before overflowing.
- **Overflow:** a value too big for fp16 becomes `inf`.
- **NaN / Inf:** "not a number" and infinity. Once they appear they spread and ruin the output.
- **Tiny (subnormal):** values below 6.1e-5, the smallest normal fp16 number. They lose precision.
- **Drift:** how far a block's fp16 output is from its fp32 output.
  - `rel_max` is the largest difference ÷ the largest value. Example: 25% in the final vision block on the stress set.
- **Mixed precision:** fp16 for most layers, fp32 for the fragile ones. This is the fix if a layer overflows.

## When to run what

- **experiments/w02-0-baseline/run.py:** once, to make the frozen fp32 baseline. Already done.
- **score.py:** every time you have new answers.
  - It runs automatically at the end of `w02-0-baseline/run.py`, `w02-2-fp16-stability/task.py` and `w02-3-gpu-preprocess/run.py`.
  - To score an optimization by hand, set `PREDICTIONS` to the new answers and `BASELINE` to `eval/baselines/fp32/...`.
- **experiments/w02-2-fp16-stability/task.py:** twice, with `SET_NAME = "regression_image"` and then `"regression_video"`.
