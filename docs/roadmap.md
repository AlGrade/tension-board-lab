# Roadmap: the boulder problem generator

The goal is a React application without a backend. You enter a grade, a wall angle, and a
style, and you see a **newly generated** problem on a rendered Tension Board 2 12x12.
Transferring the problem to the wall over Web Bluetooth is an optional last step.

What already exists: the grade predictor (2.85 million parameters, graph transformer, mean
absolute error of 0.935 V grades) and the full data pipeline over the Aurora database—21,809
consensus problems and 49,512 problems with at least one ascent. See
[`src/tension_board_lab/`](../src/tension_board_lab/README.md).

## The chosen approach

A **conditional autoregressive generator**. A new, small model learns from real problems what
a problem looks like, conditioned on angle, target grade, and style features. The existing
grade predictor stays unchanged and acts as an independent critic during sampling.

Rejected: generating at random and filtering with the grade model. With 498 hold positions per
layout and about 10 holds per problem the search space is roughly 10²⁰, and the grade model
cannot express *this is not a problem*—rejection sampling would selectively find its errors.
The autoregressive generator solves both structurally: it samples in-distribution by
construction, and its own sequence likelihood `log p(problem)` is a free plausibility score.
That also removes the need for a separate realism discriminator.

**There are exactly two models:** the existing grade critic and the new generator.

## Why one repository

1. **Split correctness.** Both models must use the same `split_examples` logic with
   `configuration_signature` and mirror canonicalization
   ([data.py:83-132](../src/tension_board_lab/data.py#L83-L132)). If the generator trains on
   problems that sit in the critic's test split, every *generator hits the target grade*
   evaluation is worthless. Shared code makes that mistake impossible; copied code makes it
   likely.
2. **Shared pipeline.** `schema.py`, `grades.py`, `aurora.py`, and `data.py` are needed
   unchanged by both models.
3. **Artifact version consistency.** The frontend loads two ONNX files, a vocabulary, and the
   hold catalog, and they have to match each other. One commit is one consistent set.
4. The cost is low: `pyproject.toml` and `web/package.json` coexist without conflict.

### Target layout

Shared code sits at the top level of the package, and each model gets a subpackage below it.
`grade/` already follows this shape; `generator/` and `export/` join it.

```
src/tension_board_lab/
  schema.py grades.py aurora.py data.py     unchanged, shared by both models
  catalog.py                                done  board placements, position -> hold lookup
  style.py                                  done  rule-based style features (shared)
  grade/
    model.py train.py predict.py            unchanged (grade critic)
  generator/
    tokenizer.py                            done  problem <-> token sequence
    model.py                                new  decoder-only transformer
    train.py                                new  tension-train-generator
    sample.py                               new  tension-sample (offline evaluation)
  export/
    onnx.py                                 new  tension-export-onnx (both models)
    artifacts.py                            new  tension-export-web (catalog, vocabulary, ids)
web/                                        new  Vite + React + TypeScript
  public/models/*.onnx                      git-ignored, produced by the export
  public/data/*.json                        hold catalog, vocabulary, style presets
  src/board/                                SVG renderer
  src/model/                                featurization + onnxruntime-web sessions
  src/generate/                             sampling loop, logit masking, style filter
  src/style.ts                              mirror of style.py, tested for parity
```

`tests/` mirrors the package, so generator and export tests land in `tests/generator/` and
`tests/export/`. Checkpoints and reports are already split by model—`checkpoints/generator/`
and `reports/generator/` need no new convention.

`.gitignore` already covers `checkpoints/*` and the `web/` entries.

## Generator design

### Representation

The decisive simplifier: **within one layout an `(x, y)` pair is unique**. Exactly one hold
exists per position, and wood and plastic never collide—498 positions per layout, 537 distinct
coordinates across both. Hold type and orientation therefore follow deterministically from the
position via a catalog lookup. The generator only picks `(position, role)`.

- **Vocabulary:** 537 coordinates x 4 roles = 2,148 hold tokens, plus 4 special tokens and 71
  conditioning tokens—**2,223** in total. Implemented in
  [generator/tokenizer.py](../src/tension_board_lab/generator/tokenizer.py).
- **Order:** problems are sets, not sequences. Sort deterministically by ascending `(y, x)`.
  That is physically sensible—bottom to top—and makes starts early tokens and finishes late
  ones.
- **Conditioning prefix:** `[layout][angle][grade][style x7]`, a fixed 10 tokens before the
  first hold. Angle and grade get one token per value rather than a bucket range, so both
  survive a round trip exactly.
- **Unspecified style:** each style feature has one slot past its buckets meaning *any*. A
  preset pins only two or three of the seven features, and inventing values for the rest would
  over-constrain sampling. Training drops individual buckets at random so the model sees
  partial style requests.
- **Sequence length:** 35 holds at most in the dataset, plus the 10-token prefix and BOS/EOS,
  so 47.

### Architecture

Decoder-only transformer, causal mask, tied input and output embeddings. Target size is width
192, 6 layers, 8 heads, feed-forward expansion 4—about **3.1 million parameters** (12.4 MB
fp32, roughly 3.1 MB int8). Both models quantized to int8 put the total frontend download at
around 6 MB.

### Training

- **Data:** the 49,512 examples in `tb2_12x12_pretrain.jsonl` (at least one ascent). For a
  generator what matters is what people actually set, not grade consensus, so noisy grade
  conditioning is acceptable. Down-weight examples with few ascents through `weights`, as in
  [data.py:181-184](../src/tension_board_lab/data.py#L181-L184).
- **Split:** reuse `split_examples` from [data.py](../src/tension_board_lab/data.py), plus the
  same exclusion rule as the critic's pretraining
  ([train.py:193-197](../src/tension_board_lab/grade/train.py#L193-L197)). The critic's
  validation and test configurations must not enter generator training.
- **Classifier-free guidance:** replace the conditioning prefix with `UNCOND` in 10% of
  training steps. At sampling time this exposes a guidance scale that trades grade fidelity
  against variety. It is the most important lever for the actual product promise—*give me a
  V7*—and costs almost nothing during training.
- Loss: standard cross-entropy over tokens, ignoring `PAD`.

### Sampling in the browser

1. The user picks grade, angle, style, and layout.
2. Sample about 24 candidates in parallel (nucleus sampling plus a CFG scale).
3. **Apply hard rules as logit masking during sampling**, not as a post-filter—far more
   efficient than discarding:
   - mask positions outside the chosen layout;
   - mask positions already used (one role per position);
   - mask `EOS` while there are fewer than 2 holds, or no start, or no finish;
   - allow finish tokens only above a minimum height and start tokens only below one;
   - cap the hold count by style.
4. Run all candidates through the grade critic in a **single** batch and take the
   **expectation** `sum p_i * i`—not `argmax`, which is too noisy at around 38% confidence.
5. Rank by `w1 * |E[grade] - target| + w2 * style_distance - w3 * log p(problem)`. Show the
   best candidate; the rest fill a *next suggestion* button.

Runtime: roughly 15 generator forward steps through a 3M model, and one batched critic call
over at most 35 nodes. Both are well under a second with `onnxruntime-web`.

## Style: rule-based, one source of truth

Style is defined in code in `src/tension_board_lab/style.py` and mirrored in `web/src/style.ts`,
with a parity test. No labels, no hand annotation of hold types.

Reference values from the corpus of 21,809 problems: on average 1.54 start, 4.51 hand, 3.35
foot, and 1.06 finish holds; median 10 holds in total, p95 16, maximum 35.

The per-problem feature vector is purely geometric, computable from `(x, y, role)`:
`hand_count`, `foot_count`, `mean_move_length`, `max_move_length`, `height_span`,
`foot_to_hand_ratio`, `move_length_variance`.

Presets are ranges over those features:

Thresholds are training-split percentiles: *high* is p70 or above, *short* is p30 or below, and
dyno's cutoff is p90. The last column is the share of the 17,477 training problems that match.

| Style | Definition | Corpus |
| --- | --- | --- |
| power | at most 4 hand holds, high mean and maximum move length | 12.1% |
| endurance | at least 8 hand holds, short moves, high y coverage | 1.3% |
| dyno | high maximum single move length, otherwise normal hold count | 7.2% |
| technical | high foot density relative to hand holds, short moves | 12.1% |

Endurance is rare because hand counts above 6 are rare on this board—p90 is 6 hand holds. The
UI should treat an endurance request the same way it treats V13: honestly.

The same code serves both uses: at training time the features are part of the conditioning
prefix (discretized into buckets), computable for every training example and therefore free of
labeling cost; at sampling time they provide `style_distance` for ranking and the hold-count
bound for logit masking.

Crimpy, slopey, and pinchy stay inexpressible—those would need a hand-annotated taxonomy of the
106 hold types. Deliberately out of scope, and retrofittable without changing anything else.

## Steps

### Step 1: style features and tokenizer — done

[style.py](../src/tension_board_lab/style.py) with the feature computation, bucket edges, and
the presets; [catalog.py](../src/tension_board_lab/catalog.py) for the position-to-hold lookup
both the tokenizer and the export need; and
[generator/tokenizer.py](../src/tension_board_lab/generator/tokenizer.py) with the
problem-to-token-sequence mapping, layout handling, and `(y, x)` ordering. Covered by
[tests/test_style.py](../tests/test_style.py) and
[tests/generator/test_tokenizer.py](../tests/generator/test_tokenizer.py).

### Step 2: train the generator

`generator/model.py` and `generator/train.py` with a `tension-train-generator` CLI, following
the style of [train.py](../src/tension_board_lab/grade/train.py)—argparse triad, JSON lines on
stdout, checkpoint dict with `format_version`. Reuse the split and exclusion rules. Add
`tension-sample` as an offline CLI so results can be judged before the frontend exists.

### Step 3: export

`export/onnx.py` (`tension-export-onnx`) for both models, and `export/artifacts.py`
(`tension-export-web`) for the hold catalog JSON, the vocabulary, `grade_labels`, the
calibration temperature of 1.67, and the `placement_id` table.

Known pitfalls when exporting the critic:

- **Do not quantize to fp16.** `masked_fill` uses `torch.finfo(dtype).min`
  ([model.py:86](../src/tension_board_lab/grade/model.py#L86),
  [model.py:145](../src/tension_board_lab/grade/model.py#L145)); in fp16 that overflows to NaN.
  Use int8 or fp32, or change the constant to `-1e4` before exporting.
- Set dynamic axes for both batch **and** node count, or the exporter bakes in the tracing
  shapes ([model.py:57](../src/tension_board_lab/grade/model.py#L57)).
- The `mask` input is bool ([data.py:171](../src/tension_board_lab/data.py#L171)). An export wrapper
  with a float mask is cleaner.
- [`forward_batch`](../src/tension_board_lab/grade/train.py#L43) already defines the exact six-input
  signature and its order.

Coordinate inversion for rendering and Bluetooth, since normalization lives only in the SQL:
`raw_x = x * 128 - 64` and `raw_y = y * 136 + 4`.

### Step 4: React app

Vite, React, and TypeScript under `web/`. An SVG board renderer driven by the catalog JSON,
with Aurora's role colors: `00DD00` start, `0066FF` hand, `FF0000` finish, `FF00FF` foot. The
featurization in `web/src/model/` is an exact mirror of `collate_routes`.

Details where a deviation silently produces wrong predictions:

- orientation is `[sin, cos]`, **in that order**
  ([data.py:195](../src/tension_board_lab/data.py#L195));
- coordinates are passed **raw**, with no rescaling;
- the angle is passed in **raw degrees**; normalization happens inside the model;
- unknown hold types silently become index 0
  ([data.py:193](../src/tension_board_lab/data.py#L193));
- role order is `start=0, hand=1, foot=2, finish=3`;
- confidence is `softmax(logits / 1.67)`.

### Step 5: sampling UI

Inputs for grade, angle, style, and layout; a generate action; the result with its grade and
confidence; a *next suggestion* button; and an editor that toggles individual holds with live
re-scoring by the critic.

### Step 6 (optional, last): Web Bluetooth

Frames in the format `p<placement_id>r<role_id>`, with role ids 1=start, 2=hand, 3=finish,
4=foot for product 4, and 5/6/7/8 for product 5. State the limitations up front: Chromium only
(desktop and Android), **not Safari or iOS**, and it requires HTTPS and a user gesture. The
wall's BLE GATT protocol is not documented in this repository and has to be verified
externally.

## Where the UI has to be honest

Grade distribution in the consensus set: V0 810, V1 1,356, V2 1,178, V3 2,546, V4 2,893,
V5 3,140, V6 2,403, V7 2,170, V8 2,544, V9 1,085, V10 893, V11 502, **V12 212, V13 74, V14 3**.
Angles: 35° 2,216, **40° 12,153**, 45° 5,153, 50° 1,898, **55° 389**.

Above V11 and at 55° both the generator's training data and the critic's accuracy are thin—mean
absolute error at 55° is 1.063 against 0.849 at 35°. The UI should show that rather than fake
precision.

## Verification

**Python tests**, following the style of [tests/test_data.py](../tests/test_data.py)—module
helpers instead of fixtures, descriptive test names, `-> None`, absolute imports:

- `test_style.py`: style features against hand-computed problems; presets separate corpus
  examples.
- `test_tokenizer.py`: lossless roundtrip; layout separation; deterministic ordering.
- `test_export.py`: ONNX parity—the same `RouteExample` objects through PyTorch and
  `onnxruntime`, logits within about 1e-4, checked at 2, 10, and 35 nodes.

**Generator evaluation** with `tension-sample` against the holdout:

- negative log-likelihood on the validation split;
- **grade fidelity:** sample N problems per target grade and measure the mean absolute error
  between the target and the critic's `E[grade]`, across several CFG scales to show the
  trade-off against variety;
- **novelty:** the fraction of samples whose `configuration_signature` already exists in the
  corpus (should be low), plus edit distance to the nearest corpus neighbor;
- **validity:** the fraction of samples violating hard constraints—must be 0;
- **distribution fidelity:** hold-count and role distributions against the corpus, targeting
  about 1.5 start, 4.5 hand, 3.4 foot, and 1.1 finish holds.

**JS/Python parity:** test the TypeScript featurization and `style.ts` against reference
fixtures exported from Python—problem, expected tensors, expected logits. This is the most
error-prone point in the whole undertaking.

**Manual, and not optional:** look at 20 generated problems and judge for yourself whether they
look climbable and whether the grade is plausible. No automated test replaces that.

**End to end:** `npm run dev` in `web/`, generate a problem, change holds in the editor, and
watch whether the displayed grade moves plausibly.
