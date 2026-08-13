# Roadmap: the boulder problem generator

> **A working record, not current documentation.** This is the plan the generator was built
> from, kept with the outcomes written back into it: what each step actually produced, where
> the plan was wrong, and what the measurements said. For how the code works today, read
> [`src/tension_board_lab/`](../src/tension_board_lab/README.md) and [`web/`](../web/README.md).

The goal is a React application without a backend. You enter a grade and a wall angle, and you
see a **newly generated** problem on a rendered Tension Board 2 12x12. Transferring the problem
to the wall over Web Bluetooth is an optional last step.

What already exists: the grade predictor (2.85 million parameters, graph transformer, mean
absolute error of 0.935 V grades) and the full data pipeline over the Aurora database—21,809
consensus problems and 49,512 problems with at least one ascent. See
[`src/tension_board_lab/`](../src/tension_board_lab/README.md).

## The chosen approach

A **conditional autoregressive generator**. A new, small model learns from real problems what
a problem looks like, conditioned on angle and target grade. The existing
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
  grade/
    model.py train.py predict.py            unchanged (grade critic)
  generator/
    tokenizer.py                            done  problem <-> token sequence
    physical.py                             done  hold type and orientation per token
    constraints.py                          done  legal-token masks for sampling
    evaluate.py                             done  tension-eval-generator
    model.py                                done  decoder-only transformer
    train.py                                done  tension-train-generator
    sample.py                               done  tension-sample (offline evaluation)
  export/
    onnx.py                                 new  tension-export-onnx (both models)
    artifacts.py                            new  tension-export-web (catalog, vocabulary, ids)
web/                                        new  Vite + React + TypeScript
  public/models/*.onnx                      git-ignored, produced by the export
  public/data/*.json                        hold catalog, vocabulary, constraints
  src/board/                                SVG renderer
  src/model/                                featurization + onnxruntime-web sessions
  src/generate/                             sampling loop, logit masking
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

- **Vocabulary:** 537 coordinates x 4 roles = 2,148 hold tokens, plus 4 special tokens and 30
  conditioning tokens—**2,182** in total. Implemented in
  [generator/tokenizer.py](../src/tension_board_lab/generator/tokenizer.py).
- **Order:** problems are sets, not sequences. Sort deterministically by ascending `(y, x)`.
  That is physically sensible—bottom to top—and makes starts early tokens and finishes late
  ones.
- **Conditioning prefix:** `[layout][angle][grade]`, a fixed 3 tokens before the first hold.
  Angle and grade get one token per value rather than a bucket range, so both survive a round
  trip exactly.
- **Physical context:** the hold type and orientation behind each token are added as an
  embedding. A token names only a position, so without this the model can learn what a position
  affords only by memorizing all 537 separately, while hold type generalizes across them. The
  layout is a second model input, because the same position carries a different hold on each
  layout and guidance blanks the request rather than the wall.
- **Sequence length:** 35 holds at most in the dataset, plus the 3-token prefix and BOS/EOS,
  so 40.

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

1. The user picks grade, angle, and layout.
2. Sample about 24 candidates in parallel (nucleus sampling plus a CFG scale).
3. **Apply hard rules as logit masking during sampling**, not as a post-filter—far more
   efficient than discarding:
   - mask positions outside the chosen layout;
   - mask positions already used (one role per position);
   - mask `EOS` while there are fewer than 2 holds, or no start, or no finish;
   - allow finish tokens only above a minimum height and start tokens only below one;
4. Run all candidates through the grade critic in a **single** batch and take the
   **expectation** `sum p_i * i`—not `argmax`, which is too noisy at around 38% confidence.
5. Rank by `w1 * |E[grade] - target| - w2 * log p(problem)`. Show the best candidate.

Runtime: roughly 15 generator forward steps through a 3M model, and one batched critic call
over at most 35 nodes. Both are well under a second with `onnxruntime-web`.

## Style: tried and dropped

Style was defined in code as seven geometric features over `(x, y, role)` — hand and foot
counts, move lengths, height span, foot-to-hand ratio — bucketed into the conditioning prefix
and mirrored in TypeScript with a parity test. Mechanically it worked. It failed on its own
terms: the four presets matched 12.1% (power), 7.2% (dyno), 1.3% (endurance) and 3.1%
(technical) of samples, which is not a useful control, and the feature was not wanted in the
product. Removed entirely — from the tokenizer, the training, the export, the app and the
tests.

Two things it left behind. The prefix is fixed-width so guidance can blank it without shifting
the holds, which is still how classifier-free guidance works here. And the corpus reference
values it was built on — 1.54 start, 4.52 hand, 3.35 foot, 1.06 finish holds per problem,
median 10 holds — are now the distribution-fidelity target in `tension-eval-generator`.

## Steps

### Step 1: tokenizer — done

[catalog.py](../src/tension_board_lab/catalog.py) for the position-to-hold lookup both the
tokenizer and the export need, and
[generator/tokenizer.py](../src/tension_board_lab/generator/tokenizer.py) with the
problem-to-token-sequence mapping, layout handling, and `(y, x)` ordering. Covered by
[tests/generator/test_tokenizer.py](../tests/generator/test_tokenizer.py).

### Step 2: train the generator — done

[generator/model.py](../src/tension_board_lab/generator/model.py) (3,116,038 parameters),
[generator/train.py](../src/tension_board_lab/generator/train.py) with `tension-train-generator`,
[generator/physical.py](../src/tension_board_lab/generator/physical.py) for the hold type behind
each token, [generator/constraints.py](../src/tension_board_lab/generator/constraints.py) for
the masking rules, [generator/sample.py](../src/tension_board_lab/generator/sample.py) with
`tension-sample`, and [generator/evaluate.py](../src/tension_board_lab/generator/evaluate.py)
with `tension-eval-generator`. Reports in [reports/generator/](../reports/generator/).

Trained on the mirror-augmented training split (36,169 problems, 59,435 after reflection),
early-stopped at epoch 24 with epoch 12 selected.

Measured over 240 samples per guidance scale, before and after removing styles, adding the
physical inputs, and augmenting:

| Guidance | Grade MAE | Feet per problem | Holds |
| --- | --- | --- | --- |
| 1.0 | 1.114 -> 1.067 | 2.89 -> **3.15** | 9.7 -> **10.0** |
| 1.5 | 0.892 -> 0.905 | 2.78 -> **3.10** | 9.5 -> **9.9** |
| 2.5 | 0.721 -> 0.791 | 2.60 -> **2.87** | 9.2 -> **9.6** |
| 3.5 | 0.780 -> 0.801 | 2.46 -> **2.81** | 9.1 -> **9.8** |

Corpus reference: 3.35 feet, 10.6 holds. Validity and novelty stayed at 100%, diversity at
about 0.96 - candidates are almost entirely distinct.

**How strong is this?** The 95% intervals overlap at every individual guidance level, so no
single comparison is decisive on its own. What carries it is that all four levels move the same
way by a similar amount, hold count moves with it, and the mechanism was verified
independently: 432 of 459 shared positions carry a different hold type on each layout, and 56
of 106 types are used for one role more than 70% of the time. A caveat on process: the old
checkpoint was overwritten by the retrain, so the baseline can no longer be re-measured at a
larger sample size.

**Guidance stays at 1.5.** It is where the realism gain is largest, and grade fidelity there is
unchanged from the baseline (0.892 -> 0.905, intervals overlapping). Raising it to 2.5 buys
about 0.11 V grades of fidelity and gives back roughly a quarter of a foot hold - the wrong
trade when the goal is problems that look set by a person.

**What is left.** Feet are still short of 3.35, and mirror augmentation did not remove the
overfitting: validation likelihood still turns upward around epoch 12, because a reflected copy
is highly correlated with its original and carries less than fresh data would. Grades still
regress toward the middle at the extremes.

One caveat on grade fidelity: the critic is the judge, so it measures agreement with the
critic, not ground truth. It is still meaningful, because the generator never saw the critic's
holdout configurations.

### Step 3: export — done

[export/onnx.py](../src/tension_board_lab/export/onnx.py) (`tension-export-onnx`) for both
models, and [export/artifacts.py](../src/tension_board_lab/export/artifacts.py)
(`tension-export-web`) for the board catalog, both vocabularies, `grade_labels`, the
calibration temperature of 1.67 (confirmed), the JS/Python parity fixtures, and the
`placement_id` table.

The int8 download is **3.10 MB for the critic and 3.77 MB for the generator, 6.87 MB
together**—a little above the 6 MB estimated here. Quantization costs little: over 200 real
problems the int8 critic agrees with fp32 on 98.0% of predicted grades, and
`examples/route.json` still reads V9.

Pitfalls, as they actually turned out:

- Set dynamic axes for both batch **and** the variable length, or the exporter bakes in the
  shapes it traced. Real, and the reason parity is checked at 2, 10, and 35 holds.
- The `mask` input is bool ([data.py:171](../src/tension_board_lab/data.py#L171)). Real: the
  export wrapper takes a float mask and thresholds it inside the graph.
- [`forward_batch`](../src/tension_board_lab/grade/train.py#L43) defines the exact six-input
  signature and its order. Real, and `critic.json` records it for the web side.
- **`aten::deg2rad` has no ONNX symbolic function** and blocks the export outright. Not
  anticipated. [grade/model.py](../src/tension_board_lab/grade/model.py) now multiplies by a
  constant, which is the same operation; the critic still returns `V9 / 0.3832`.
- **fp16 was a false alarm.** This plan predicted that `masked_fill` with
  `torch.finfo(dtype).min` would overflow to NaN in half precision. Measured on both trained
  models it does not: PyTorch's softmax subtracts the row maximum and saturates, giving logits
  within 0.01 of fp32 and identical argmaxes. int8 remains the right choice on size alone, so
  nothing changes in practice. onnxruntime-web's fp16 kernels are a separate implementation and
  would need their own check before anyone relies on this.

Coordinate inversion for rendering and Bluetooth, since normalization lives only in the SQL:
`raw_x = x * 128 - 64` and `raw_y = y * 136 + 4`. Emitted in `board.json` so the web side never
recomputes it.

### Step 4: React app — done

Vite, React, and TypeScript under [web/](../web/README.md). An SVG board renderer driven by
`board.json` with Aurora's role colors, and the featurization in `web/src/model/` as an exact
mirror of `collate_routes`.

Driven in a real browser: 498 holds render, selecting a start, two hands, a foot, and a finish
scores V12 at 51.1% confidence, editing the problem re-scores it, and switching 40° to 55°
moves it to V13—steeper reading harder, as it should.

Every deviation below produces a confident wrong grade rather than an error, so each one is
covered by a test against fixtures recorded from PyTorch:

- orientation is `[sin, cos]`, **in that order**
  ([data.py:195](../src/tension_board_lab/data.py#L195));
- coordinates are passed **raw**, with no rescaling;
- the angle is passed in **raw degrees**; normalization happens inside the model;
- unknown hold types silently become index 0
  ([data.py:193](../src/tension_board_lab/data.py#L193));
- role order is `start=0, hand=1, foot=2, finish=3`;
- holds keep the order they were given;
- confidence is `softmax(logits / 1.67)`.

`test/critic.test.ts` goes further than tensor comparison: it runs the exported graph through
onnxruntime and checks the resulting probabilities against PyTorch's, including that
`examples/route.json` still reads V9 at 0.3832 in JavaScript.

**Download budget, corrected.** The int8 graphs total 6.87 MB, but onnxruntime-web's wasm
runtime is another **6.4 MB gzipped** on top—this plan only ever counted the models. Loading
the generator lazily, when someone first asks for a problem, keeps the initial load to the
critic alone.

### Step 5: sampling UI — done

Inputs for grade, angle, and layout; a generate action; the result with its grade and
confidence; and an editor that toggles individual holds with live re-scoring by the critic.

The plan's *next suggestion* button was built and then dropped as clutter: twelve candidates
are still sampled and ranked, but only the best one reaches the board.

The sampling loop is a mirror of [sample.py](../src/tension_board_lab/generator/sample.py),
with the conditional and unconditional rows batched together so guidance costs one model call
per step rather than two.

Driven in a real browser: asking for a **V6 dyno** produced a problem the critic scores at an
expected **V6.02**, *next suggestion* moved to another candidate at V6.58, and editing a hold
re-scored it live. No console errors.

**The generator is loaded on first use, not at startup.** Nothing generator-shaped is fetched
until the button is clicked; someone who only wants to grade a problem they built never pays
the extra 3.8 MB.

Parity is checked piece by piece rather than end to end, because the two sides draw from
different random number generators: `tension-export-web` records token sequences, conditioning
prefixes, and constraint masks, and `test/tokenizer.test.ts` matches all of them exactly —
including the masks token for token. `test/sample.test.ts` then samples from the real graph and
asserts every problem is valid.

That paid for itself immediately: the fixtures caught an off-by-one in a conditioning slot that
would have mis-conditioned every generation without throwing anything.

One honest wrinkle in the UI: the headline grade is the critic's argmax while the ranking uses
its expectation, so a problem ranked as V6.02 can be labelled V5 at 27% confidence. Both
numbers are shown rather than one being hidden.

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

- `test_tokenizer.py`: lossless roundtrip; layout separation; deterministic ordering.
- `test_export.py`: ONNX parity—the same `RouteExample` objects through PyTorch and
  `onnxruntime`, logits within about 1e-4, checked at 2, 10, and 35 nodes.

**Generator evaluation** with `tension-eval-generator`, which bundles all of the below into one
command with bootstrap confidence intervals, a fixed seed, and CPU-only execution so runs are
comparable across machines:

- negative log-likelihood on the validation split;
- **grade fidelity:** sample N problems per target grade and measure the mean absolute error
  between the target and the critic's `E[grade]`, across several CFG scales to show the
  trade-off against variety;
- **novelty:** the fraction of samples whose `configuration_signature` already exists in the
  corpus (should be low), plus edit distance to the nearest corpus neighbor;
- **validity:** the fraction of samples violating hard constraints—must be 0;
- **distribution fidelity:** hold-count and role distributions against the corpus, targeting
  about 1.5 start, 4.5 hand, 3.4 foot, and 1.1 finish holds;
- **diversity:** mean pairwise Jaccard distance within a candidate set, so twelve variations of
  one problem cannot pass as twelve suggestions.

**JS/Python parity:** test the TypeScript featurization against reference fixtures exported
from Python—problem, expected tensors, expected logits. This is the most
error-prone point in the whole undertaking.

**Manual, and not optional:** look at 20 generated problems and judge for yourself whether they
look climbable and whether the grade is plausible. No automated test replaces that.

**End to end:** `npm run dev` in `web/`, generate a problem, change holds in the editor, and
watch whether the displayed grade moves plausibly.
