# Data pipeline, grade critic, and generator

This package holds three things for the **Tension Board 2 12x12**: the pipeline that turns a
local Aurora database into training data, the model that predicts how hard a boulder problem
is, and the model that invents new ones.

Give the **grade critic** the wall angle and the selected holds. It returns a V grade and its
confidence—for example, `V9` with `38%` confidence. The target is the community's consensus
grade at that particular angle, not an objective measure of difficulty.

Give the **generator** an angle, a grade, and optionally a style, and it produces problems the
critic then scores and ranks.

```mermaid
flowchart LR
    A["Wall angle<br/>e.g. 45°"] --> C
    B["Selected holds<br/>type · rotation · position · role"] --> C["Graph Transformer"]
    C --> D["Predicted grade"]
    C --> E["Confidence"]
```

## Modules

Shared modules sit at the top level of the package; each model has its own subpackage. An
`export/` subpackage will join them—see [`docs/roadmap.md`](../../docs/roadmap.md).

| Module | Purpose |
| --- | --- |
| `schema.py` | `RouteExample` and `HoldNode`, the canonical problem representation |
| `grades.py` | V-grade parsing and the ordinal grade axis |
| `aurora.py` | Import from the Aurora SQLite database into JSONL |
| `data.py` | Vocabulary, batching, and the split logic |
| `catalog.py` | The board's fixed placements; resolves a position to a hold |
| `style.py` | Rule-based style features, buckets, and presets |
| `grade/model.py` | The geometry-biased graph transformer and its loss |
| `grade/train.py` | Two-stage training, calibration, and evaluation |
| `grade/predict.py` | Single-problem inference |
| `generator/tokenizer.py` | Problem to token sequence for the generator |
| `generator/constraints.py` | Legal-token masks that keep sampled problems valid |
| `generator/model.py` | The decoder-only transformer and its loss |
| `generator/train.py` | Generator training with classifier-free guidance |
| `generator/sample.py` | Sampling, critic scoring, and candidate ranking |

## The data

The pipeline reads a local Aurora database and joins each problem's hold references against
[`../../configs/tb2_12x12_hold_catalog.csv`](../../configs/tb2_12x12_hold_catalog.csv), which
describes all 996 hold placements of the board—498 positions per layout, across the Mirror and
Spray layouts. Within one layout a coordinate identifies exactly one hold, so hold type and
rotation follow from the position.

Coordinates are normalized during import: `x = (raw_x + 64) / 128` and `y = (raw_y - 4) / 136`.
Everything downstream works in `[0, 1]`.

The consensus dataset contains 21,809 `(climb, angle)` examples with at least three
ascensionists from the Mirror and Spray layouts at 35°, 40°, 45°, 50°, and 55°. A typical
problem uses 10 holds; the largest uses 35.

### Splits and leakage

Examples are grouped by an exact signature of their model inputs before splitting, so
input-equivalent, renamed, and mirrored problems always land in the same split. Mirror-layout
problems are canonicalized by reflecting `x` and negating rotations, and the angle is
deliberately excluded from the signature—every angle of one hold set stays together. Grouping
is stratified by median grade and ordered by a hash, so splits are reproducible without a seed.

This yields 17,477 training, 2,158 validation, and 2,174 test examples, with zero duplicate and
zero cross-layout-equivalent groups across splits.

### Rebuilding the datasets

Place the local Aurora database at `data/raw/tension.sqlite3`, then run from the repository
root:

```bash
tension-import-aurora data/raw/tension.sqlite3 \
  --query configs/aurora_tb2_12x12.sql \
  --catalog configs/tb2_12x12_hold_catalog.csv \
  --output data/processed/tb2_12x12.jsonl

tension-import-aurora data/raw/tension.sqlite3 \
  --query configs/aurora_tb2_12x12_pretrain.sql \
  --catalog configs/tb2_12x12_hold_catalog.csv \
  --output data/processed/tb2_12x12_pretrain.jsonl
```

The two queries differ in one filter only: at least three ascensionists for the consensus set,
at least one for the pretraining set.

## The grade critic

### What it learns from

Each selected hold becomes one point in a graph. The model receives:

- the hold type;
- its rotation and X/Y position;
- its role: start, hand, foot, or finish;
- the wall angle as a continuous numeric input.

The hold-type token includes its material prefix—for example, `wood:SHLP` or `plastic:12`.
Material is not provided as an additional, separate feature.

It does **not** receive the layout name, climb name, placement ID, ascent count, or known grade
when making a prediction. Mirror and Spray are both training sources, but the model is not told
which layout an example came from.

### Architecture

The model embeds every hold and uses pairwise geometry—such as distance, direction, and
wall-relative movement—to connect it to every other hold. Six graph-transformer blocks with
eight attention heads learn which holds and possible moves matter. An ordinal-aware classifier
then produces probabilities for `V0` through `V14`; those probabilities are trained against
Aurora's unrounded community average. Validation temperature calibration turns them into the
reported confidence.

The canonical model has 2.85 million parameters.

### Training

Training happens in two stages. First, the model pretrains on 44,232 leakage-free examples:
18,045 with one grade, 8,710 with two, and 17,477 consensus training examples. One- and
two-ascent grades receive lower weights and wider soft targets. Every validation/test
configuration—and all its angles—is excluded from pretraining. The model is then fine-tuned
only on the 17,477 consensus training examples.

```bash
tension-train-grade data/processed/tb2_12x12.jsonl \
  --pretrain-dataset data/processed/tb2_12x12_pretrain.jsonl \
  --output checkpoints/grade/tb2_12x12.pt
```

### Results

On the untouched test split of 2,174 examples:

- **0.935 V grades** mean absolute error;
- **77.97%** of predictions within one V grade;
- **34.82%** exact-grade accuracy.

Accuracy tracks data density. At 35° the error is 0.849 V grades; at 55°, where only 389
examples exist, it is 1.063. Grades above V11 are similarly thin—212 examples at V12, 74 at
V13, and 3 at V14.

Grades are subjective, so confidence means model confidence—not certainty that every climber
will experience the problem the same way.

The versioned model specification is in
[`../../configs/tb2_12x12.json`](../../configs/tb2_12x12.json), and the full evaluation report
is in [`../../reports/grade/tb2_12x12.metrics.json`](../../reports/grade/tb2_12x12.metrics.json).

### Predicting

```bash
tension-predict checkpoints/grade/tb2_12x12.pt examples/route.json
```

```json
{"predicted_grade": "V9", "confidence": 0.3832}
```

See [`../../examples/route.json`](../../examples/route.json) for the input format.

## The generator

A decoder-only transformer over token sequences, 3,104,175 parameters. A sequence is

```
[BOS] layout angle grade style x7 | hold hold ... hold | [EOS]
```

Within one layout a position identifies exactly one hold, so a hold token carries only
`(position, role)`—type and orientation come from the catalog. That keeps the vocabulary at
2,223 tokens and the longest sequence at 47.

Holds are emitted bottom to top, in `(y, x)` order. Problems are sets, not sequences, so the
order is a convention rather than a climbing sequence; it puts starts early and finishes late.

### Conditioning

The prefix has fixed width, so classifier-free guidance can blank it without shifting the
holds. Training replaces the whole prefix with `UNCOND` 10% of the time, which gives sampling a
guidance scale that trades variety for fidelity to the request. Separately, each style feature
is dropped to its "any" slot 25% of the time, because a preset pins only two or three of the
seven features.

### Training

The generator trains on problems with at least one ascent, minus every configuration the critic
holds out—44,234 problems, split 36,169 / 4,382 / 3,683 by the same `split_examples` the critic
uses. Sharing that function is the point: a copy would eventually drift and quietly invalidate
every "the generator hits the target grade" number.

```bash
tension-train-generator data/processed/tb2_12x12_pretrain.jsonl \
  --consensus-dataset data/processed/tb2_12x12.jsonl \
  --output checkpoints/generator/tb2_12x12.pt
```

### Sampling

Hard rules are applied as logit masks during sampling rather than as a filter afterwards, so a
violation is unreachable instead of merely unlikely: positions off the layout are masked, a
used position retires all four of its roles, `EOS` stays blocked until the problem has a start
and a finish, and starts and finishes are held to loose height bounds.

```bash
tension-sample checkpoints/generator/tb2_12x12.pt \
  --critic checkpoints/grade/tb2_12x12.pt \
  --grade V5 --angle 40 --style power
```

Candidates go through the critic in one batch and are ranked by grade error, style distance,
and sequence likelihood. The critic's **expectation** is used, not its argmax, which is too
noisy to rank by at around 38% confidence.

### Results

Test negative log-likelihood is 3.481 per token, against 7.707 for a uniform choice over the
vocabulary. Over 480 samples: **100% valid**, **100% novel**—no sampled problem reproduced any
of the 35,524 configurations in the corpus. Grade error against the critic falls from 1.027 to
0.752 V grades as guidance rises from 1.0 to 2.5, and conditioning on the dyno preset lifts its
match rate from 6.2% to 78.1%.

Feet are under-generated—2.7 per problem against the corpus 3.35—and problems run short, at
most 21 holds against the corpus 35. Technical steering is weak at 3.1% for exactly that
reason: the preset needs a high foot-to-hand ratio. Grade fidelity is agreement with the
critic, not with ground truth. The full report is in
[`../../reports/generator/tb2_12x12.metrics.json`](../../reports/generator/tb2_12x12.metrics.json).
