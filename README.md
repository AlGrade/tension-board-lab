# Tension Grade Predictor

A board-aware graph transformer that predicts an angle-specific V grade for
Tension Board 2 Mirror, 12x12, at 35°, 40°, 45°, 50°, or 55°.

The prediction contract is intentionally small:

```json
{"predicted_grade":"V7","confidence":0.6812}
```

Confidence is the winning class probability after validation-set temperature
calibration. It should not be interpreted as objective certainty about a subjective
climbing grade.

## Architecture

Each selected hold is a graph node with three kinds of information:

- fixed placement identity;
- route role (`start`, `hand`, `foot`, or `finish`);
- normalized board coordinate.

The wall angle uses the original learned embedding for the five supported settings.
No weak proxy fields are presented as hold characteristics; Aurora does not expose
reliable hold-shape, orientation, or hold-difficulty metadata.

Six transformer blocks perform attention between every selected hold. Each attention
head receives a learned geometric bias built from horizontal distance, vertical
distance, absolute distance, and direction. Attention pooling produces one route
representation, and an ordinal-aware classification head predicts the V grade.

The default model is about 192 units wide with eight attention heads and six layers.
This is deliberately substantial for the dataset, but still practical to train on
Apple Silicon via MPS.

## Setup

Python 3.10 or newer is required. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

## Obtain the public board database

[BoardLib](https://github.com/lemeryfertitta/BoardLib) can download the public
database bundled with the Tension app without signing in:

```bash
boardlib database tension data/raw/tension.sqlite3
```

To synchronize the latest shared community data, add your Tension username. BoardLib
will securely prompt for the password; do not put the password in a command or file.

```bash
boardlib database tension data/raw/tension.sqlite3 --username YOUR_USERNAME
```

The database and derived dataset are gitignored. Check Aurora/Tension's terms before
redistributing either one.

## Inspect and import

The checked-in SQL mapping is pinned to the audited Aurora schema and selects:

- layout 10: Tension Board 2 Mirror;
- product size 6 / sets 12 and 13: 12x12 wood and plastic holds;
- angles 35°, 40°, 45°, 50°, and 55°;
- listed, single-frame boulders;
- community labels backed by at least three ascensionists.

Inspect a future database before trusting the mapping:

```bash
tension-inspect-db data/raw/tension.sqlite3 --output work/schema.json
```

Then create the canonical dataset:

```bash
tension-import-aurora data/raw/tension.sqlite3 \
  --query configs/aurora_tb2_mirror_12x12.sql \
  --output data/processed/tb2_mirror_12x12.jsonl
```

Each JSONL row is one `(climb, angle)` training example. Ascensionist count is used
only to weight label reliability. It is never passed to the model as an input.

## Train

```bash
tension-train data/processed/tb2_mirror_12x12.jsonl \
  --output checkpoints/tb2_mirror_12x12.pt
```

Training uses AdamW, cosine decay, gradient clipping, early stopping, an ordinal
distance penalty, and deterministic grade-stratified group splitting. The group key
is the actual hold-role configuration rather than the climb UUID. Consequently, all
angles and exact copies saved under different names remain in one split.

Useful overrides:

```bash
tension-train data/processed/tb2_mirror_12x12.jsonl \
  --batch-size 64 --epochs 120 --width 256 --heads 8 --layers 8
```

## Predict

A route JSON contains the angle and selected placements. It intentionally has no
grade field. See `examples/route.json` for the canonical shape.

```bash
tension-predict checkpoints/tb2_mirror_12x12.pt examples/route.json
```

Only the predicted V grade and calibrated confidence are printed.

## Test

```bash
pytest
ruff check .
```

The first meaningful success criterion is held-out mean absolute error below one
V-grade step, followed by confidence calibration and a separate benchmark-climb
evaluation.

## Current checkpoint

`checkpoints/tb2_mirror_12x12.pt` was trained on 13,043 public Mirror examples with
native Aurora V-grade labels. Early stopping ended the 40-epoch run at epoch 29 and
selected epoch 17. Its duplicate-safe, grade-stratified held-out results are:

- mean absolute error: 1.0502 V-grade steps;
- within one V-grade step: 73.75%;
- exact-grade accuracy: 31.43%;
- confidence temperature: 1.41.

See `reports/tb2_mirror_12x12.metrics.json` for the exact split and evaluation data.
