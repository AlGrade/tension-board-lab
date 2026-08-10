# Tension Grade Predictor

A board-aware graph transformer that predicts an angle-specific Font grade for
Tension Board 2 Spray, 12x12, at 35°, 40°, 45°, 50°, or 55°.

The prediction contract is intentionally small:

```json
{"predicted_grade":"7A+","confidence":0.6812}
```

Confidence is the winning class probability after validation-set temperature
calibration. It should not be interpreted as objective certainty about a subjective
climbing grade.

## Architecture

Each selected hold is a graph node with four kinds of information:

- fixed placement identity;
- route role (`start`, `hand`, `foot`, or `finish`);
- normalized board coordinate;
- wall angle.

Six transformer blocks perform attention between every selected hold. Each attention
head receives a learned geometric bias built from horizontal distance, vertical
distance, absolute distance, and direction. Attention pooling produces one route
representation, and an ordinal-aware classification head predicts the Font grade.

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

- layout 11: Tension Board 2 Spray;
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
  --query configs/aurora_tb2_spray_12x12.sql \
  --output data/processed/tb2_spray_12x12.jsonl
```

Each JSONL row is one `(climb, angle)` training example. Ascensionist count is used
only to weight label reliability. It is never passed to the model as an input.

## Train

```bash
tension-train data/processed/tb2_spray_12x12.jsonl \
  --output checkpoints/tb2_spray_12x12.pt
```

Training uses AdamW, cosine decay, gradient clipping, early stopping, an ordinal
distance penalty, and deterministic group splitting. All angles belonging to the
same climb stay in one split, preventing the model from seeing a route at 40° during
training and being tested on the identical route at 45°.

Useful overrides:

```bash
tension-train data/processed/tb2_spray_12x12.jsonl \
  --batch-size 64 --epochs 120 --width 256 --heads 8 --layers 8
```

## Predict

A route JSON contains the angle and selected placements. It intentionally has no
grade field. See `examples/route.json` for the canonical shape.

```bash
tension-predict checkpoints/tb2_spray_12x12.pt examples/route.json
```

Only the predicted Font grade and calibrated confidence are printed.

## Test

```bash
pytest
ruff check .
```

The first meaningful success criterion is held-out mean absolute error below one
Font grade step, followed by confidence calibration and a separate benchmark-climb
evaluation.

## Current checkpoint

`checkpoints/tb2_spray_12x12.pt` was trained on the initial 8,766-example public
dataset. Early stopping selected epoch 21 from a 40-epoch run. Its strictly held-out
test results are:

- mean absolute error: 1.6421 Font-grade steps;
- within one Font-grade step: 52.59%;
- exact-grade accuracy: 20.94%;
- confidence temperature: 1.28.

This is an honest first checkpoint, not a production benchmark. The most promising
next gains are adding hold-shape/material/orientation metadata, increasing the sparse
35° and 55° labels, and comparing the transformer against an engineered-feature tree
baseline. Making the network larger before those changes is unlikely to help.
