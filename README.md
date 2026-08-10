# Tension Grade Predictor

A shape-aware graph transformer that predicts an angle-specific V grade for
Tension Board 2 Mirror and Spray, 12x12, at 35°, 40°, 45°, 50°, or 55°.

The prediction contract is intentionally small:

```json
{"predicted_grade":"V7","confidence":0.6812}
```

Confidence is the winning class probability after validation-set temperature
calibration. It is model confidence, not objective certainty about a subjective
climbing grade.

## What the model sees

Each selected hold is a graph node containing:

- hold family and left/right variant;
- wood or plastic material;
- physical orientation as continuous sine/cosine values;
- normalized 12x12 board coordinates;
- its route role (`start`, `hand`, `foot`, or `finish`).

Placement IDs are retained only for traceability and are **not** passed to the
network. The hold catalog is derived from the official 2024 Mirror and Spray install
guides and maps each Aurora placement to its labeled hold, material, coordinate, and
16-step compass orientation.

The wall angle is also a continuous numeric input. Its encoding contains the degree
value plus sine/cosine terms; it is not a categorical angle ID. Six transformer
blocks perform attention between all selected holds. Their learned graph bias uses
horizontal and vertical separation, Euclidean distance, direction, wall-relative
vertical gain, and overhang depth. Attention pooling and an ordinal-aware classifier
produce the grade distribution.

Mirror and Spray share the shape embeddings and transformer. A small layout
embedding lets the network account for systematic layout differences without
memorizing placement IDs.

## Setup

Python 3.10 or newer is required. From the project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

## Database and import

The project currently uses the existing local Aurora/Tension database snapshot at
`data/raw/tension.sqlite3`. The database and derived datasets are gitignored. To
replace the snapshot later, [BoardLib](https://github.com/lemeryfertitta/BoardLib)
can download the public database bundled with the app:

```bash
boardlib database tension data/raw/tension.sqlite3
```

The audited SQL selects:

- layout 10 (Mirror) and layout 11 (Spray);
- product size 6 / sets 12 and 13 (12x12 wood and plastic holds);
- angles 35°, 40°, 45°, 50°, and 55°;
- listed, single-frame boulders;
- community labels backed by at least three ascensionists.

Inspect an unfamiliar database before trusting the mapping:

```bash
tension-inspect-db data/raw/tension.sqlite3 --output work/schema.json
```

Create the canonical shape-enriched dataset:

```bash
tension-import-aurora data/raw/tension.sqlite3 \
  --query configs/aurora_tb2_12x12_shapes.sql \
  --catalog configs/tb2_12x12_hold_catalog.csv \
  --output data/processed/tb2_12x12_shapes.jsonl
```

Each JSONL row is one `(climb, angle)` example. Ascension count weights label
reliability during training but is never passed to the model as an input.

## Train

```bash
tension-train data/processed/tb2_12x12_shapes.jsonl \
  --output checkpoints/tb2_12x12_shapes.pt
```

Training uses AdamW, cosine decay, gradient clipping, early stopping, an ordinal
distance penalty, and deterministic grade-stratified group splitting. The group key
is the hold/role/shape/orientation configuration, not the climb UUID. All angles and
renamed exact copies remain in one split. Mirror-reflected copies are canonicalized
too, including left/right hold variants and mirrored orientations.

Useful overrides:

```bash
tension-train data/processed/tb2_12x12_shapes.jsonl \
  --batch-size 64 --epochs 120 --width 256 --heads 8 --layers 8
```

## Predict

A route JSON contains its layout, angle, and enriched selected holds, but no grade.
See `examples/route.json` for the canonical format.

```bash
tension-predict checkpoints/tb2_12x12_shapes.pt examples/route.json
```

Only the predicted V grade and calibrated confidence are printed.

## Test

```bash
pytest
ruff check .
```

## Current checkpoint

`checkpoints/tb2_12x12_shapes.pt` was trained on 21,809 Mirror and Spray examples
from the existing database snapshot. Early stopping ended at epoch 25 and selected
epoch 13. Its untouched, duplicate-safe test results are:

- mean absolute error: 1.0455 V-grade steps;
- within one V-grade step: 73.73%;
- exact-grade accuracy: 30.73%;
- Mirror MAE: 1.0606 across 1,271 test examples;
- Spray MAE: 1.0248 across 929 test examples;
- confidence temperature: 1.30.

The checkpoint and generated dataset stay gitignored because they are reproducible
binary artifacts. See `reports/tb2_12x12_shapes.metrics.json` for exact split and
angle-by-angle metrics.
