# Tension Grade Predictor

A geometry-aware graph transformer that predicts angle-specific V grades for the
Tension Board 2 12x12 using one deliberately small input contract.

For a non-technical German explanation, see
[`README_EINFACH_ERKLAERT.md`](README_EINFACH_ERKLAERT.md).

## Input and output

The model receives the wall angle and, for every selected hold:

- `hold_type`: one of 106 physical hold types;
- `orientation_degrees`: its rotation on the board;
- `x` and `y`: normalized board coordinates;
- `role`: `start`, `hand`, `foot`, or `finish`.

It does **not** receive a layout, separate material, left/right variant, placement
ID, climb name, ascent count, or grade. Mirror and Spray are only source metadata in
the training dataset and never become model tensors.

The output is intentionally limited to the most likely grade and its calibrated
class probability:

```json
{"predicted_grade":"V8","confidence":0.5128}
```

Confidence is model confidence, not objective certainty about a subjective grade.

## Architecture

The canonical model has 2,851,263 parameters, width 192, eight attention heads, and
six graph-transformer blocks. Hold type, orientation, coordinates, and route role
form one node per selected hold. The continuous angle encoding contains the degree
value plus sine/cosine terms.

Attention between every pair of holds receives a learned geometric bias containing
horizontal and vertical separation, Euclidean distance, direction, wall-relative
vertical gain, and overhang depth. An ordinal-aware classification head returns a
distribution over V0 through V14.

## Setup

Python 3.10 or newer is required:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

## Dataset

The current local database snapshot is `data/raw/tension.sqlite3`. Databases and
generated datasets are gitignored. The audited importer selects:

- Tension Board 2 Mirror and Spray;
- board size 12x12 with wood and plastic sets;
- 35°, 40°, 45°, 50°, and 55°;
- listed, single-frame boulders;
- angle-specific community grades with at least three ascensionists.

The official install-guide catalog maps Aurora placements to hold type, orientation,
and coordinates. Material and variant remain source-audit columns in that catalog,
but are discarded when the canonical dataset is written.

Regenerate the only processed dataset:

```bash
tension-import-aurora data/raw/tension.sqlite3 \
  --query configs/aurora_tb2_12x12.sql \
  --catalog configs/tb2_12x12_hold_catalog.csv \
  --output data/processed/tb2_12x12.jsonl
```

The result contains 21,809 `(climb, angle)` examples: 13,043 from Mirror and 8,766
from Spray. `source_layout`, ascent count, and the target grade are training metadata,
not prediction inputs.

## Train

```bash
tension-train data/processed/tb2_12x12.jsonl \
  --output checkpoints/tb2_12x12.pt
```

Training uses AdamW, cosine learning-rate decay, gradient clipping, early stopping,
an ordinal distance penalty, and validation-set temperature calibration.

The deterministic split groups the exact inputs seen by the model. All angles,
renamed copies, and input-equivalent Mirror/Spray configurations remain together.
Mirror reflections are canonicalized too. The current split contains 17,477 training,
2,158 validation, and 2,174 test examples with zero shared configuration groups.

## Predict

See `examples/route.json` for the complete minimal input shape.

```bash
tension-predict checkpoints/tb2_12x12.pt examples/route.json
```

## Verify

```bash
pytest
ruff check .
```

## Current checkpoint

The canonical checkpoint selected epoch 12 and stopped at epoch 24. On the untouched
test split it reaches:

- mean absolute error: 1.0281 V-grade steps;
- within one V-grade step: 73.64%;
- exact-grade accuracy: 32.84%;
- Mirror MAE: 0.9977 across 1,314 examples;
- Spray MAE: 1.0744 across 860 examples;
- confidence temperature: 1.37.

The checkpoint is a reproducible, gitignored binary artifact. Exact model specs are
in `configs/tb2_12x12.json`; the complete evaluation is in
`reports/tb2_12x12.metrics.json`.
