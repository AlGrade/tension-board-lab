# Tension Board Lab

Models and tools for boulder problems on a **Tension Board 2 12x12**. A Tension Board is a
standardized indoor climbing wall: the holds stay in fixed positions, but climbers choose
which ones belong to a problem.

## Parts

| Part | Status | Documentation |
| --- | --- | --- |
| Data pipeline and grade predictor | Working | [`src/tension_board_lab/`](src/tension_board_lab/README.md) |
| Boulder problem generator | Working | [`src/tension_board_lab/`](src/tension_board_lab/README.md) |
| Web application | In progress | [`web/`](web/README.md) |

The **grade predictor** takes a wall angle and a set of selected holds and returns a V grade
with a confidence—for example, `V9` with `38%`. On its untouched test split of 2,174 examples
it reaches a mean absolute error of 0.935 V grades, with 77.97% of predictions within one V
grade. See [`src/tension_board_lab/`](src/tension_board_lab/README.md) for the architecture, the
training procedure, and how to rebuild the datasets.

The **generator** proposes new problems for a requested grade, angle, and style, using the grade
predictor as an independent critic. Every sampled problem is valid by construction, and none of
480 evaluated samples reproduced a problem from the corpus.

The **web application** runs both models in the browser with no backend. It renders the board,
scores a problem as you edit it, and is tested for exact parity with the Python featurization.
Generation moves into the UI next; see [`docs/roadmap.md`](docs/roadmap.md).

## Setup

Python 3.10 or newer is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

Predict the grade of a problem:

```bash
tension-predict checkpoints/grade/tb2_12x12.pt examples/route.json
```

```json
{"predicted_grade": "V9", "confidence": 0.3832}
```

Copy [`examples/route.json`](examples/route.json) and replace its angle and holds to make your
own prediction. Coordinates are normalized to the board; rotations are degrees.

## Checks

```bash
pytest
ruff check .
```

Both commands expect the repository root as the working directory.

## Repository layout

```
configs/               board layout catalog, Aurora SQL queries, model specification
data/                  local databases and generated datasets (git-ignored)
docs/                  design notes and the roadmap
checkpoints/<model>/   trained models (git-ignored)
reports/<model>/       evaluation reports
examples/              example input for tension-predict
src/tension_board_lab/ the Python package
tests/                 test suite, mirroring the package
web/public/            exported ONNX graphs and JSON artifacts (git-ignored)
```

Inside the package, the modules every model needs—the problem schema, the grade axis, the
Aurora import, and the batching and split logic—sit at the top level. Each model gets its own
subpackage below them: [`grade/`](src/tension_board_lab/grade/) today, `generator/` and
`export/` when they arrive. `checkpoints/` and `reports/` are split by model the same way, so
two models never contend for one filename.

Board databases, generated datasets, trained checkpoints, and the exported web artifacts are
intentionally kept out of Git—the databases may contain account-derived or licensed data, and
everything else on that list is reproducible from them with a documented command.
