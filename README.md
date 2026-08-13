# Tension Board Lab

[![CI](https://github.com/AlGrade/tension-board-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/AlGrade/tension-board-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Two small models for boulder problems on a **Tension Board 2 12x12**, and a web app that runs
both of them in the browser with no backend. A Tension Board is a standardized indoor climbing
wall: the holds stay in fixed positions, and a problem is a choice of which ones you may use.

One model **grades** a problem. The other **invents** problems at a grade you ask for, using the
first as an independent critic. You can then edit what it produced and watch the grade move.

![A generated problem on a photo of the board, with its predicted grade](docs/screenshot.jpg)

## What you can run from a clone

Honestly: not the models. They are trained on a local Aurora board database that is not in this
repository — it holds account-derived data — and neither the database, the derived datasets, nor
the trained checkpoints are committed.

| From a fresh clone | Needs the board database |
| --- | --- |
| Read the code and the design notes | Import the datasets |
| `pytest` — 81 tests, 18 skip | Train either model |
| `npm test`, `npm run build` in `web/` | Export ONNX and run the web app |

If you do have the database, [`src/tension_board_lab/`](src/tension_board_lab/README.md)
documents the whole path from import to trained checkpoint to exported artifacts.

## The parts

| Part | Documentation |
| --- | --- |
| Data pipeline and grade critic | [`src/tension_board_lab/`](src/tension_board_lab/README.md) |
| Boulder problem generator | [`src/tension_board_lab/`](src/tension_board_lab/README.md) |
| Web application | [`web/`](web/README.md) |

The **grade critic** takes a wall angle and a set of selected holds and returns a V grade with a
confidence — for example `V9` at `38%`. On its untouched test split of 2,174 problems it reaches
a mean absolute error of 0.935 V grades, with 77.97% of predictions within one V grade.

The **generator** is a 3.1M-parameter decoder-only transformer over `(position, role)` tokens.
Hard rules are applied as logit masks while sampling, so an invalid problem is unreachable
rather than unlikely: every evaluated sample was valid, and none reproduced a problem from the
corpus.

The **web application** ships both models as int8 ONNX and is tested for exact numerical parity
with the Python featurization — a drift there would show wrong grades rather than fail, so the
frontend is checked against tensors recorded from PyTorch.

## Setup

Python 3.10 or newer.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
```

With a trained checkpoint, grade a problem:

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
cd web && npm test && npx tsc --noEmit
```

The Python commands expect the repository root as the working directory.

## Repository layout

```
configs/               board layout catalog, Aurora SQL queries, model specification
data/                  local databases and generated datasets (git-ignored)
docs/                  design notes
checkpoints/<model>/   trained models (git-ignored)
reports/<model>/       evaluation reports
examples/              example input for tension-predict
src/tension_board_lab/ the Python package
tests/                 test suite, mirroring the package
web/                   the browser application
web/public/            exported ONNX graphs and JSON artifacts (git-ignored)
```

Inside the package, the modules every model needs — the problem schema, the grade axis, the
Aurora import, the board catalog, and the batching and split logic — sit at the top level. Each
model gets its own subpackage below them: [`grade/`](src/tension_board_lab/grade/),
[`generator/`](src/tension_board_lab/generator/), and
[`export/`](src/tension_board_lab/export/). `checkpoints/` and `reports/` are split by model the
same way, so two models never contend for one filename.

Board databases, generated datasets, trained checkpoints, and the exported web artifacts are
kept out of Git. The databases may contain account-derived or licensed data, and everything else
on that list is reproducible from them with a documented command.

## Design notes

[`docs/board-images.md`](docs/board-images.md) covers how the board photographs were calibrated
so that normalized coordinates land on the right holds — the mapping is fitted against the hold
lattice rather than eyeballed.

## License

MIT — see [LICENSE](LICENSE).
