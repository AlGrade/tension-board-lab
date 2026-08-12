# Web application

A React app that runs both models in the browser. No backend: the ONNX graphs and their
supporting JSON are static files.

Today it renders the board, lets you build a problem by clicking holds, and scores it live with
the grade critic. Generation is the next step—see [`docs/roadmap.md`](../docs/roadmap.md).

## Running it

The artifacts under `public/` are produced by the Python package and are git-ignored, so build
them first, from the repository root:

```bash
python -m pip install -e ".[export]"
tension-export-onnx
tension-export-web
```

Then, in this directory:

```bash
npm install
npm run dev
```

`npm test` runs the parity suite, `npm run typecheck` the types, and `npm run build` a
production bundle.

## Layout

```
src/style.ts        mirror of style.py
src/types.ts        shapes shared with the exported artifacts
src/model/          featurization and the onnxruntime-web sessions
src/board/          SVG renderer
test/               parity against fixtures recorded from PyTorch
```

## Parity is the whole game

The frontend has to reproduce `collate_routes` exactly. Where it drifts, nothing throws—the app
just shows a confident wrong grade. So `tension-export-web` records real problems together with
the tensors and logits PyTorch produced for them, and the test suite checks this side against
them:

- [`test/featurize.test.ts`](test/featurize.test.ts) — every tensor, for every fixture;
- [`test/critic.test.ts`](test/critic.test.ts) — the whole chain, featurizer through the ONNX
  graph, against PyTorch's logits;
- [`test/style.test.ts`](test/style.test.ts) — features, buckets, and preset distances against
  what `style.py` computed.

The details that actually cause silent drift, all covered above:

- orientation is `[sin, cos]`, in that order;
- coordinates are passed raw in `[0, 1]`, never rescaled;
- the angle is passed in raw degrees and normalized inside the model;
- an unknown hold type becomes index 0 rather than an error;
- role order is `start=0, hand=1, foot=2, finish=3`;
- holds keep the order they were given;
- confidence is `softmax(logits / 1.67)`.

## Download size

The int8 graphs are 3.10 MB for the critic and 3.69 MB for the generator. onnxruntime-web's
wasm runtime is a separate 6.4 MB gzipped on top of that, which the roadmap's original estimate
did not account for. Loading the generator lazily, only when someone asks for a problem, keeps
the first paint cheap.

## Rendering

Positions arrive normalized to `[0, 1]` with y measured upwards from the bottom of the wall;
SVG measures y downwards, so `BoardView` flips it once and nothing else does. Role colors come
from `board.json` and match Aurora's, so the board looks like the app climbers already use.
`board.json` also carries the raw coordinates, which Web Bluetooth needs later.
