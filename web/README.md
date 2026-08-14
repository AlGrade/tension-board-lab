# Web application

A React app that runs both models in the browser. No backend: the ONNX graphs and their
supporting JSON are static files.

Ask for a grade and an angle and it samples twelve candidates, scores them with the critic, and
puts the best one on the board. Or build a problem by clicking holds and it grades that. Either
way you can edit what is on the board and watch the grade move.

The generator is loaded on first use rather than at startup: it is a separate 3.8 MB, and
someone who only wants to grade a problem they built should not pay for it.

## Running it

The models the app runs are committed under `public/`, so `npm run dev` works from a clone. To
rebuild them after retraining, or to get the extra artifacts the parity tests need, run this
from the repository root:

```bash
python -m pip install -e ".[export]"
tension-export-onnx
tension-export-web
```

Then, in this directory:

```bash
nvm use
npm install
npm run dev
```

Node 24 — `.nvmrc` pins the exact version this was built against, and `engines.node` in
`package.json` keeps it to the major. npm only warns on a mismatch rather than refusing, so
`nvm use` is the step that matters locally.

`npm test` runs the parity suite, `npm run typecheck` the types, and `npm run build` a
production bundle.

## Deploying

Vercel, from this directory: the project's root directory has to be set to `web`, because the
repository root is a Python package and carries no `package.json`. Everything else the Vite
preset infers correctly. Pushes to `main` deploy to production, other branches to previews.

The build runs on whatever `engines.node` asks for, which is why that is a `24.x` range and not
the exact version in `.nvmrc`: Vercel picks the minor within the major, and pinning past it
would eventually fail the build for a patch release nobody chose.

`vercel.json` sets headers, and JSON takes no comments, so the reasons are here:

- **`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`** buy `SharedArrayBuffer`,
  without which onnxruntime-web silently falls back to a single thread. It loads the threaded
  wasm build either way, so the cost of omitting them is invisible — just a slower grade. Safe
  to require here because every subresource the page loads is same-origin; adding a font or a
  script from a CDN would break the page until that host sends `Cross-Origin-Resource-Policy`.
- **`/assets/` is immutable for a year.** Vite content-hashes those filenames, which includes
  the 26 MB wasm runtime, so this is the cache that actually matters.
- **`/models/` and `/data/` get an hour**, not a year: `tension-export-web` overwrites them
  under the same names, so a long cache would pin a stale graph indefinitely. An hour also
  bounds something worse than staleness. A cached `grade.int8.onnx` paired with a freshly
  re-exported `critic.json` is exactly the drift the parity suite exists to catch, and it fails
  the same silent way — a confident wrong grade. Keep the two lifetimes equal. Content-hashing
  the artifact names, as Vite does, would remove the window altogether.
- **`/board/` gets a day.** Wall photographs and a calibration fitted against the hold
  lattice; nothing pairs with a model.

## Layout

```
src/types.ts        shapes shared with the exported artifacts
src/model/          featurization, tokenizer, and the onnxruntime-web sessions
src/generate/       constraint masks and the sampling loop
src/board/          SVG renderer
test/               parity against fixtures recorded from Python
```

## Parity is the whole game

The frontend has to reproduce `collate_routes` exactly. Where it drifts, nothing throws—the app
just shows a confident wrong grade. So `tension-export-web` records real problems together with
the tensors and logits PyTorch produced for them, and the test suite checks this side against
them:

- [`test/featurize.test.ts`](test/featurize.test.ts) — every tensor, for every fixture;
- [`test/critic.test.ts`](test/critic.test.ts) — the whole chain, featurizer through the ONNX
  graph, against PyTorch's logits;
- [`test/tokenizer.test.ts`](test/tokenizer.test.ts) — token sequences, conditioning prefixes,
  and constraint masks, the last of these token for token;
- [`test/sample.test.ts`](test/sample.test.ts) — samples from the real generator graph and
  asserts every problem satisfies every rule.

The sampling loop itself cannot be compared step for step, because the two sides draw from
different random number generators. So the pieces it is built from are compared instead. That
has already caught one silent drift: an off-by-one in a conditioning slot that would have
mis-conditioned every generation without throwing anything.

The generator graph takes a second input beside the tokens: the layout index. A hold token is
only `(position, role)`, and the same position carries a different hold on mirror than on
spray, so the model is told which wall it is working on. Guidance blanks the request, never the
layout.

The details that actually cause silent drift, all covered above:

- orientation is `[sin, cos]`, in that order;
- coordinates are passed raw in `[0, 1]`, never rescaled;
- the angle is passed in raw degrees and normalized inside the model;
- an unknown hold type becomes index 0 rather than an error;
- role order is `start=0, hand=1, foot=2, finish=3`;
- holds keep the order they were given;
- confidence is `softmax(logits / 1.67)`.

## Download size

The int8 graphs are 3.10 MB for the critic and 3.77 MB for the generator. onnxruntime-web's
wasm runtime is a separate 6.4 MB gzipped on top of that, which is easy to forget when counting
only the models. Loading the generator lazily, only when someone asks for a problem, keeps
the first paint cheap.

## Rendering

The board is a photograph of the real wall, with a coloured ring per selected hold. Rings
rather than filled dots, because a dot hides the hold it is marking — which would defeat the
point of using a photo. Unselected positions are invisible click targets.

Positions arrive normalized to `[0, 1]` with y measured upwards from the bottom of the wall;
SVG measures y downwards, so `BoardView` flips it once and nothing else does. Where those
coordinates land inside each photo comes from `public/board/calibration.json`, which was fitted
against the hold lattice — see [`docs/board-images.md`](../docs/board-images.md). Role colors
come from `board.json` and match Aurora's, so the board looks like the app climbers already
use. `board.json` also carries the raw coordinates, which Web Bluetooth needs later.
