/**
 * Loading the exported artifacts, or reporting that they are not there.
 *
 * Everything under `public/data` and `public/models` is produced by `tension-export-web` and
 * `tension-export-onnx` from a trained checkpoint, so a fresh clone does not have it. The
 * parity suites depend on it entirely; they skip rather than fail, which keeps `npm test`
 * meaningful before anyone has run the Python side.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const DATA = new URL("../public/data/", import.meta.url);
const MODELS = new URL("../public/models/", import.meta.url);

/** Parsed artifact, or `undefined` when the export has not been run. */
export function readArtifact<T>(name: string): T | undefined {
  const url = new URL(name, DATA);
  if (!existsSync(url)) return undefined;
  return JSON.parse(readFileSync(url, "utf8")) as T;
}

/** Absolute path to an exported ONNX graph, or `undefined` when it is missing. */
export function modelPath(name: string): string | undefined {
  const path = fileURLToPath(new URL(name, MODELS));
  return existsSync(path) ? path : undefined;
}

export const MISSING_ARTIFACTS =
  "exported artifacts missing - run tension-export-web and tension-export-onnx";
