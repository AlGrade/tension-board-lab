/**
 * Mirror of `style.py`. The Python module is the source of truth; this is the copy the
 * browser runs, and `test/style.test.ts` checks it against values Python computed.
 *
 * The thresholds themselves are not hard-coded here—they arrive in `style.json`, so changing
 * a bucket edge in Python cannot leave the frontend stale.
 */

import type { Hold, HoldRole, StyleArtifact } from "./types";

export const FEATURE_NAMES = [
  "handCount",
  "footCount",
  "meanMoveLength",
  "maxMoveLength",
  "moveLengthVariance",
  "heightSpan",
  "footToHandRatio",
] as const;

export type FeatureName = (typeof FEATURE_NAMES)[number];

/** The exported artifact uses snake_case; keep one mapping rather than string-munging. */
export const PYTHON_FEATURE_NAMES: Record<FeatureName, string> = {
  handCount: "hand_count",
  footCount: "foot_count",
  meanMoveLength: "mean_move_length",
  maxMoveLength: "max_move_length",
  moveLengthVariance: "move_length_variance",
  heightSpan: "height_span",
  footToHandRatio: "foot_to_hand_ratio",
};

export type StyleFeatures = Record<FeatureName, number>;

const HAND_PATH_ROLES: ReadonlySet<HoldRole> = new Set<HoldRole>(["start", "hand", "finish"]);

/**
 * The holds the hands use, bottom to top.
 *
 * Problems are sets, so this order is a convention, not a climbing sequence. It matches the
 * order the generator emits holds in.
 */
export function handPath(holds: Hold[]): Hold[] {
  return holds
    .filter((hold) => HAND_PATH_ROLES.has(hold.role))
    .sort((a, b) => a.y - b.y || a.x - b.x);
}

export function moveLengths(holds: Hold[]): number[] {
  const path = handPath(holds);
  const lengths: number[] = [];
  for (let index = 1; index < path.length; index += 1) {
    const previous = path[index - 1];
    const current = path[index];
    lengths.push(Math.hypot(current.x - previous.x, current.y - previous.y));
  }
  return lengths;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** Population variance, matching `statistics.pvariance` — divided by n, not n - 1. */
function populationVariance(values: number[]): number {
  const average = mean(values);
  return mean(values.map((value) => (value - average) ** 2));
}

export function computeStyleFeatures(holds: Hold[]): StyleFeatures {
  if (holds.length === 0) {
    throw new Error("Cannot compute style features for a problem without holds");
  }
  const path = handPath(holds);
  if (path.length === 0) {
    throw new Error("A problem needs at least one hand, start, or finish hold");
  }
  const moves = moveLengths(holds);
  const footCount = holds.filter((hold) => hold.role === "foot").length;
  const ys = holds.map((hold) => hold.y);

  return {
    handCount: holds.filter((hold) => hold.role === "hand").length,
    footCount,
    meanMoveLength: moves.length > 0 ? mean(moves) : 0,
    maxMoveLength: moves.length > 0 ? Math.max(...moves) : 0,
    moveLengthVariance: moves.length > 1 ? populationVariance(moves) : 0,
    heightSpan: Math.max(...ys) - Math.min(...ys),
    // Divided by the hand path, which is never empty, rather than by handCount, which can be 0.
    footToHandRatio: footCount / path.length,
  };
}

/** `bisect_right`: the number of edges at or below the value. */
export function featureBucket(edges: number[], value: number): number {
  let count = 0;
  while (count < edges.length && edges[count] <= value) {
    count += 1;
  }
  return count;
}

export function styleBuckets(features: StyleFeatures, style: StyleArtifact): number[] {
  return FEATURE_NAMES.map((name) =>
    featureBucket(style.bucket_edges[PYTHON_FEATURE_NAMES[name]], features[name]),
  );
}

/** Mean scale-normalized violation across the preset's constrained features; 0 inside them. */
export function styleDistance(
  features: StyleFeatures,
  presetName: string,
  style: StyleArtifact,
): number {
  const preset = style.presets[presetName];
  if (preset === undefined) {
    throw new Error(`Unknown style preset: ${presetName}`);
  }
  const violations: number[] = [];
  for (const [pythonName, [low, high]] of Object.entries(preset.bounds)) {
    const name = FEATURE_NAMES.find((candidate) => PYTHON_FEATURE_NAMES[candidate] === pythonName);
    if (name === undefined) {
      throw new Error(`Unknown style feature: ${pythonName}`);
    }
    const value = features[name];
    const below = low !== null && value < low ? low - value : 0;
    const above = high !== null && value > high ? value - high : 0;
    violations.push((below + above) / style.feature_scales[pythonName]);
  }
  return violations.length > 0 ? mean(violations) : 0;
}

export function matchesPreset(
  features: StyleFeatures,
  presetName: string,
  style: StyleArtifact,
): boolean {
  return styleDistance(features, presetName, style) === 0;
}
