/**
 * Exact mirror of `collate_routes` in `data.py`.
 *
 * Every line here is a contract with the Python side. A deviation does not throw; it produces
 * a confident, wrong grade. The details that have actually bitten, all checked by
 * `test/featurize.test.ts` against fixtures recorded from PyTorch:
 *
 * - orientation is `[sin, cos]`, in that order;
 * - coordinates are passed raw, in [0, 1], with no rescaling;
 * - the angle is passed in raw degrees and normalized inside the model;
 * - an unknown hold type becomes index 0 rather than an error;
 * - role order is start=0, hand=1, foot=2, finish=3;
 * - holds keep the order they were given, and are never sorted.
 */

import { ROLE_TO_INDEX, type CriticArtifact, type Problem } from "../types";

export interface CriticTensors {
  holdTypeIds: BigInt64Array;
  orientations: Float32Array;
  roles: BigInt64Array;
  coordinates: Float32Array;
  mask: Float32Array;
  angles: Float32Array;
  batch: number;
  nodes: number;
}

const DEGREES_TO_RADIANS = Math.PI / 180;

/** Pads to the longest problem in the batch, exactly as the Python collate does. */
export function featurize(problems: Problem[], critic: CriticArtifact): CriticTensors {
  if (problems.length === 0) {
    throw new Error("Cannot featurize an empty batch");
  }
  const batch = problems.length;
  const nodes = Math.max(...problems.map((problem) => problem.holds.length));

  const holdTypeIds = new BigInt64Array(batch * nodes);
  const orientations = new Float32Array(batch * nodes * 2);
  const roles = new BigInt64Array(batch * nodes);
  const coordinates = new Float32Array(batch * nodes * 2);
  const mask = new Float32Array(batch * nodes);
  const angles = new Float32Array(batch);

  problems.forEach((problem, row) => {
    angles[row] = problem.angle;
    problem.holds.forEach((hold, column) => {
      const cell = row * nodes + column;
      // Unknown hold types fall back to 0, matching dict.get(hold_type, 0).
      holdTypeIds[cell] = BigInt(critic.hold_type_to_index[hold.holdType] ?? 0);
      const radians = hold.orientationDegrees * DEGREES_TO_RADIANS;
      orientations[cell * 2] = Math.sin(radians);
      orientations[cell * 2 + 1] = Math.cos(radians);
      roles[cell] = BigInt(ROLE_TO_INDEX[hold.role]);
      coordinates[cell * 2] = hold.x;
      coordinates[cell * 2 + 1] = hold.y;
      mask[cell] = 1;
    });
  });

  return { holdTypeIds, orientations, roles, coordinates, mask, angles, batch, nodes };
}

/** `softmax(logits / temperature)`, the calibrated confidence the critic reports. */
export function calibratedProbabilities(logits: number[], temperature: number): number[] {
  const scaled = logits.map((value) => value / Math.max(temperature, 1e-4));
  const largest = Math.max(...scaled);
  const exponentials = scaled.map((value) => Math.exp(value - largest));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

/**
 * Expected grade index, `sum(p_i * i)`.
 *
 * Used for ranking rather than the argmax: at roughly 38% confidence the argmax is too noisy
 * to order candidates by.
 */
export function expectedGradeIndex(probabilities: number[]): number {
  return probabilities.reduce((sum, value, index) => sum + value * index, 0);
}
