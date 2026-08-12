/**
 * Parity with PyTorch, from fixtures recorded by `tension-export-web`.
 *
 * This is the test the whole frontend rests on. If the featurization drifts from
 * `collate_routes`, the app shows wrong grades and nothing errors.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { calibratedProbabilities, expectedGradeIndex, featurize } from "../src/model/featurize";
import type { CriticArtifact, Hold, HoldRole, Problem } from "../src/types";

const DATA = new URL("../public/data/", import.meta.url);

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(name, DATA), "utf8")) as T;
}

interface Fixture {
  problem: {
    angle: number;
    source_layout?: string;
    holds: {
      role: HoldRole;
      x: number;
      y: number;
      hold_type: string;
      orientation_degrees: number;
    }[];
  };
  tensors: {
    hold_type_ids: number[];
    orientations: number[][];
    roles: number[];
    coordinates: number[][];
    mask: number[];
    angles: number[];
  };
  logits: number[];
}

const critic = readJson<CriticArtifact>("critic.json");
const fixtures = readJson<{ temperature: number; cases: Fixture[] }>("fixtures.json");

function toProblem(fixture: Fixture): Problem {
  return {
    angle: fixture.problem.angle,
    layout: fixture.problem.source_layout ?? "mirror",
    holds: fixture.problem.holds.map(
      (hold): Hold => ({
        role: hold.role,
        x: hold.x,
        y: hold.y,
        holdType: hold.hold_type,
        orientationDegrees: hold.orientation_degrees,
      }),
    ),
  };
}

describe("featurize", () => {
  it("has fixtures to check against", () => {
    expect(fixtures.cases.length).toBeGreaterThan(0);
  });

  fixtures.cases.forEach((fixture, index) => {
    const holds = fixture.problem.holds.length;

    it(`matches PyTorch tensors for fixture ${index} (${holds} holds)`, () => {
      const tensors = featurize([toProblem(fixture)], critic);
      expect(tensors.nodes).toBe(holds);

      expect([...tensors.holdTypeIds].map(Number)).toEqual(fixture.tensors.hold_type_ids);
      expect([...tensors.roles].map(Number)).toEqual(fixture.tensors.roles);
      expect([...tensors.mask]).toEqual(fixture.tensors.mask);
      expect([...tensors.angles]).toEqual(fixture.tensors.angles);

      fixture.tensors.orientations.forEach(([sine, cosine], column) => {
        // [sin, cos] in that order; swapping them is silent and wrong.
        expect(tensors.orientations[column * 2]).toBeCloseTo(sine, 6);
        expect(tensors.orientations[column * 2 + 1]).toBeCloseTo(cosine, 6);
      });

      fixture.tensors.coordinates.forEach(([x, y], column) => {
        // Raw [0, 1] coordinates, never rescaled.
        expect(tensors.coordinates[column * 2]).toBeCloseTo(x, 6);
        expect(tensors.coordinates[column * 2 + 1]).toBeCloseTo(y, 6);
      });
    });
  });

  it("pads a mixed batch to the longest problem and masks the rest", () => {
    const short = toProblem(fixtures.cases[0]);
    const long = toProblem(fixtures.cases[fixtures.cases.length - 1]);
    const tensors = featurize([short, long], critic);
    expect(tensors.nodes).toBe(Math.max(short.holds.length, long.holds.length));

    for (let column = 0; column < tensors.nodes; column += 1) {
      const active = column < short.holds.length;
      expect(tensors.mask[column]).toBe(active ? 1 : 0);
      if (!active) {
        expect(tensors.holdTypeIds[column]).toBe(0n);
        expect(tensors.coordinates[column * 2]).toBe(0);
      }
    }
  });

  it("keeps holds in the order they were given", () => {
    const problem = toProblem(fixtures.cases[1]);
    const reversed: Problem = { ...problem, holds: [...problem.holds].reverse() };
    const forward = featurize([problem], critic);
    const backward = featurize([reversed], critic);
    expect([...backward.roles].map(Number)).toEqual([...forward.roles].map(Number).reverse());
  });

  it("maps an unknown hold type to index 0 rather than throwing", () => {
    const problem = toProblem(fixtures.cases[0]);
    problem.holds[0].holdType = "wood:NOT-A-REAL-HOLD";
    expect(featurize([problem], critic).holdTypeIds[0]).toBe(0n);
  });

  it("uses the same role order as the exported contract", () => {
    expect(critic.role_to_index).toEqual({ start: 0, hand: 1, foot: 2, finish: 3 });
  });
});

describe("calibration", () => {
  it("reproduces softmax(logits / temperature) from the fixtures", () => {
    const fixture = fixtures.cases[0];
    const probabilities = calibratedProbabilities(fixture.logits, fixtures.temperature);
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    expect(total).toBeCloseTo(1, 10);
    expect(probabilities.length).toBe(critic.grade_labels.length);

    // Recomputed independently, without the max-subtraction guard.
    const naive = fixture.logits.map((value) => Math.exp(value / fixtures.temperature));
    const naiveTotal = naive.reduce((sum, value) => sum + value, 0);
    probabilities.forEach((value, index) => {
      expect(value).toBeCloseTo(naive[index] / naiveTotal, 10);
    });
  });

  it("uses the temperature the critic was calibrated with", () => {
    expect(critic.temperature).toBeCloseTo(1.67, 2);
    expect(fixtures.temperature).toBeCloseTo(critic.temperature, 10);
  });

  it("computes the expected grade as a probability-weighted index", () => {
    expect(expectedGradeIndex([0, 1, 0])).toBeCloseTo(1, 10);
    expect(expectedGradeIndex([0.5, 0, 0.5])).toBeCloseTo(1, 10);
    expect(expectedGradeIndex([0.25, 0.75])).toBeCloseTo(0.75, 10);
  });
});
