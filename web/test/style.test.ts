/**
 * Parity between `style.ts` and `style.py`.
 *
 * Style drives both the generator's conditioning and the ranking, so a drift here changes what
 * the app produces without ever failing. Values come from fixtures Python computed.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  FEATURE_NAMES,
  PYTHON_FEATURE_NAMES,
  computeStyleFeatures,
  featureBucket,
  handPath,
  matchesPreset,
  moveLengths,
  styleBuckets,
  styleDistance,
} from "../src/style";
import type { Hold, HoldRole, StyleArtifact } from "../src/types";

const DATA = new URL("../public/data/", import.meta.url);

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(name, DATA), "utf8")) as T;
}

interface Fixture {
  problem: {
    holds: {
      role: HoldRole;
      x: number;
      y: number;
      hold_type: string;
      orientation_degrees: number;
    }[];
  };
  style: {
    features: Record<string, number>;
    buckets: number[];
    preset_distances: Record<string, number>;
  };
}

const style = readJson<StyleArtifact>("style.json");
const fixtures = readJson<{ cases: Fixture[] }>("fixtures.json");

function toHolds(fixture: Fixture): Hold[] {
  return fixture.problem.holds.map((hold) => ({
    role: hold.role,
    x: hold.x,
    y: hold.y,
    holdType: hold.hold_type,
    orientationDegrees: hold.orientation_degrees,
  }));
}

describe("style features", () => {
  it("has fixtures carrying Python-computed style values", () => {
    expect(fixtures.cases.length).toBeGreaterThan(0);
    expect(fixtures.cases[0].style).toBeDefined();
  });

  fixtures.cases.forEach((fixture, index) => {
    it(`matches Python features for fixture ${index}`, () => {
      const features = computeStyleFeatures(toHolds(fixture));
      FEATURE_NAMES.forEach((name) => {
        const expected = fixture.style.features[PYTHON_FEATURE_NAMES[name]];
        expect(features[name]).toBeCloseTo(expected, 9);
      });
    });

    it(`matches Python buckets for fixture ${index}`, () => {
      const features = computeStyleFeatures(toHolds(fixture));
      expect(styleBuckets(features, style)).toEqual(fixture.style.buckets);
    });

    it(`matches Python preset distances for fixture ${index}`, () => {
      const features = computeStyleFeatures(toHolds(fixture));
      Object.entries(fixture.style.preset_distances).forEach(([name, expected]) => {
        expect(styleDistance(features, name, style)).toBeCloseTo(expected, 9);
      });
    });
  });
});

describe("hand path", () => {
  const holds: Hold[] = [
    { role: "foot", x: 0.2, y: 0.2, holdType: "a", orientationDegrees: 0 },
    { role: "finish", x: 0.5, y: 0.7, holdType: "a", orientationDegrees: 0 },
    { role: "start", x: 0.5, y: 0.1, holdType: "a", orientationDegrees: 0 },
    { role: "hand", x: 0.5, y: 0.4, holdType: "a", orientationDegrees: 0 },
  ];

  it("excludes feet and runs bottom to top", () => {
    expect(handPath(holds).map((hold) => hold.role)).toEqual(["start", "hand", "finish"]);
  });

  it("breaks ties on y by x, matching Python's (y, x) sort", () => {
    const tied: Hold[] = [
      { role: "hand", x: 0.8, y: 0.5, holdType: "a", orientationDegrees: 0 },
      { role: "hand", x: 0.2, y: 0.5, holdType: "a", orientationDegrees: 0 },
    ];
    expect(handPath(tied).map((hold) => hold.x)).toEqual([0.2, 0.8]);
  });

  it("measures moves between consecutive hand holds only", () => {
    expect(moveLengths(holds)).toHaveLength(2);
    expect(moveLengths(holds)[0]).toBeCloseTo(0.3, 10);
  });

  it("does not mutate the array it was given", () => {
    const original = holds.map((hold) => hold.role);
    handPath(holds);
    expect(holds.map((hold) => hold.role)).toEqual(original);
  });
});

describe("bucketing", () => {
  it("counts edges at or below the value, like bisect_right", () => {
    expect(featureBucket([3, 4, 5, 6], 2.9)).toBe(0);
    expect(featureBucket([3, 4, 5, 6], 3)).toBe(1);
    expect(featureBucket([3, 4, 5, 6], 3.5)).toBe(1);
    expect(featureBucket([3, 4, 5, 6], 6)).toBe(4);
    expect(featureBucket([3, 4, 5, 6], 99)).toBe(4);
  });

  it("keeps every bucket index inside the exported edges", () => {
    fixtures.cases.forEach((fixture) => {
      const buckets = styleBuckets(computeStyleFeatures(toHolds(fixture)), style);
      buckets.forEach((bucket, index) => {
        const edges = style.bucket_edges[PYTHON_FEATURE_NAMES[FEATURE_NAMES[index]]];
        expect(bucket).toBeGreaterThanOrEqual(0);
        expect(bucket).toBeLessThanOrEqual(edges.length);
      });
    });
  });
});

describe("presets", () => {
  it("exposes the same presets as Python", () => {
    expect(Object.keys(style.presets).sort()).toEqual([
      "dyno",
      "endurance",
      "power",
      "technical",
    ]);
  });

  it("agrees that matching means zero distance", () => {
    fixtures.cases.forEach((fixture) => {
      const features = computeStyleFeatures(toHolds(fixture));
      Object.keys(style.presets).forEach((name) => {
        expect(matchesPreset(features, name, style)).toBe(
          styleDistance(features, name, style) === 0,
        );
      });
    });
  });

  it("rejects an unknown preset instead of scoring it zero", () => {
    const features = computeStyleFeatures(toHolds(fixtures.cases[0]));
    expect(() => styleDistance(features, "not-a-style", style)).toThrow();
  });
});
