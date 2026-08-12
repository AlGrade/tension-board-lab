/**
 * The sampler, end to end against the real generator graph.
 *
 * Validity is the claim that matters: masking should make an invalid problem unreachable, not
 * merely unlikely, so every sample must satisfy every rule.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { BoulderGenerator, nucleus, rankCandidates } from "../src/generate/sample";
import { positionKey } from "../src/model/tokenizer";
import type { BoardArtifact, GeneratorArtifact } from "../src/types";

const DATA = new URL("../public/data/", import.meta.url);
const MODEL = fileURLToPath(new URL("../public/models/generator.onnx", import.meta.url));

function readJson<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(name, DATA), "utf8")) as T;
}

const board = readJson<BoardArtifact>("board.json");
const artifact = readJson<GeneratorArtifact>("generator.json");
const hasModel = existsSync(MODEL);

/** Deterministic PRNG so a failure can be reproduced. */
function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

describe("nucleus filter", () => {
  it("keeps the smallest set covering the threshold", () => {
    const kept = nucleus(Float64Array.from([0.5, 0.3, 0.15, 0.05]), 0.7);
    expect(kept[0]).toBeGreaterThan(0);
    expect(kept[1]).toBeGreaterThan(0);
    expect(kept[2]).toBe(0);
    expect(kept[3]).toBe(0);
  });

  it("never empties the set, even at a tiny threshold", () => {
    const kept = nucleus(Float64Array.from([0.9, 0.05, 0.05]), 0.01);
    expect(kept.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(0);
  });

  it("passes everything through at p = 1", () => {
    const input = Float64Array.from([0.6, 0.4]);
    expect(nucleus(input, 1)).toBe(input);
  });
});

describe("ranking", () => {
  const candidates = [
    { problem: { angle: 40, layout: "mirror", holds: [] }, logLikelihood: -10 },
    { problem: { angle: 40, layout: "mirror", holds: [] }, logLikelihood: -10 },
    { problem: { angle: 40, layout: "mirror", holds: [] }, logLikelihood: -30 },
  ];

  it("puts the closest grade first", () => {
    const ranked = rankCandidates(
      candidates,
      [
        { expectedGradeIndex: 7, confidence: 0.4, grade: "V7" },
        { expectedGradeIndex: 5, confidence: 0.4, grade: "V5" },
        { expectedGradeIndex: 5, confidence: 0.4, grade: "V5" },
      ],
      { targetIndex: 5 },
    );
    expect(ranked[0].expectedGrade).toBe(5);
    expect(ranked.map((c) => c.score)).toEqual([...ranked.map((c) => c.score)].sort((a, b) => a - b));
  });

  it("prefers the more likely problem when grades tie", () => {
    const ranked = rankCandidates(
      [candidates[1], candidates[2]],
      [
        { expectedGradeIndex: 5, confidence: 0.4, grade: "V5" },
        { expectedGradeIndex: 5, confidence: 0.4, grade: "V5" },
      ],
      { targetIndex: 5 },
    );
    expect(ranked[0].logLikelihood).toBe(-10);
  });
});

describe.skipIf(!hasModel)("sampling against the real model", () => {
  const layout = "mirror";

  async function sample(overrides: Record<string, unknown> = {}) {
    const generator = await BoulderGenerator.load(MODEL, artifact, board);
    return generator.sample({
      layout,
      angle: 40,
      grade: "V5",
      count: 6,
      random: seeded(7),
      ...overrides,
    });
  }

  it("produces only valid problems", async () => {
    const onLayout = new Set(board.layouts[layout].map((p) => positionKey(p.x, p.y)));
    const candidates = await sample();
    expect(candidates).toHaveLength(6);
    for (const { problem } of candidates) {
      const roles = problem.holds.map((hold) => hold.role);
      expect(problem.holds.length).toBeGreaterThanOrEqual(2);
      expect(roles).toContain("start");
      expect(roles).toContain("finish");

      const positions = problem.holds.map((hold) => positionKey(hold.x, hold.y));
      expect(new Set(positions).size).toBe(positions.length);
      for (const position of positions) expect(onLayout.has(position)).toBe(true);

      for (const hold of problem.holds) {
        if (hold.role === "start") {
          expect(hold.y).toBeLessThanOrEqual(artifact.constraints.max_start_height);
        }
        if (hold.role === "finish") {
          expect(hold.y).toBeGreaterThanOrEqual(artifact.constraints.min_finish_height);
        }
      }
    }
  }, 300_000);

  it("carries the requested conditioning onto the problem", async () => {
    const candidates = await sample({ angle: 55, grade: "V8", count: 3 });
    for (const { problem } of candidates) {
      expect(problem.angle).toBe(55);
      expect(problem.grade).toBe("V8");
      expect(problem.layout).toBe(layout);
    }
  }, 300_000);

  it("respects a tightened hold limit", async () => {
    const candidates = await sample({ maxHolds: 5, count: 4 });
    for (const { problem } of candidates) {
      expect(problem.holds.length).toBeLessThanOrEqual(5);
      expect(problem.holds.length).toBeGreaterThanOrEqual(2);
    }
  }, 300_000);

  it("reports a finite negative log-likelihood", async () => {
    for (const candidate of await sample({ count: 3 })) {
      expect(Number.isFinite(candidate.logLikelihood)).toBe(true);
      expect(candidate.logLikelihood).toBeLessThan(0);
    }
  }, 300_000);

  it("is reproducible for one seed and varies across seeds", async () => {
    const first = await sample({ count: 3, random: seeded(11) });
    const again = await sample({ count: 3, random: seeded(11) });
    const other = await sample({ count: 3, random: seeded(12) });
    const holds = (list: Awaited<ReturnType<typeof sample>>) =>
      JSON.stringify(list.map((c) => c.problem.holds));
    expect(holds(again)).toBe(holds(first));
    expect(holds(other)).not.toBe(holds(first));
  }, 300_000);

});
