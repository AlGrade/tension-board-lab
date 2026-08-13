/**
 * End-to-end parity: a problem goes through the TypeScript featurizer and the exported ONNX
 * graph, and must land on the logits PyTorch recorded for it.
 *
 * `featurize.test.ts` proves the tensors match. This proves the whole chain does, which is
 * what the app actually depends on.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { Critic } from "../src/model/critic";
import type { CriticArtifact, Hold, HoldRole, Problem } from "../src/types";
import { modelPath, readArtifact } from "./artifacts";

const MODEL = modelPath("grade.onnx");

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
  logits: number[];
}

const critic = readArtifact<CriticArtifact>("critic.json");
const fixtures = readArtifact<{ temperature: number; cases: Fixture[] }>("fixtures.json");

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

describe.skipIf(!MODEL || !critic || !fixtures)("critic through onnxruntime", () => {
  it("reproduces the PyTorch logits for every fixture", async () => {
    const model = await Critic.load(MODEL!, critic!);
    for (const fixture of fixtures!.cases) {
      const [prediction] = await model.predict([toProblem(fixture)]);
      const expected = fixture.logits;
      prediction.probabilities.forEach((_, index) => {
        expect(prediction.probabilities[index]).toBeGreaterThanOrEqual(0);
      });
      // Compare logits by way of the probabilities they produce, which is what the UI shows.
      const scaled = expected.map((value) => value / fixtures!.temperature);
      const largest = Math.max(...scaled);
      const exponentials = scaled.map((value) => Math.exp(value - largest));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      expected.forEach((_, index) => {
        expect(prediction.probabilities[index]).toBeCloseTo(exponentials[index] / total, 4);
      });
    }
  }, 120_000);

  it("scores a batch identically to one problem at a time", async () => {
    const model = await Critic.load(MODEL!, critic!);
    const problems = fixtures!.cases.slice(0, 5).map(toProblem);
    const batched = await model.predict(problems);
    for (const [index, problem] of problems.entries()) {
      const [single] = await model.predict([problem]);
      // Padding to the longest problem in the batch must not change any result.
      expect(batched[index].expectedGrade).toBeCloseTo(single.expectedGrade, 4);
      expect(batched[index].grade).toBe(single.grade);
    }
  }, 120_000);

  it("agrees with the documented example prediction", async () => {
    const model = await Critic.load(MODEL!, critic!);
    const route = JSON.parse(
      readFileSync(new URL("../../examples/route.json", import.meta.url), "utf8"),
    ) as {
      angle: number;
      holds: {
        role: HoldRole;
        x: number;
        y: number;
        hold_type: string;
        orientation_degrees: number;
      }[];
    };
    const [prediction] = await model.predict([
      {
        angle: route.angle,
        layout: "mirror",
        holds: route.holds.map((hold) => ({
          role: hold.role,
          x: hold.x,
          y: hold.y,
          holdType: hold.hold_type,
          orientationDegrees: hold.orientation_degrees,
        })),
      },
    ]);
    expect(prediction.grade).toBe("V9");
    expect(prediction.confidence).toBeCloseTo(0.3832, 3);
  }, 120_000);
});
