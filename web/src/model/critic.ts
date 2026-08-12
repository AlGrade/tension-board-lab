/**
 * The grade critic, running in the browser through onnxruntime-web.
 *
 * The graph is dynamic in both batch and hold count, so a whole set of candidates goes through
 * in one call — which is how the sampler is meant to use it.
 */

import * as ort from "onnxruntime-web";

import type { CriticArtifact, Problem } from "../types";
import { calibratedProbabilities, expectedGradeIndex, featurize } from "./featurize";

export interface Prediction {
  /** Most likely grade label, for example `V9`. */
  grade: string;
  /** Probability of that label, after temperature calibration. */
  confidence: number;
  /** Probability-weighted grade index; the ranking signal. */
  expectedGradeIndex: number;
  /** `expectedGradeIndex` shifted onto the absolute V scale. */
  expectedGrade: number;
  probabilities: number[];
}

export class Critic {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly artifact: CriticArtifact,
  ) {}

  static async load(modelUrl: string, artifact: CriticArtifact): Promise<Critic> {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new Critic(session, artifact);
  }

  get gradeLabels(): string[] {
    return this.artifact.grade_labels;
  }

  /** Index of a grade label on the critic's own axis, for comparing against a request. */
  gradeIndex(label: string): number {
    const index = this.artifact.grade_labels.indexOf(label);
    if (index < 0) {
      throw new Error(`Grade ${label} is outside the critic's range`);
    }
    return index;
  }

  async predict(problems: Problem[]): Promise<Prediction[]> {
    const tensors = featurize(problems, this.artifact);
    const { batch, nodes } = tensors;
    const feeds: Record<string, ort.Tensor> = {
      hold_type_ids: new ort.Tensor("int64", tensors.holdTypeIds, [batch, nodes]),
      orientations: new ort.Tensor("float32", tensors.orientations, [batch, nodes, 2]),
      roles: new ort.Tensor("int64", tensors.roles, [batch, nodes]),
      coordinates: new ort.Tensor("float32", tensors.coordinates, [batch, nodes, 2]),
      mask: new ort.Tensor("float32", tensors.mask, [batch, nodes]),
      angles: new ort.Tensor("float32", tensors.angles, [batch]),
    };

    const output = await this.session.run(feeds);
    const logits = output.logits.data as Float32Array;
    const grades = this.artifact.grade_labels.length;

    return problems.map((_, row) => {
      const slice = Array.from(logits.slice(row * grades, (row + 1) * grades));
      const probabilities = calibratedProbabilities(slice, this.artifact.temperature);
      let best = 0;
      probabilities.forEach((value, index) => {
        if (value > probabilities[best]) best = index;
      });
      const expected = expectedGradeIndex(probabilities);
      return {
        grade: this.artifact.grade_labels[best],
        confidence: probabilities[best],
        expectedGradeIndex: expected,
        expectedGrade: expected + this.artifact.grade_offset,
        probabilities,
      };
    });
  }
}
