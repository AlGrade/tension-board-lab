/**
 * Mirror of `generator/sample.py`, running in the browser.
 *
 * Both the conditional and the unconditional sequence go through the model in one batch, so
 * classifier-free guidance costs one call per step rather than two. Hard rules are applied as
 * logit masks during sampling, so an invalid problem is unreachable rather than unlikely.
 */

import * as ort from "onnxruntime-web";

import { GeneratorVocabulary, decodeHolds, encodePrefix } from "../model/tokenizer";
import { computeStyleFeatures, styleDistance } from "../style";
import type { BoardArtifact, GeneratorArtifact, Problem, StyleArtifact } from "../types";
import { BoulderConstraints } from "./constraints";

export interface SampleRequest {
  layout: string;
  angle: number;
  grade: string;
  preset?: string;
  count?: number;
  temperature?: number;
  topP?: number;
  guidance?: number;
  maxHolds?: number;
  /** Injectable for tests; defaults to `Math.random`. */
  random?: () => number;
}

export interface Candidate {
  problem: Problem;
  logLikelihood: number;
}

/** Keeps the smallest set of tokens covering `topP` of the mass, never empty. */
export function nucleus(probabilities: Float64Array, topP: number): Float64Array {
  if (topP >= 1) return probabilities;
  const order = Array.from(probabilities.keys()).sort(
    (a, b) => probabilities[b] - probabilities[a],
  );
  const kept = new Float64Array(probabilities.length);
  let cumulative = 0;
  for (const index of order) {
    // Keep the token that crosses the threshold, so the set is never empty.
    if (cumulative >= topP) break;
    kept[index] = probabilities[index];
    cumulative += probabilities[index];
  }
  return kept;
}

function drawFrom(weights: Float64Array, random: () => number): number {
  let total = 0;
  for (const value of weights) total += value;
  if (!(total > 0)) throw new Error("Cannot sample from an empty distribution");
  let target = random() * total;
  for (let index = 0; index < weights.length; index += 1) {
    target -= weights[index];
    if (target <= 0) return index;
  }
  // Floating point can undershoot; fall back to the last non-zero token.
  for (let index = weights.length - 1; index >= 0; index -= 1) {
    if (weights[index] > 0) return index;
  }
  throw new Error("Cannot sample from an empty distribution");
}

export class BoulderGenerator {
  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly vocabulary: GeneratorVocabulary,
    private readonly board: BoardArtifact,
    private readonly artifact: GeneratorArtifact,
    private readonly style: StyleArtifact,
  ) {}

  static async load(
    modelUrl: string,
    artifact: GeneratorArtifact,
    board: BoardArtifact,
    style: StyleArtifact,
  ): Promise<BoulderGenerator> {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new BoulderGenerator(
      session,
      new GeneratorVocabulary(artifact, style),
      board,
      artifact,
      style,
    );
  }

  async sample(request: SampleRequest): Promise<Candidate[]> {
    const {
      layout,
      angle,
      grade,
      preset,
      count = 12,
      temperature = 1,
      topP = 0.92,
      guidance = 1.5,
      random = Math.random,
    } = request;

    const vocabulary = this.vocabulary;
    const constraints = new BoulderConstraints(
      vocabulary,
      this.board,
      this.artifact,
      layout,
      { maxHolds: request.maxHolds },
    );
    const style = preset ? this.style.presets[preset]?.conditioning_buckets : undefined;

    const conditional = encodePrefix(vocabulary, { layout, angle, grade, style });
    const unconditional = encodePrefix(vocabulary, {
      layout,
      angle,
      grade,
      unconditional: true,
    });

    // Rows 0..count-1 are conditional, count..2*count-1 their unconditional twins.
    const rows: number[][] = [];
    for (let index = 0; index < count; index += 1) rows.push([...conditional]);
    for (let index = 0; index < count; index += 1) rows.push([...unconditional]);

    const chosen: number[][] = Array.from({ length: count }, () => []);
    const finished = new Array<boolean>(count).fill(false);
    const logLikelihoods = new Array<number>(count).fill(0);
    const steps = request.maxHolds ?? vocabulary.maxHolds;

    for (let step = 0; step <= steps; step += 1) {
      const logits = await this.forward(rows);
      const next: number[] = [];

      for (let row = 0; row < count; row += 1) {
        if (finished[row]) {
          next.push(vocabulary.eos);
          continue;
        }
        const guided = new Float64Array(vocabulary.size);
        for (let token = 0; token < vocabulary.size; token += 1) {
          const conditionalLogit = logits[row][token];
          const freeLogit = logits[count + row][token];
          const combined =
            guidance === 1
              ? conditionalLogit
              : freeLogit + guidance * (conditionalLogit - freeLogit);
          guided[token] = combined / Math.max(temperature, 1e-4);
        }

        const mask = constraints.maskFor(chosen[row]);
        let largest = -Infinity;
        for (let token = 0; token < guided.length; token += 1) {
          if (mask[token] === 1 && guided[token] > largest) largest = guided[token];
        }
        const distribution = new Float64Array(guided.length);
        let total = 0;
        for (let token = 0; token < guided.length; token += 1) {
          if (mask[token] !== 1) continue;
          const value = Math.exp(guided[token] - largest);
          distribution[token] = value;
          total += value;
        }
        for (let token = 0; token < distribution.length; token += 1) {
          distribution[token] /= total;
        }

        let filtered = nucleus(distribution, topP);
        let filteredTotal = 0;
        for (const value of filtered) filteredTotal += value;
        if (!(filteredTotal > 0)) filtered = distribution;

        const token = drawFrom(filtered, random);
        logLikelihoods[row] += Math.log(Math.max(distribution[token], 1e-12));
        next.push(token);
        if (token === vocabulary.eos) {
          finished[row] = true;
        } else {
          chosen[row].push(token);
        }
      }

      for (let row = 0; row < count; row += 1) {
        rows[row].push(next[row]);
        rows[count + row].push(next[row]);
      }
      if (finished.every(Boolean)) break;
    }

    return chosen.map((tokens, row) => ({
      problem: {
        angle,
        layout,
        grade,
        holds: decodeHolds(tokens, vocabulary, this.board, layout),
      },
      logLikelihood: logLikelihoods[row],
    }));
  }

  /** Last-position logits for every row, as plain arrays. */
  private async forward(rows: number[][]): Promise<Float32Array[]> {
    const batch = rows.length;
    const length = rows[0].length;
    const flat = new BigInt64Array(batch * length);
    rows.forEach((row, index) => {
      row.forEach((token, position) => {
        flat[index * length + position] = BigInt(token);
      });
    });
    const output = await this.session.run({
      tokens: new ort.Tensor("int64", flat, [batch, length]),
    });
    const logits = output.logits.data as Float32Array;
    const size = this.vocabulary.size;
    return rows.map((_, index) => {
      const start = (index * length + (length - 1)) * size;
      return logits.subarray(start, start + size);
    });
  }
}

export interface RankedCandidate extends Candidate {
  expectedGrade: number;
  confidence: number;
  grade: string;
  styleDistance: number;
  score: number;
}

/**
 * Lower is better: grade error, then style error, minus plausibility.
 *
 * Mirrors `rank_candidates` in `sample.py`, including its weights.
 */
export function rankCandidates(
  candidates: Candidate[],
  scored: { expectedGradeIndex: number; confidence: number; grade: string }[],
  options: {
    targetIndex: number;
    preset?: string;
    style: StyleArtifact;
    gradeWeight?: number;
    styleWeight?: number;
    likelihoodWeight?: number;
  },
): RankedCandidate[] {
  const {
    targetIndex,
    preset,
    style,
    gradeWeight = 1,
    styleWeight = 0.5,
    likelihoodWeight = 0.05,
  } = options;

  return candidates
    .map((candidate, index) => {
      const distance = preset
        ? styleDistance(computeStyleFeatures(candidate.problem.holds), preset, style)
        : 0;
      const score =
        gradeWeight * Math.abs(scored[index].expectedGradeIndex - targetIndex) +
        styleWeight * distance -
        likelihoodWeight * candidate.logLikelihood;
      return {
        ...candidate,
        expectedGrade: scored[index].expectedGradeIndex,
        confidence: scored[index].confidence,
        grade: scored[index].grade,
        styleDistance: distance,
        score,
      };
    })
    .sort((a, b) => a.score - b.score);
}
