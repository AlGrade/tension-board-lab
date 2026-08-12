/**
 * Mirror of `generator/constraints.py`.
 *
 * Masking during sampling rather than filtering afterwards is what keeps generation cheap and
 * makes an invalid problem unreachable rather than unlikely. Thresholds arrive in
 * `generator.json`, so they cannot drift from the Python side.
 */

import { GeneratorVocabulary, positionKey } from "../model/tokenizer";
import { HOLD_ROLES, type BoardArtifact, type GeneratorArtifact } from "../types";

export interface ConstraintOptions {
  maxHolds?: number;
}

export class BoulderConstraints {
  private readonly allowedHolds: Uint8Array;
  private readonly startTokens: number[] = [];
  private readonly finishTokens: number[] = [];
  private readonly maxHolds: number;
  private readonly minHolds: number;

  constructor(
    private readonly vocabulary: GeneratorVocabulary,
    board: BoardArtifact,
    artifact: GeneratorArtifact,
    readonly layout: string,
    options: ConstraintOptions = {},
  ) {
    if (!vocabulary.layouts.includes(layout)) {
      throw new Error(`Unknown layout ${layout}`);
    }
    const { max_start_height, min_finish_height, min_holds } = artifact.constraints;
    this.maxHolds = options.maxHolds ?? vocabulary.maxHolds;
    this.minHolds = min_holds;
    if (this.maxHolds < this.minHolds) {
      throw new Error("maxHolds cannot be below minHolds");
    }

    const onLayout = new Set(
      (board.layouts[layout] ?? []).map((placement) =>
        positionKey(placement.x, placement.y),
      ),
    );
    this.allowedHolds = new Uint8Array(vocabulary.size);
    vocabulary.positions.forEach(([x, y]) => {
      if (!onLayout.has(positionKey(x, y))) return;
      for (const role of HOLD_ROLES) {
        if (role === "start" && y > max_start_height) continue;
        if (role === "finish" && y < min_finish_height) continue;
        const token = vocabulary.holdToken(x, y, role);
        this.allowedHolds[token] = 1;
        if (role === "start") this.startTokens.push(token);
        if (role === "finish") this.finishTokens.push(token);
      }
    });
  }

  /** `true` at a token means it may be sampled next. */
  maskFor(chosen: number[]): Uint8Array {
    const allowed = this.allowedHolds.slice();
    let starts = 0;
    let finishes = 0;
    for (const token of chosen) {
      const { x, y, role } = this.vocabulary.holdFromToken(token);
      // One position carries a single role, so retire all of its tokens at once.
      for (const other of this.vocabulary.positionTokens(this.vocabulary.positionIndexOf(x, y))) {
        allowed[other] = 0;
      }
      if (role === "start") starts += 1;
      if (role === "finish") finishes += 1;
    }

    const missing = (starts === 0 ? 1 : 0) + (finishes === 0 ? 1 : 0);
    const complete = chosen.length >= this.minHolds && missing === 0;

    // Leave room to still place whatever the problem is missing.
    if (chosen.length + missing >= this.maxHolds) {
      const required = new Uint8Array(allowed.length);
      if (starts === 0) for (const token of this.startTokens) required[token] = 1;
      if (finishes === 0) for (const token of this.finishTokens) required[token] = 1;
      for (let index = 0; index < allowed.length; index += 1) {
        allowed[index] = missing > 0 ? allowed[index] & required[index] : 0;
      }
    }

    allowed[this.vocabulary.eos] = complete ? 1 : 0;
    return allowed;
  }

  isComplete(chosen: number[]): boolean {
    return this.maskFor(chosen)[this.vocabulary.eos] === 1;
  }
}
