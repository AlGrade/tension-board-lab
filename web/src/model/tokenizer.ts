/**
 * Mirror of `generator/tokenizer.py`.
 *
 * A sequence is `[BOS] layout angle grade style x7 | hold ... hold | [EOS]`. The prefix has a
 * fixed width so guidance can blank it without shifting the holds, and a hold token carries
 * only `(position, role)` — type and orientation come from the board catalog.
 */

import { PYTHON_FEATURE_NAMES, FEATURE_NAMES, type FeatureName } from "../style";
import {
  HOLD_ROLES,
  ROLE_TO_INDEX,
  type BoardArtifact,
  type GeneratorArtifact,
  type Hold,
  type HoldRole,
  type StyleArtifact,
} from "../types";

const COORDINATE_PRECISION = 6;

export function positionKey(x: number, y: number): string {
  return `${x.toFixed(COORDINATE_PRECISION)},${y.toFixed(COORDINATE_PRECISION)}`;
}

export class GeneratorVocabulary {
  readonly pad: number;
  readonly bos: number;
  readonly eos: number;
  readonly uncond: number;
  readonly prefixLength: number;
  readonly size: number;
  readonly maxHolds: number;
  readonly maxSequenceLength: number;

  private readonly positionIndex = new Map<string, number>();
  private readonly styleSlots: number[];

  constructor(
    private readonly artifact: GeneratorArtifact,
    private readonly style: StyleArtifact,
  ) {
    const { special_tokens } = artifact;
    this.pad = special_tokens.pad;
    this.bos = special_tokens.bos;
    this.eos = special_tokens.eos;
    this.uncond = special_tokens.uncond;
    this.prefixLength = artifact.prefix_length;
    this.size = artifact.size;
    this.maxHolds = artifact.max_holds;
    this.maxSequenceLength = artifact.max_sequence_length;

    artifact.vocabulary.positions.forEach(([x, y], index) => {
      this.positionIndex.set(positionKey(x, y), index);
    });
    // Buckets per feature, plus the "any" slot the Python side appends.
    this.styleSlots = FEATURE_NAMES.map(
      (name) => style.bucket_edges[PYTHON_FEATURE_NAMES[name]].length + 1,
    );
  }

  get positions(): [number, number][] {
    return this.artifact.vocabulary.positions;
  }

  get layouts(): string[] {
    return this.artifact.vocabulary.layouts;
  }

  get holdOffset(): number {
    return this.artifact.vocabulary.hold_offset;
  }

  layoutToken(layout: string): number {
    const index = this.artifact.vocabulary.layouts.indexOf(layout);
    if (index < 0) throw new Error(`Unknown layout ${layout}`);
    return this.artifact.vocabulary.layout_offset + index;
  }

  angleToken(angle: number): number {
    const index = this.artifact.vocabulary.angles.indexOf(angle);
    if (index < 0) throw new Error(`Unsupported angle ${angle}`);
    return this.artifact.vocabulary.angle_offset + index;
  }

  gradeToken(grade: string): number {
    const index = this.artifact.vocabulary.grade_labels.indexOf(grade);
    if (index < 0) throw new Error(`Grade ${grade} is outside the generator vocabulary`);
    return this.artifact.vocabulary.grade_offset + index;
  }

  /**
   * `null` means unspecified and maps to the slot just past the real buckets.
   *
   * A feature with `n` edges has `n + 1` buckets numbered `0..n`, so "any" is `n + 1` — the
   * same as Python's `bucket_count(feature)`, not `len(edges)`.
   */
  styleToken(feature: FeatureName, bucket: number | null): number {
    const index = FEATURE_NAMES.indexOf(feature);
    const anySlot = this.styleSlots[index];
    const resolved = bucket === null ? anySlot : bucket;
    if (resolved < 0 || resolved > anySlot) {
      throw new Error(`Bucket ${bucket} is out of range for ${feature}`);
    }
    return this.artifact.vocabulary.style_offsets[index] + resolved;
  }

  positionIndexOf(x: number, y: number): number {
    const index = this.positionIndex.get(positionKey(x, y));
    if (index === undefined) throw new Error(`Position ${x},${y} is not on the board`);
    return index;
  }

  holdToken(x: number, y: number, role: HoldRole): number {
    return this.holdOffset + this.positionIndexOf(x, y) * HOLD_ROLES.length + ROLE_TO_INDEX[role];
  }

  isHoldToken(token: number): boolean {
    return token >= this.holdOffset && token < this.size;
  }

  holdFromToken(token: number): { x: number; y: number; role: HoldRole } {
    if (!this.isHoldToken(token)) throw new Error(`Token ${token} is not a hold token`);
    const offset = token - this.holdOffset;
    const positionIndex = Math.floor(offset / HOLD_ROLES.length);
    const roleIndex = offset % HOLD_ROLES.length;
    const [x, y] = this.artifact.vocabulary.positions[positionIndex];
    return { x, y, role: HOLD_ROLES[roleIndex] };
  }

  /** All four token ids for one position; a used position retires every one of them. */
  positionTokens(positionIndex: number): number[] {
    const base = this.holdOffset + positionIndex * HOLD_ROLES.length;
    return HOLD_ROLES.map((_, index) => base + index);
  }

  get styleArtifact(): StyleArtifact {
    return this.style;
  }
}

export interface PrefixRequest {
  layout: string;
  angle: number;
  grade: string;
  /** One entry per style feature, in `FEATURE_NAMES` order; `null` leaves it unspecified. */
  style?: (number | null)[];
  unconditional?: boolean;
}

export function encodePrefix(
  vocabulary: GeneratorVocabulary,
  request: PrefixRequest,
): number[] {
  if (request.unconditional) {
    return [vocabulary.bos, ...Array(vocabulary.prefixLength).fill(vocabulary.uncond)];
  }
  const buckets = request.style ?? FEATURE_NAMES.map(() => null);
  if (buckets.length !== FEATURE_NAMES.length) {
    throw new Error(`Expected ${FEATURE_NAMES.length} style buckets, got ${buckets.length}`);
  }
  return [
    vocabulary.bos,
    vocabulary.layoutToken(request.layout),
    vocabulary.angleToken(request.angle),
    vocabulary.gradeToken(request.grade),
    ...FEATURE_NAMES.map((name, index) => vocabulary.styleToken(name, buckets[index])),
  ];
}

/** Rebuild the holds of a problem from its hold tokens, using the board catalog. */
export function decodeHolds(
  tokens: number[],
  vocabulary: GeneratorVocabulary,
  board: BoardArtifact,
  layout: string,
): Hold[] {
  const placements = new Map(
    (board.layouts[layout] ?? []).map((placement) => [
      positionKey(placement.x, placement.y),
      placement,
    ]),
  );
  return tokens.filter((token) => vocabulary.isHoldToken(token)).map((token) => {
    const { x, y, role } = vocabulary.holdFromToken(token);
    const placement = placements.get(positionKey(x, y));
    if (placement === undefined) {
      throw new Error(`No hold at ${x},${y} in layout ${layout}`);
    }
    return {
      role,
      x: placement.x,
      y: placement.y,
      holdType: placement.hold_type,
      orientationDegrees: placement.orientation_degrees,
    };
  });
}
