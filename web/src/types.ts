/** Shapes shared by the exported artifacts and the app. Mirrors `schema.py`. */

export const HOLD_ROLES = ["start", "hand", "foot", "finish"] as const;

export type HoldRole = (typeof HOLD_ROLES)[number];

/** Role order is part of the model contract: start=0, hand=1, foot=2, finish=3. */
export const ROLE_TO_INDEX: Record<HoldRole, number> = {
  start: 0,
  hand: 1,
  foot: 2,
  finish: 3,
};

export interface Hold {
  role: HoldRole;
  x: number;
  y: number;
  holdType: string;
  orientationDegrees: number;
}

export interface Problem {
  angle: number;
  layout: string;
  holds: Hold[];
  grade?: string;
}

export interface Placement {
  x: number;
  y: number;
  raw_x: number;
  raw_y: number;
  hold_type: string;
  orientation_degrees: number;
}

export interface BoardArtifact {
  board: string;
  layouts: Record<string, Placement[]>;
  role_colors: Record<HoldRole, string>;
  coordinate_inverse: {
    x_scale: number;
    x_offset: number;
    y_scale: number;
    y_offset: number;
  };
}

export interface CriticArtifact {
  hold_type_to_index: Record<string, number>;
  grade_labels: string[];
  grade_offset: number;
  temperature: number;
  role_to_index: Record<HoldRole, number>;
  input_order: string[];
}

export interface GeneratorArtifact {
  vocabulary: {
    positions: [number, number][];
    layouts: string[];
    angles: number[];
    grade_labels: string[];
    layout_offset: number;
    angle_offset: number;
    grade_offset: number;
    hold_offset: number;
  };
  size: number;
  special_tokens: { pad: number; bos: number; eos: number; uncond: number };
  prefix_length: number;
  max_sequence_length: number;
  max_holds: number;
  roles: HoldRole[];
  /** Index of each layout in the model's physical lookup tables. */
  layout_order: string[];
  constraints: {
    max_start_height: number;
    min_finish_height: number;
    min_holds: number;
  };
}
