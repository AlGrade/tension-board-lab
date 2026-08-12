/**
 * The board, rendered as a photo of the real wall with hold markers over it.
 *
 * Positions arrive normalized to [0, 1] with y measured upwards from the bottom of the wall;
 * SVG measures y downwards, so the flip happens here and nowhere else. Where those normalized
 * coordinates sit inside each photo comes from `calibration.json`, which was fitted against
 * the hold lattice rather than eyeballed — see `docs/board-images.md`.
 */

import type { BoardArtifact, Hold, HoldRole, Placement } from "../types";

export interface BoardCalibration {
  image: string;
  width: number;
  height: number;
  /** Fractions of the image where normalized x = 0 and x = 1 fall. */
  x0: number;
  x1: number;
  /** Fractions of the image where normalized y = 0 (bottom) and y = 1 (top) fall. */
  y0: number;
  y1: number;
}

export type CalibrationMap = Record<string, BoardCalibration>;

export interface BoardViewProps {
  board: BoardArtifact;
  calibration: CalibrationMap;
  layout: string;
  holds: Hold[];
  onToggle?: (placement: Placement) => void;
}

function key(x: number, y: number): string {
  return `${x.toFixed(6)},${y.toFixed(6)}`;
}

export function BoardView({ board, calibration, layout, holds, onToggle }: BoardViewProps) {
  const placements = board.layouts[layout] ?? [];
  const photo = calibration[layout];
  const selected = new Map<string, HoldRole>(
    holds.map((hold) => [key(hold.x, hold.y), hold.role]),
  );

  if (!photo) {
    return <p className="footnote">No board image for the {layout} layout.</p>;
  }

  const { width, height } = photo;
  const project = (x: number, y: number) => ({
    cx: (photo.x0 + x * (photo.x1 - photo.x0)) * width,
    cy: (photo.y0 + y * (photo.y1 - photo.y0)) * height,
  });

  // One grid step, used to size the markers so they scale with the board.
  const step = ((photo.x1 - photo.x0) * width) / 32;
  // Wider than the grid spacing, so the ring encircles the hold instead of sitting on it.
  // Rings of adjacent selections then overlap slightly, which reads fine.
  const radius = step * 0.68;
  const stroke = step * 0.075;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="board"
      role="img"
      aria-label={`${board.board}, ${layout} layout`}
    >
      <image href={photo.image} x={0} y={0} width={width} height={height} />

      {placements.map((placement) => {
        const { cx, cy } = project(placement.x, placement.y);
        const role = selected.get(key(placement.x, placement.y));
        const color = role ? `#${board.role_colors[role]}` : undefined;
        return (
          <g key={`${placement.raw_x},${placement.raw_y}`}>
            {role ? (
              <>
                {/* A dark halo under the ring, so it reads on cream holds as well as black. */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  className="hold-halo"
                  style={{ strokeWidth: stroke * 1.9 }}
                  pointerEvents="none"
                />
                <circle
                  cx={cx}
                  cy={cy}
                  r={radius}
                  className="hold-ring"
                  style={{ stroke: color, strokeWidth: stroke }}
                  pointerEvents="none"
                />
              </>
            ) : null}
            {/* An invisible target: the photo already shows the hold, so nothing covers it.
                Kept under half a grid step so neighbouring targets never overlap. */}
            <circle
              cx={cx}
              cy={cy}
              r={step * 0.46}
              className="hold-target"
              onClick={onToggle ? () => onToggle(placement) : undefined}
            >
              <title>
                {placement.hold_type} at ({placement.raw_x}, {placement.raw_y})
                {role ? ` — ${role}` : ""}
              </title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}
