/**
 * SVG renderer for the board, driven entirely by `board.json`.
 *
 * Positions arrive normalized to [0, 1] with y measured upwards from the bottom of the wall.
 * SVG measures y downwards, so the flip happens here and nowhere else.
 */

import type { BoardArtifact, Hold, HoldRole, Placement } from "../types";

export interface BoardViewProps {
  board: BoardArtifact;
  layout: string;
  holds: Hold[];
  onToggle?: (placement: Placement) => void;
}

const VIEW = 1000;
const PADDING = 24;

function project(x: number, y: number): { cx: number; cy: number } {
  const span = VIEW - PADDING * 2;
  return { cx: PADDING + x * span, cy: PADDING + (1 - y) * span };
}

function key(x: number, y: number): string {
  return `${x.toFixed(6)},${y.toFixed(6)}`;
}

export function BoardView({ board, layout, holds, onToggle }: BoardViewProps) {
  const placements = board.layouts[layout] ?? [];
  const selected = new Map<string, HoldRole>(
    holds.map((hold) => [key(hold.x, hold.y), hold.role]),
  );

  return (
    <svg
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      className="board"
      role="img"
      aria-label={`${board.board}, ${layout} layout`}
    >
      <rect x={0} y={0} width={VIEW} height={VIEW} rx={16} className="board-face" />

      {placements.map((placement) => {
        const { cx, cy } = project(placement.x, placement.y);
        const role = selected.get(key(placement.x, placement.y));
        const color = role ? `#${board.role_colors[role]}` : undefined;
        return (
          <g key={`${placement.raw_x},${placement.raw_y}`}>
            <circle
              cx={cx}
              cy={cy}
              r={7}
              className={role ? "hold hold-selected" : "hold"}
              style={color ? { fill: color } : undefined}
              onClick={onToggle ? () => onToggle(placement) : undefined}
            >
              <title>
                {placement.hold_type} at ({placement.raw_x}, {placement.raw_y})
                {role ? ` — ${role}` : ""}
              </title>
            </circle>
            {role ? (
              <circle
                cx={cx}
                cy={cy}
                r={17}
                className="hold-ring"
                style={{ stroke: color }}
                pointerEvents="none"
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

export function RoleLegend({ board }: { board: BoardArtifact }) {
  const roles: HoldRole[] = ["start", "hand", "foot", "finish"];
  return (
    <ul className="legend">
      {roles.map((role) => (
        <li key={role}>
          <span className="swatch" style={{ background: `#${board.role_colors[role]}` }} />
          {role}
        </li>
      ))}
    </ul>
  );
}
