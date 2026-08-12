"""The board's fixed hold placements, keyed by layout and normalized position.

Within one layout a position identifies exactly one hold, so hold type and orientation follow
from ``(x, y)``. The generator relies on that: it only chooses a position and a role, and this
catalog supplies the rest.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .schema import SUPPORTED_SOURCE_LAYOUTS

DEFAULT_CATALOG_PATH = Path("configs/tb2_12x12_hold_catalog.csv")

# Positions arrive as floats from JSON and as decimal strings from the catalog. Both are
# rounded to this many places before comparison, matching ``configuration_signature``.
COORDINATE_PRECISION = 6

Position = tuple[float, float]


def position_key(x: float, y: float) -> Position:
    """Round a raw position to the precision at which two holds count as the same."""

    return (round(float(x), COORDINATE_PRECISION), round(float(y), COORDINATE_PRECISION))


@dataclass(frozen=True)
class HoldPlacement:
    layout: str
    x: float
    y: float
    hold_type: str
    orientation_degrees: float


@dataclass(frozen=True)
class HoldCatalog:
    """Placements per layout, plus the position vocabulary shared across layouts."""

    placements: dict[str, dict[Position, HoldPlacement]]
    positions: tuple[Position, ...]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> HoldCatalog:
        with Path(path).open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        if not rows:
            raise ValueError(f"Hold catalog is empty: {path}")

        placements: dict[str, dict[Position, HoldPlacement]] = {}
        for row in rows:
            layout = row["layout"].strip().lower()
            if layout not in SUPPORTED_SOURCE_LAYOUTS:
                raise ValueError(f"Hold catalog contains unsupported layout {layout!r}")
            hold_type = row["hold_type"].strip()
            if not hold_type:
                raise ValueError(f"Hold catalog row for layout {layout!r} has no hold type")
            key = position_key(row["x"], row["y"])
            layout_placements = placements.setdefault(layout, {})
            if key in layout_placements:
                raise ValueError(f"Layout {layout!r} has two holds at position {key}")
            layout_placements[key] = HoldPlacement(
                layout=layout,
                x=key[0],
                y=key[1],
                hold_type=hold_type,
                orientation_degrees=float(row["orientation_degrees"]) % 360.0,
            )

        # Ordered bottom to top, matching the order the generator emits holds in.
        positions = tuple(
            sorted({key for layout in placements.values() for key in layout}, key=lambda p: p[::-1])
        )
        return cls(placements=placements, positions=positions)

    @property
    def layouts(self) -> tuple[str, ...]:
        return tuple(sorted(self.placements))

    def contains(self, layout: str, x: float, y: float) -> bool:
        return position_key(x, y) in self.placements.get(layout, {})

    def placement(self, layout: str, x: float, y: float) -> HoldPlacement:
        try:
            return self.placements[layout][position_key(x, y)]
        except KeyError as error:
            raise KeyError(
                f"No hold at position ({x}, {y}) in layout {layout!r}"
            ) from error
