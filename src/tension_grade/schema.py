"""Canonical Essential route schema shared by import, training, and prediction."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .grades import normalize_v_grade

SUPPORTED_ANGLES = frozenset({35, 40, 45, 50, 55})
SUPPORTED_SOURCE_LAYOUTS = frozenset({"mirror", "spray"})
HOLD_ROLES = ("start", "hand", "foot", "finish")


@dataclass(frozen=True)
class HoldNode:
    role: str
    x: float
    y: float
    hold_type: str
    orientation_degrees: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> HoldNode:
        role = str(raw["role"]).strip().lower()
        if role not in HOLD_ROLES:
            raise ValueError(f"Unknown hold role: {role!r}")
        hold_type = str(raw["hold_type"]).strip()
        if not hold_type:
            raise ValueError("Hold type cannot be empty")
        return cls(
            role=role,
            x=float(raw["x"]),
            y=float(raw["y"]),
            hold_type=hold_type,
            orientation_degrees=float(raw["orientation_degrees"]) % 360.0,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "x": self.x,
            "y": self.y,
            "hold_type": self.hold_type,
            "orientation_degrees": self.orientation_degrees,
        }


@dataclass(frozen=True)
class RouteExample:
    climb_id: str
    angle: int
    holds: tuple[HoldNode, ...]
    source_layout: str | None = None
    grade: str | None = None
    ascents: int = 0
    votes: int = 0
    group_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, require_grade: bool = False) -> RouteExample:
        angle = int(raw["angle"])
        if angle not in SUPPORTED_ANGLES:
            raise ValueError(
                f"Unsupported angle {angle}; expected one of {sorted(SUPPORTED_ANGLES)}"
            )
        holds = tuple(HoldNode.from_dict(item) for item in raw["holds"])
        if len(holds) < 2:
            raise ValueError("A climb needs at least two selected holds")
        source_layout_raw = raw.get("source_layout")
        source_layout = (
            str(source_layout_raw).strip().lower() if source_layout_raw is not None else None
        )
        if source_layout is not None and source_layout not in SUPPORTED_SOURCE_LAYOUTS:
            raise ValueError(
                f"Unsupported source layout {source_layout!r}; "
                f"expected one of {sorted(SUPPORTED_SOURCE_LAYOUTS)}"
            )
        grade_raw = raw.get("grade")
        if require_grade and grade_raw is None:
            raise ValueError("Training examples require a grade")
        grade = normalize_v_grade(str(grade_raw)) if grade_raw is not None else None
        return cls(
            climb_id=str(raw.get("climb_id", "prediction")),
            angle=angle,
            holds=holds,
            source_layout=source_layout,
            grade=grade,
            ascents=max(0, int(raw.get("ascents", 0))),
            votes=max(0, int(raw.get("votes", 0))),
            group_id=str(raw["group_id"]) if raw.get("group_id") is not None else None,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "climb_id": self.climb_id,
            "angle": self.angle,
            "holds": [hold.as_dict() for hold in self.holds],
            "ascents": self.ascents,
            "votes": self.votes,
        }
        if self.source_layout is not None:
            result["source_layout"] = self.source_layout
        if self.grade is not None:
            result["grade"] = self.grade
        if self.group_id is not None:
            result["group_id"] = self.group_id
        return result


def load_jsonl(path: str | Path, *, require_grade: bool = True) -> list[RouteExample]:
    examples: list[RouteExample] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                examples.append(
                    RouteExample.from_dict(json.loads(line), require_grade=require_grade)
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid example on line {line_number}: {error}") from error
    if not examples:
        raise ValueError(f"No route examples found in {path}")
    return examples


def write_jsonl(examples: Iterable[RouteExample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for example in examples:
            output.write(json.dumps(example.as_dict(), separators=(",", ":")) + "\n")
