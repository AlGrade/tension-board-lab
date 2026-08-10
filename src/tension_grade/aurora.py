"""Tools for inspecting and importing the local Aurora/Tension SQLite database."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .schema import SUPPORTED_ANGLES, HoldNode, RouteExample, write_jsonl

_FRAME_TOKEN = re.compile(r"p(?P<placement>\d+)r(?P<role>\d+)")
_ROLE_NAMES = {
    "start": "start",
    "middle": "hand",
    "hand": "hand",
    "foot": "foot",
    "finish": "finish",
    "end": "finish",
}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_database(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables: dict[str, Any] = {}
    names = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    for name_row in names:
        name = str(name_row["name"])
        columns = [
            dict(row)
            for row in connection.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
        ]
        row_count = connection.execute(
            f"SELECT COUNT(*) AS count FROM {quote_identifier(name)}"
        ).fetchone()["count"]
        tables[name] = {"rows": row_count, "columns": columns}
    connection.close()
    return {"database": str(path), "tables": tables}


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Print Aurora SQLite tables and columns as JSON")
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(inspect_database(args.database), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)


def _role_lookup(connection: sqlite3.Connection) -> dict[int, str]:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(placement_roles)")}
    name_column = next((name for name in ("name", "title", "role") if name in columns), None)
    if name_column is None:
        raise ValueError("Could not identify the role-name column in placement_roles")
    result: dict[int, str] = {}
    for role_id, raw_name in connection.execute(
        f"SELECT id, {quote_identifier(name_column)} FROM placement_roles"
    ):
        normalized = str(raw_name).strip().lower()
        mapped = next((value for key, value in _ROLE_NAMES.items() if key in normalized), None)
        if mapped:
            result[int(role_id)] = mapped
    return result


def _parse_frame(
    frame: str, roles: dict[int, str], placements: dict[str, dict[str, Any]]
) -> tuple[HoldNode, ...]:
    holds: list[HoldNode] = []
    for match in _FRAME_TOKEN.finditer(frame):
        placement = match.group("placement")
        role_id = int(match.group("role"))
        if placement not in placements or role_id not in roles:
            continue
        metadata = placements[placement]
        holds.append(
            HoldNode(
                placement_id=placement,
                role=roles[role_id],
                x=float(metadata["x"]),
                y=float(metadata["y"]),
                hold_id=str(metadata["hold_id"]),
                hold_family=str(metadata["hold_family"]),
                variant=str(metadata["variant"]),
                material=str(metadata["material"]),
                orientation_degrees=float(metadata["orientation_degrees"]),
            )
        )
    return tuple(holds)


def _load_hold_catalog(path: Path) -> dict[tuple[int, int, int, int], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    catalog = {
        (
            int(row["layout_id"]),
            int(row["set_id"]),
            int(row["raw_x"]),
            int(row["raw_y"]),
        ): row
        for row in rows
    }
    if len(catalog) != len(rows):
        raise ValueError("Hold catalog contains duplicate layout/set/coordinate rows")
    return catalog


def import_with_query(database: Path, query_file: Path, catalog_file: Path, output: Path) -> int:
    """Run a schema-specific SELECT and convert its rows to canonical JSONL.

    The route query must return layout, climb_id, angle, grade, and frames. Optional
    columns are ascents, votes, and group_id. The placement query after the
    ``-- placements`` separator must return placement_id, layout_id, set_id, raw_x,
    raw_y, x, and y so every placement can be joined to the audited hold catalog.
    """

    sql = query_file.read_text(encoding="utf-8")
    parts = re.split(
        r"^\s*--\s*placements\s*$", sql, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE
    )
    if len(parts) != 2:
        raise ValueError("Import SQL must contain a '-- placements' separator")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    roles = _role_lookup(connection)
    catalog = _load_hold_catalog(catalog_file)
    placements: dict[str, dict[str, Any]] = {}
    for row in connection.execute(parts[1]):
        key = (
            int(row["layout_id"]),
            int(row["set_id"]),
            int(row["raw_x"]),
            int(row["raw_y"]),
        )
        if key not in catalog:
            raise ValueError(f"Placement {row['placement_id']} is missing from the hold catalog")
        placements[str(row["placement_id"])] = catalog[key] | {
            "x": float(row["x"]),
            "y": float(row["y"]),
        }
    examples: list[RouteExample] = []
    for row in connection.execute(parts[0]):
        if int(row["angle"]) not in SUPPORTED_ANGLES:
            continue
        holds = _parse_frame(str(row["frames"]), roles, placements)
        if len(holds) < 2:
            continue
        raw = dict(row)
        examples.append(
            RouteExample.from_dict(
                {
                    "climb_id": raw["climb_id"],
                    "layout": raw["layout"],
                    "angle": raw["angle"],
                    "grade": raw["grade"],
                    "ascents": raw.get("ascents", 0),
                    "votes": raw.get("votes", 0),
                    "group_id": raw.get("group_id"),
                    "holds": [hold.as_dict() for hold in holds],
                },
                require_grade=True,
            )
        )
    connection.close()
    write_jsonl(examples, output)
    return len(examples)


def import_main() -> None:
    parser = argparse.ArgumentParser(description="Import TB2 routes using an audited SQL mapping")
    parser.add_argument("database", type=Path)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/tb2_12x12_hold_catalog.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/tb2_12x12_shapes.jsonl")
    )
    args = parser.parse_args()
    count = import_with_query(args.database, args.query, args.catalog, args.output)
    print(json.dumps({"examples": count, "output": str(args.output)}))


if __name__ == "__main__":
    inspect_main()
