"""Export everything the browser needs besides the two ONNX graphs.

The frontend has to reproduce the critic's featurization exactly. Where it drifts, predictions
go quietly wrong rather than failing, so this module also emits reference fixtures: real
problems with the tensors and logits PyTorch produced for them. The TypeScript side is tested
against those, which turns the most error-prone part of the project into a checkable one.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch

from ..catalog import DEFAULT_CATALOG_PATH, HoldCatalog
from ..data import ROLE_TO_INDEX, Vocabulary, collate_routes
from ..generator import constraints as generator_constraints
from ..generator.tokenizer import (
    BOS,
    EOS,
    MAX_HOLDS,
    MAX_SEQUENCE_LENGTH,
    PAD,
    PREFIX_LENGTH,
    UNCOND,
    GeneratorVocabulary,
)
from ..grade.model import ModelConfig, TensionGradeTransformer
from ..grade.train import forward_batch
from ..schema import HOLD_ROLES, RouteExample, load_jsonl
from ..style import BUCKET_EDGES, FEATURE_NAMES, FEATURE_SCALES, PRESETS

# Aurora's role colours, so the rendered board matches the app climbers already use.
ROLE_COLORS = {"start": "00DD00", "hand": "0066FF", "finish": "FF0000", "foot": "FF00FF"}

# Normalization lives only in the import SQL, so the inverse lives here.
# raw_x = x * 128 - 64 and raw_y = y * 136 + 4.
COORDINATE_INVERSE = {"x_scale": 128.0, "x_offset": -64.0, "y_scale": 136.0, "y_offset": 4.0}


def board_artifact(catalog: HoldCatalog) -> dict[str, Any]:
    layouts: dict[str, list[dict[str, Any]]] = {}
    for layout in catalog.layouts:
        layouts[layout] = [
            {
                "x": placement.x,
                "y": placement.y,
                "raw_x": round(placement.x * 128.0 - 64.0),
                "raw_y": round(placement.y * 136.0 + 4.0),
                "hold_type": placement.hold_type,
                "orientation_degrees": placement.orientation_degrees,
            }
            for placement in sorted(
                catalog.placements[layout].values(), key=lambda p: (p.y, p.x)
            )
        ]
    return {
        "board": "Tension Board 2 12x12",
        "layouts": layouts,
        "role_colors": ROLE_COLORS,
        "coordinate_inverse": COORDINATE_INVERSE,
    }


def critic_artifact(checkpoint: dict[str, Any], vocabulary: Vocabulary) -> dict[str, Any]:
    return {
        "hold_type_to_index": vocabulary.hold_type_to_index,
        "grade_labels": list(vocabulary.grade_labels),
        "grade_offset": vocabulary.grade_offset,
        "temperature": float(checkpoint.get("temperature", 1.0)),
        "role_to_index": dict(ROLE_TO_INDEX),
        "input_order": [
            "hold_type_ids",
            "orientations",
            "roles",
            "coordinates",
            "mask",
            "angles",
        ],
        "notes": {
            "orientation": "[sin, cos], in that order",
            "coordinates": "passed raw, no rescaling",
            "angle": "passed in raw degrees; the model normalizes internally",
            "unknown_hold_type": "index 0",
            "confidence": "softmax(logits / temperature)",
        },
    }


def generator_artifact(vocabulary: GeneratorVocabulary) -> dict[str, Any]:
    return {
        "vocabulary": vocabulary.as_dict(),
        "size": vocabulary.size,
        "special_tokens": {"pad": PAD, "bos": BOS, "eos": EOS, "uncond": UNCOND},
        "prefix_length": PREFIX_LENGTH,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "max_holds": MAX_HOLDS,
        "roles": list(HOLD_ROLES),
        "constraints": {
            "max_start_height": generator_constraints.MAX_START_HEIGHT,
            "min_finish_height": generator_constraints.MIN_FINISH_HEIGHT,
            "min_holds": generator_constraints.MIN_HOLDS,
        },
    }


def style_artifact() -> dict[str, Any]:
    return {
        "feature_names": list(FEATURE_NAMES),
        "bucket_edges": {name: list(edges) for name, edges in BUCKET_EDGES.items()},
        "feature_scales": dict(FEATURE_SCALES),
        "hand_path_roles": ["start", "hand", "finish"],
        "presets": {
            name: {
                "bounds": {
                    feature: list(bounds) for feature, bounds in preset.bounds.items()
                },
                "conditioning_buckets": list(preset.conditioning_buckets()),
            }
            for name, preset in PRESETS.items()
        },
    }


def parity_fixtures(
    examples: list[RouteExample],
    model: TensionGradeTransformer,
    vocabulary: Vocabulary,
) -> list[dict[str, Any]]:
    """Problems with the exact tensors and logits PyTorch produced for them."""

    fixtures = []
    for example in examples:
        batch = collate_routes([example], vocabulary)
        with torch.no_grad():
            logits = forward_batch(model, batch)
        fixtures.append(
            {
                "problem": example.as_dict(),
                "tensors": {
                    "hold_type_ids": batch["hold_type_ids"][0].tolist(),
                    "orientations": [
                        [round(value, 7) for value in row]
                        for row in batch["orientations"][0].tolist()
                    ],
                    "roles": batch["roles"][0].tolist(),
                    "coordinates": batch["coordinates"][0].tolist(),
                    "mask": [int(value) for value in batch["mask"][0].tolist()],
                    "angles": [batch["angles"][0].item()],
                },
                "logits": [round(value, 6) for value in logits[0].tolist()],
            }
        )
    return fixtures


def select_fixture_examples(path: Path, count: int) -> list[RouteExample]:
    """A spread of hold counts, so the fixtures exercise short and long problems."""

    examples = load_jsonl(path)
    ordered = sorted(examples, key=lambda example: len(example.holds))
    if len(ordered) <= count:
        return ordered
    step = (len(ordered) - 1) / (count - 1)
    return [ordered[round(index * step)] for index in range(count)]


def placement_ids(database: Path, catalog_file: Path) -> dict[str, Any]:
    """Map Aurora placement ids to positions, for the Web Bluetooth frames.

    Only available from the board database: the committed catalog is keyed by raw coordinates
    and carries no placement id.
    """

    import csv

    with Path(catalog_file).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    by_key = {
        (int(row["layout_id"]), int(row["set_id"]), int(row["raw_x"]), int(row["raw_y"])): row
        for row in rows
    }
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    query = """
        SELECT p.layout_id, p.id AS placement_id, p.set_id, h.x AS raw_x, h.y AS raw_y
        FROM placements AS p
        INNER JOIN holes AS h ON h.id = p.hole_id
        WHERE p.layout_id IN (10, 11) AND p.set_id IN (12, 13)
    """
    mapping: dict[str, dict[str, Any]] = {}
    for row in connection.execute(query):
        key = (
            int(row["layout_id"]),
            int(row["set_id"]),
            int(row["raw_x"]),
            int(row["raw_y"]),
        )
        entry = by_key.get(key)
        if entry is None:
            continue
        mapping[str(row["placement_id"])] = {
            "layout": entry["layout"],
            "x": float(entry["x"]),
            "y": float(entry["y"]),
        }
    connection.close()
    return {
        "placements": mapping,
        # Product 4 role ids; product 5 uses 5/6/7/8.
        "role_ids": {"start": 1, "hand": 2, "finish": 3, "foot": 4},
        "frame_format": "p<placement_id>r<role_id>",
    }


def write_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", type=Path, default=Path("checkpoints/grade/tb2_12x12.pt"))
    parser.add_argument(
        "--generator", type=Path, default=Path("checkpoints/generator/tb2_12x12.pt")
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=Path("web/public/data"))
    parser.add_argument(
        "--fixtures-dataset",
        type=Path,
        default=Path("data/processed/tb2_12x12.jsonl"),
        help="Source of the JS/Python parity fixtures",
    )
    parser.add_argument("--fixtures", type=int, default=12)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Aurora database, needed only for the Web Bluetooth placement ids",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = HoldCatalog.load(args.catalog)
    written: dict[str, int] = {}

    written["board.json"] = write_json(args.output / "board.json", board_artifact(catalog))
    written["style.json"] = write_json(args.output / "style.json", style_artifact())

    critic_checkpoint = torch.load(args.critic, map_location="cpu", weights_only=False)
    critic_vocabulary = Vocabulary.from_dict(critic_checkpoint["vocabulary"])
    written["critic.json"] = write_json(
        args.output / "critic.json", critic_artifact(critic_checkpoint, critic_vocabulary)
    )

    if args.generator.exists():
        generator_checkpoint = torch.load(
            args.generator, map_location="cpu", weights_only=False
        )
        written["generator.json"] = write_json(
            args.output / "generator.json",
            generator_artifact(GeneratorVocabulary.from_dict(generator_checkpoint["vocabulary"])),
        )

    if args.fixtures_dataset.exists() and args.fixtures > 0:
        model = TensionGradeTransformer(ModelConfig(**critic_checkpoint["model_config"]))
        model.load_state_dict(critic_checkpoint["model_state"])
        model.eval()
        examples = select_fixture_examples(args.fixtures_dataset, args.fixtures)
        written["fixtures.json"] = write_json(
            args.output / "fixtures.json",
            {
                "comment": "Reference tensors and logits from PyTorch; the web featurizer "
                "must reproduce these.",
                "temperature": float(critic_checkpoint.get("temperature", 1.0)),
                "cases": parity_fixtures(examples, model, critic_vocabulary),
            },
        )

    if args.database is not None:
        written["bluetooth.json"] = write_json(
            args.output / "bluetooth.json", placement_ids(args.database, args.catalog)
        )

    print(json.dumps({"output": str(args.output), "bytes": written}), flush=True)


if __name__ == "__main__":
    main()
