"""Predict a V grade and calibrated confidence from one canonical route JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import Vocabulary, collate_routes
from .model import ModelConfig, TensionGradeTransformer, probabilities
from .schema import RouteExample
from .train import forward_batch, move_batch, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("route", type=Path, help="JSON route without a grade field")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary.from_dict(checkpoint["vocabulary"])
    config = ModelConfig(**checkpoint["model_config"])
    model = TensionGradeTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    with args.route.open(encoding="utf-8") as source:
        route = RouteExample.from_dict(json.load(source), require_grade=False)
    batch = move_batch(collate_routes([route], vocabulary), device)
    with torch.no_grad():
        distribution = probabilities(
            forward_batch(model, batch), float(checkpoint.get("temperature", 1.0))
        )[0]
    predicted_index = int(distribution.argmax().item())
    result = {
        "predicted_grade": vocabulary.grade_labels[predicted_index],
        "confidence": round(float(distribution[predicted_index].item()), 4),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
