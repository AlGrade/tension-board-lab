"""Train the canonical Essential-input TB2 grade predictor."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from ..data import (
    RouteDataset,
    Vocabulary,
    collate_routes,
    configuration_signature,
    select_pretraining_examples,
    split_examples,
)
from ..schema import RouteExample, load_jsonl
from .model import ModelConfig, TensionGradeTransformer, grade_loss, probabilities


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def forward_batch(model: TensionGradeTransformer, batch: dict[str, Tensor]) -> Tensor:
    return model(
        batch["hold_type_ids"],
        batch["orientations"],
        batch["roles"],
        batch["coordinates"],
        batch["mask"],
        batch["angles"],
    )


@torch.no_grad()
def collect_logits(
    model: TensionGradeTransformer,
    loader: Iterable[dict[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    model.eval()
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        all_logits.append(forward_batch(model, batch).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def calibrate_temperature(logits: Tensor, labels: Tensor) -> float:
    candidates = torch.linspace(0.5, 3.0, 251)
    losses = torch.stack([F.cross_entropy(logits / value, labels) for value in candidates])
    return float(candidates[losses.argmin()].item())


def metrics(logits: Tensor, labels: Tensor, temperature: float = 1.0) -> dict[str, float]:
    predicted = probabilities(logits, temperature).argmax(dim=-1)
    distance = (predicted - labels).abs().float()
    return {
        "mae_grades": float(distance.mean().item()),
        "within_one_grade": float((distance <= 1).float().mean().item()),
        "accuracy": float((distance == 0).float().mean().item()),
    }


def breakdown_metrics(
    logits: Tensor,
    labels: Tensor,
    examples: Sequence[RouteExample],
    temperature: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {"layout": {}, "angle": {}}
    fields = (("layout", "source_layout"), ("angle", "angle"))
    for output_name, field in fields:
        values = [str(getattr(example, field)) for example in examples]
        for value in sorted(set(values)):
            indices = torch.tensor([index for index, item in enumerate(values) if item == value])
            result[output_name][value] = metrics(logits[indices], labels[indices], temperature) | {
                "examples": len(indices)
            }
    return result


def train_epoch(
    model: TensionGradeTransformer,
    loader: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    seen = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_batch(model, batch)
        loss = grade_loss(
            logits,
            batch["target_values"],
            batch["weights"],
            target_spreads=batch["target_spreads"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item()) * len(batch["labels"])
        seen += len(batch["labels"])
    return running_loss / max(seen, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Canonical JSONL dataset")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/grade/tb2_12x12.pt"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--pretrain-dataset", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = load_jsonl(args.dataset)
    if any(example.grade_value is None for example in examples):
        raise ValueError("Training examples require an unrounded community grade value")
    train_examples, validation_examples, test_examples = split_examples(examples)
    vocabulary = Vocabulary.build(train_examples)
    for example in validation_examples + test_examples:
        if example.grade not in vocabulary.grade_labels:
            raise ValueError(
                f"Grade {example.grade} appears outside the training grade span; collect more data"
            )

    collate = lambda batch: collate_routes(batch, vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        RouteDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        RouteDataset(validation_examples), batch_size=args.batch_size, collate_fn=collate
    )
    test_loader = DataLoader(
        RouteDataset(test_examples), batch_size=args.batch_size, collate_fn=collate
    )

    config = ModelConfig(
        num_hold_types=len(vocabulary.hold_type_to_index),
        num_grades=len(vocabulary.grade_labels),
        width=args.width,
        heads=args.heads,
        layers=args.layers,
    )
    device = select_device(args.device)
    model = TensionGradeTransformer(config).to(device)

    excluded_examples = validation_examples + test_examples
    excluded_signatures = {configuration_signature(example) for example in excluded_examples}
    pretraining_examples = select_pretraining_examples(
        load_jsonl(args.pretrain_dataset), vocabulary, excluded_examples
    )
    if not pretraining_examples:
        raise ValueError("No leakage-free pretraining examples remain")
    pretrain_collate = lambda batch: collate_routes(batch, vocabulary, uncertain_targets=True)
    pretrain_loader = DataLoader(
        RouteDataset(pretraining_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pretrain_collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pretrain_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pretrain_optimizer, T_max=args.pretrain_epochs
    )
    evidence_counts = {
        threshold: sum(example.ascents == threshold for example in pretraining_examples)
        for threshold in (1, 2)
    }
    evidence_counts[3] = sum(example.ascents >= 3 for example in pretraining_examples)
    print(
        json.dumps(
            {
                "phase": "pretrain_setup",
                "examples": len(pretraining_examples),
                "ascents_1": evidence_counts[1],
                "ascents_2": evidence_counts[2],
                "ascents_3_plus": evidence_counts[3],
                "excluded_validation_test_groups": len(excluded_signatures),
            }
        )
    )
    for epoch in range(1, args.pretrain_epochs + 1):
        pretrain_loss = train_epoch(model, pretrain_loader, pretrain_optimizer, device)
        pretrain_scheduler.step()
        print(json.dumps({"phase": "pretrain", "epoch": epoch, "loss": pretrain_loss}))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_validation_mae = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    stopped_epoch = 0
    for epoch in range(1, args.epochs + 1):
        stopped_epoch = epoch
        running_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        validation_logits, validation_labels = collect_logits(model, validation_loader, device)
        validation_metrics = metrics(validation_logits, validation_labels)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": running_loss,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
            )
        )
        if validation_metrics["mae_grades"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae_grades"]
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    validation_logits, validation_labels = collect_logits(model, validation_loader, device)
    temperature = calibrate_temperature(validation_logits, validation_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_metrics = metrics(test_logits, test_labels, temperature)
    test_breakdown = breakdown_metrics(test_logits, test_labels, test_examples, temperature)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "model_state": best_state,
            "model_config": config.as_dict(),
            "vocabulary": vocabulary.as_dict(),
            "temperature": temperature,
            "test_metrics": test_metrics,
            "test_breakdown": test_breakdown,
            "selected_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "scope": {
                "board": "Tension Board 2",
                "training_layouts": ["Mirror", "Spray"],
                "size": "12x12",
                "angles": [35, 40, 45, 50, 55],
                "grade_scale": "V",
                "training_pipeline": "uncertain pretraining followed by consensus fine-tuning",
                "training_target": "unrounded community average mapped to soft V-grade labels",
                "model_inputs": [
                    "hold_type",
                    "orientation",
                    "coordinates",
                    "route_role",
                    "continuous_wall_angle",
                ],
            },
            "training_args": vars(args)
            | {
                "dataset": str(args.dataset),
                "pretrain_dataset": str(args.pretrain_dataset),
                "output": str(args.output),
            },
            "pretraining": {
                "examples": len(pretraining_examples),
                "ascents_1": evidence_counts[1],
                "ascents_2": evidence_counts[2],
                "ascents_3_plus": evidence_counts[3],
                "excluded_validation_test_groups": len(excluded_signatures),
                "epochs": args.pretrain_epochs,
            },
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "selected_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "temperature": temperature,
                **test_metrics,
                "breakdown": test_breakdown,
            }
        )
    )


if __name__ == "__main__":
    main()
