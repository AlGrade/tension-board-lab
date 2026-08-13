"""Train the conditional autoregressive boulder generator."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..catalog import DEFAULT_CATALOG_PATH, HoldCatalog
from ..data import configuration_signature, sample_weight, split_examples
from ..schema import HoldNode, RouteExample, load_jsonl
from .model import BoulderGenerator, GeneratorConfig, sequence_loss
from .physical import physical_tables
from .tokenizer import (
    MAX_SEQUENCE_LENGTH,
    PAD,
    PREFIX_LENGTH,
    UNCOND,
    GeneratorVocabulary,
    canonical_holds,
    encode,
)


@dataclass(frozen=True)
class EncodedProblem:
    """A problem tokenized once; only its prefix is re-randomized between epochs."""

    tokens: tuple[int, ...]
    weight: float
    layout: int


def mirrored(example: RouteExample) -> RouteExample:
    """Reflect a problem left to right.

    Only valid on the mirror layout, which is exactly symmetric: all 498 positions have a
    partner carrying the same hold type. Matches the reflection ``configuration_signature``
    already applies, so a reflected copy keeps its original's signature and cannot cross a
    split boundary.
    """

    return RouteExample(
        climb_id=f"{example.climb_id}:mirrored",
        angle=example.angle,
        source_layout=example.source_layout,
        grade=example.grade,
        grade_value=example.grade_value,
        ascents=example.ascents,
        holds=tuple(
            HoldNode(
                role=hold.role,
                x=round(1.0 - hold.x, 6),
                y=hold.y,
                hold_type=hold.hold_type,
                orientation_degrees=(-hold.orientation_degrees) % 360.0,
            )
            for hold in example.holds
        ),
    )


def augment(
    examples: Sequence[RouteExample],
    catalog: HoldCatalog,
    excluded: Sequence[RouteExample],
) -> list[RouteExample]:
    """Add the mirror image of every mirror-layout problem; 65% more training data.

    A reflected copy does not always keep its original's signature. ``encode`` re-reads hold
    type and orientation from the catalog, and 17 of the 498 mirror positions are
    orientation-asymmetric, so 29.7% of copies canonicalize to a different configuration than
    the problem they came from. That breaks the assumption that a copy inherits its original's
    split, so every copy is checked against the held-out signatures directly. On current data
    nothing is dropped, but the guard is what makes the invariant true rather than lucky.
    """

    holdout = {configuration_signature(example) for example in excluded}
    grown = list(examples)
    for example in examples:
        if example.source_layout != "mirror":
            continue
        copy = mirrored(example)
        if not all(catalog.contains("mirror", hold.x, hold.y) for hold in copy.holds):
            continue
        # Compare what training actually sees, after the catalog overrides type and orientation.
        canonical = RouteExample(
            climb_id=copy.climb_id,
            angle=copy.angle,
            source_layout=copy.source_layout,
            grade=copy.grade,
            holds=canonical_holds(copy, catalog),
        )
        if configuration_signature(canonical) in holdout:
            continue
        grown.append(copy)
    return grown


class SequenceDataset(Dataset[EncodedProblem]):
    def __init__(self, problems: Sequence[EncodedProblem]) -> None:
        self.problems = list(problems)

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> EncodedProblem:
        return self.problems[index]


def encode_dataset(
    examples: Sequence[RouteExample],
    vocabulary: GeneratorVocabulary,
    catalog: HoldCatalog,
) -> list[EncodedProblem]:
    return [
        EncodedProblem(
            tokens=encode(example, vocabulary, catalog),
            weight=sample_weight(example),
            layout=vocabulary.layouts.index(example.source_layout),
        )
        for example in examples
    ]


def randomize_prefix(tokens: list[int], *, guidance_dropout: float) -> None:
    """Blank the conditioning in place, for classifier-free guidance.

    Dropping the whole prefix some of the time is what gives sampling a guidance scale to
    trade grade fidelity against variety.
    """

    if random.random() < guidance_dropout:
        tokens[1 : 1 + PREFIX_LENGTH] = [UNCOND] * PREFIX_LENGTH


def collate_sequences(
    batch: Sequence[EncodedProblem],
    *,
    guidance_dropout: float = 0.0,
) -> dict[str, Tensor]:
    """Pad to the longest sequence and build shifted next-token targets.

    The conditioning prefix is an input, not something to predict, so its target positions are
    set to PAD and drop out of the loss.
    """

    width = max(len(problem.tokens) for problem in batch)
    inputs = torch.full((len(batch), width - 1), PAD, dtype=torch.long)
    targets = torch.full((len(batch), width - 1), PAD, dtype=torch.long)
    weights = torch.ones(len(batch), dtype=torch.float32)
    layouts = torch.zeros(len(batch), dtype=torch.long)

    for row, problem in enumerate(batch):
        tokens = list(problem.tokens)
        randomize_prefix(tokens, guidance_dropout=guidance_dropout)
        length = len(tokens) - 1
        inputs[row, :length] = torch.tensor(tokens[:-1], dtype=torch.long)
        shifted = torch.tensor(tokens[1:], dtype=torch.long)
        shifted[:PREFIX_LENGTH] = PAD
        targets[row, :length] = shifted
        weights[row] = problem.weight
        layouts[row] = problem.layout

    return {"inputs": inputs, "targets": targets, "weights": weights, "layouts": layouts}


def select_generator_examples(
    pretraining_examples: Sequence[RouteExample],
    excluded_examples: Sequence[RouteExample],
) -> list[RouteExample]:
    """Drop every configuration the critic holds out, so its evaluation stays honest.

    Uses the critic's own ``configuration_signature``, which folds together renamed, mirrored,
    and input-equivalent problems across all angles.
    """

    excluded = {configuration_signature(example) for example in excluded_examples}
    return [
        example
        for example in pretraining_examples
        if example.source_layout is not None
        and example.grade is not None
        and configuration_signature(example) not in excluded
    ]


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


def train_epoch(
    model: BoulderGenerator,
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
        logits = model(batch["inputs"], batch["layouts"])
        loss = sequence_loss(logits, batch["targets"], batch["weights"], PAD)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        count = len(batch["weights"])
        running_loss += float(loss.item()) * count
        seen += count
    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: BoulderGenerator,
    loader: Iterable[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Unweighted negative log-likelihood per token and per problem."""

    model.eval()
    total_tokens = 0.0
    total_token_loss = 0.0
    total_sequence_loss = 0.0
    problems = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch["inputs"], batch["layouts"])
        targets = batch["targets"]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=PAD,
            reduction="none",
        ).view(targets.shape)
        counted = (targets != PAD).to(losses.dtype)
        total_token_loss += float((losses * counted).sum().item())
        total_tokens += float(counted.sum().item())
        total_sequence_loss += float((losses * counted).sum(dim=-1).sum().item())
        problems += targets.shape[0]
    return {
        "nll_per_token": total_token_loss / max(total_tokens, 1.0),
        "nll_per_problem": total_sequence_loss / max(problems, 1),
        "perplexity": float(np.exp(total_token_loss / max(total_tokens, 1.0))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Pretraining JSONL, at least one ascent")
    parser.add_argument(
        "--consensus-dataset",
        type=Path,
        required=True,
        help="Consensus JSONL, used only to reproduce the critic's held-out configurations",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/generator/tb2_12x12.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--guidance-dropout", type=float, default=0.1)
    parser.add_argument("--max-batches", type=int, default=0, help="Debug: cap batches per epoch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    catalog = HoldCatalog.load(args.catalog)
    vocabulary = GeneratorVocabulary.build(catalog)

    _, critic_validation, critic_test = split_examples(load_jsonl(args.consensus_dataset))
    pool = select_generator_examples(
        load_jsonl(args.dataset), critic_validation + critic_test
    )
    if not pool:
        raise ValueError("No leakage-free generator examples remain")
    train_examples, validation_examples, test_examples = split_examples(pool)
    # Augment only the training split; a reflected copy would otherwise inflate the holdouts.
    augmented = augment(
        train_examples,
        catalog,
        critic_validation + critic_test + validation_examples + test_examples,
    )
    print(
        json.dumps(
            {
                "phase": "setup",
                "vocabulary": vocabulary.size,
                "pool": len(pool),
                "excluded_critic_groups": len(
                    {configuration_signature(e) for e in critic_validation + critic_test}
                ),
                "train": len(train_examples),
                "train_after_mirror_augmentation": len(augmented),
                "validation": len(validation_examples),
                "test": len(test_examples),
            }
        ),
        flush=True,
    )

    collate = partial(collate_sequences, guidance_dropout=args.guidance_dropout)
    evaluation_collate = collate_sequences  # no dropout when measuring
    train_loader = DataLoader(
        SequenceDataset(encode_dataset(augmented, vocabulary, catalog)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        SequenceDataset(encode_dataset(validation_examples, vocabulary, catalog)),
        batch_size=args.batch_size,
        collate_fn=evaluation_collate,
    )
    test_loader = DataLoader(
        SequenceDataset(encode_dataset(test_examples, vocabulary, catalog)),
        batch_size=args.batch_size,
        collate_fn=evaluation_collate,
    )

    type_ids, orientations, hold_types = physical_tables(vocabulary, catalog)
    config = GeneratorConfig(
        vocabulary_size=vocabulary.size,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        num_hold_types=len(hold_types),
        num_layouts=len(vocabulary.layouts),
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    )
    device = select_device(args.device)
    model = BoulderGenerator(
        config,
        hold_type_ids=torch.tensor(type_ids, dtype=torch.long),
        hold_orientations=torch.tensor(orientations, dtype=torch.float32),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_nll = float("inf")
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    stopped_epoch = 0
    for epoch in range(1, args.epochs + 1):
        stopped_epoch = epoch
        loader = _limited(train_loader, args.max_batches)
        training_loss = train_epoch(model, loader, optimizer, device)
        scheduler.step()
        validation_metrics = evaluate(model, validation_loader, device)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": training_loss,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
            ),
            flush=True,
        )
        if validation_metrics["nll_per_token"] < best_nll:
            best_nll = validation_metrics["nll_per_token"]
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
    test_metrics = evaluate(model, test_loader, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": best_state,
            "model_config": config.as_dict(),
            "vocabulary": vocabulary.as_dict(),
            "hold_type_to_index": hold_types,
            "test_metrics": test_metrics,
            "selected_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "scope": {
                "board": "Tension Board 2",
                "training_layouts": ["Mirror", "Spray"],
                "size": "12x12",
                "angles": [35, 40, 45, 50, 55],
                "grade_scale": "V",
                "training_pipeline": "next-token prediction with classifier-free guidance",
                "conditioning": ["layout", "angle", "grade"],
                "held_out": "the grade critic's validation and test configurations",
            },
            "training_args": vars(args)
            | {
                "dataset": str(args.dataset),
                "consensus_dataset": str(args.consensus_dataset),
                "catalog": str(args.catalog),
                "output": str(args.output),
            },
            "data": {
                "pool": len(pool),
                "train": len(train_examples),
                "train_after_mirror_augmentation": len(augmented),
                "validation": len(validation_examples),
                "test": len(test_examples),
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
                **test_metrics,
            }
        )
    )


def _limited(loader: Iterable[dict[str, Tensor]], limit: int) -> Iterable[dict[str, Tensor]]:
    if limit <= 0:
        return loader
    return (batch for index, batch in enumerate(loader) if index < limit)


if __name__ == "__main__":
    main()
