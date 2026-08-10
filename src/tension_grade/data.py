"""Dataset preparation and leakage-resistant route splitting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .grades import v_grade_index, v_grade_span
from .schema import HOLD_ROLES, SUPPORTED_ANGLES, RouteExample

ROLE_TO_INDEX = {role: index for index, role in enumerate(HOLD_ROLES)}
ANGLE_TO_INDEX = {angle: index for index, angle in enumerate(sorted(SUPPORTED_ANGLES))}


@dataclass(frozen=True)
class Vocabulary:
    placement_to_index: dict[str, int]
    grade_labels: tuple[str, ...]
    grade_offset: int

    @classmethod
    def build(cls, examples: Sequence[RouteExample]) -> Vocabulary:
        placements = sorted({hold.placement_id for route in examples for hold in route.holds})
        placement_to_index = {placement: index + 1 for index, placement in enumerate(placements)}
        grade_indices = [v_grade_index(route.grade) for route in examples if route.grade]
        if not grade_indices:
            raise ValueError("Cannot build a training vocabulary without grades")
        grade_offset = min(grade_indices)
        labels = v_grade_span(grade_offset, max(grade_indices))
        return cls(placement_to_index, labels, grade_offset)

    def as_dict(self) -> dict[str, object]:
        return {
            "placement_to_index": self.placement_to_index,
            "grade_labels": list(self.grade_labels),
            "grade_offset": self.grade_offset,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Vocabulary:
        return cls(
            placement_to_index={str(k): int(v) for k, v in dict(raw["placement_to_index"]).items()},
            grade_labels=tuple(str(value) for value in raw["grade_labels"]),
            grade_offset=int(raw["grade_offset"]),
        )


class RouteDataset(Dataset[RouteExample]):
    def __init__(self, examples: Sequence[RouteExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> RouteExample:
        return self.examples[index]


def split_examples(
    examples: Sequence[RouteExample],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[RouteExample], list[RouteExample], list[RouteExample]]:
    """Split deterministically by group, keeping mirrors/variants together."""

    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must leave non-empty train, validation, and test ranges")
    splits: tuple[list[RouteExample], list[RouteExample], list[RouteExample]] = ([], [], [])
    train_cutoff = int(train_fraction * 10_000)
    validation_cutoff = int((train_fraction + validation_fraction) * 10_000)
    for example in examples:
        group = example.group_id or example.climb_id
        bucket = int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % 10_000
        split_index = 0 if bucket < train_cutoff else 1 if bucket < validation_cutoff else 2
        splits[split_index].append(example)
    if any(not split for split in splits):
        raise ValueError("A split is empty; provide more climbs or change the split fractions")
    return splits


def _sample_weight(example: RouteExample) -> float:
    evidence = max(example.votes, example.ascents)
    return min(3.0, 1.0 + math.log1p(evidence) / 5.0)


def collate_routes(batch: Sequence[RouteExample], vocabulary: Vocabulary) -> dict[str, Tensor]:
    max_holds = max(len(example.holds) for example in batch)
    batch_size = len(batch)
    placement_ids = torch.zeros((batch_size, max_holds), dtype=torch.long)
    roles = torch.zeros((batch_size, max_holds), dtype=torch.long)
    coordinates = torch.zeros((batch_size, max_holds, 2), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_holds), dtype=torch.bool)
    angles = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.full((batch_size,), -1, dtype=torch.long)
    weights = torch.ones(batch_size, dtype=torch.float32)

    for row, example in enumerate(batch):
        angles[row] = ANGLE_TO_INDEX[example.angle]
        weights[row] = _sample_weight(example)
        if example.grade is not None:
            labels[row] = v_grade_index(example.grade) - vocabulary.grade_offset
        for column, hold in enumerate(example.holds):
            placement_ids[row, column] = vocabulary.placement_to_index.get(hold.placement_id, 0)
            roles[row, column] = ROLE_TO_INDEX[hold.role]
            coordinates[row, column] = torch.tensor((hold.x, hold.y), dtype=torch.float32)
            mask[row, column] = True

    return {
        "placement_ids": placement_ids,
        "roles": roles,
        "coordinates": coordinates,
        "mask": mask,
        "angles": angles,
        "labels": labels,
        "weights": weights,
    }
