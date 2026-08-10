"""Dataset preparation and leakage-resistant route splitting."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
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
    """Split deterministic, grade-stratified hold configurations as indivisible groups.

    UUIDs are not trusted as group identifiers because community members can save
    the same hold configuration under another name. All angles and exact copies of
    a hold-role configuration therefore remain in one split.
    """

    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must leave non-empty train, validation, and test ranges")
    groups: dict[str, list[RouteExample]] = defaultdict(list)
    for example in examples:
        signature = "|".join(
            f"{hold.placement_id}:{hold.role}"
            for hold in sorted(example.holds, key=lambda hold: (hold.placement_id, hold.role))
        )
        groups[signature].append(example)

    strata: dict[int, list[tuple[str, list[RouteExample]]]] = defaultdict(list)
    for signature, group in groups.items():
        grade_indices = sorted(v_grade_index(example.grade) for example in group if example.grade)
        if not grade_indices:
            raise ValueError("Splitting training data requires grades")
        median_grade = grade_indices[len(grade_indices) // 2]
        strata[median_grade].append((signature, group))

    splits: tuple[list[RouteExample], list[RouteExample], list[RouteExample]] = ([], [], [])
    test_fraction = 1.0 - train_fraction - validation_fraction
    for stratum_groups in strata.values():
        ordered = sorted(
            stratum_groups,
            key=lambda item: hashlib.sha256(item[0].encode()).digest(),
        )
        group_count = len(ordered)
        if group_count < 3:
            validation_count = test_count = 0
        else:
            validation_count = max(1, round(group_count * validation_fraction))
            test_count = max(1, round(group_count * test_fraction))
            while validation_count + test_count >= group_count:
                if validation_count >= test_count and validation_count > 1:
                    validation_count -= 1
                elif test_count > 1:
                    test_count -= 1
                else:
                    break
        training_count = group_count - validation_count - test_count
        boundaries = (training_count, training_count + validation_count)
        for index, (_, group) in enumerate(ordered):
            split_index = 0 if index < boundaries[0] else 1 if index < boundaries[1] else 2
            splits[split_index].extend(group)
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
