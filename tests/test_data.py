import pytest

from tension_board_lab.data import (
    Vocabulary,
    collate_routes,
    select_pretraining_examples,
    split_examples,
)
from tension_board_lab.schema import HoldNode, RouteExample


def example(
    climb_id: str,
    grade: str,
    hold_types: tuple[str, ...],
    *,
    source_layout: str = "mirror",
    grade_value: float | None = None,
    ascents: int = 0,
) -> RouteExample:
    return RouteExample(
        climb_id=climb_id,
        source_layout=source_layout,
        angle=40,
        grade=grade,
        grade_value=grade_value,
        ascents=ascents,
        holds=tuple(
            HoldNode(
                role="hand",
                x=index / 10,
                y=index / 10,
                hold_type=hold_type,
                orientation_degrees=0,
            )
            for index, hold_type in enumerate(hold_types)
        ),
    )


def test_input_equivalent_copies_remain_in_one_split() -> None:
    examples = [
        example(f"ordinary-{index}", f"V{index % 4}", (str(index), str(index + 100)))
        for index in range(80)
    ]
    examples.extend(
        (
            example("mirror-copy", "V5", ("900", "901"), source_layout="mirror"),
            example("spray-copy", "V5", ("900", "901"), source_layout="spray"),
        )
    )
    splits = split_examples(examples)
    locations = {
        split_index
        for split_index, split in enumerate(splits)
        for route in split
        if route.climb_id in {"mirror-copy", "spray-copy"}
    }
    assert len(locations) == 1


def test_split_is_deterministic() -> None:
    examples = [
        example(f"climb-{index}", f"V{index % 6}", (str(index), str(index + 100)))
        for index in range(120)
    ]
    first = [[route.climb_id for route in split] for split in split_examples(examples)]
    second = [[route.climb_id for route in split] for split in split_examples(examples)]
    assert first == second


def test_mirrored_copies_remain_in_one_split() -> None:
    examples = [
        example(f"ordinary-{index}", f"V{index % 4}", (str(index), str(index + 100)))
        for index in range(80)
    ]
    left = RouteExample(
        climb_id="left",
        source_layout="mirror",
        angle=40,
        grade="V5",
        holds=(
            HoldNode("start", 0.2, 0.1, "plastic:10", 45),
            HoldNode("finish", 0.7, 0.9, "plastic:20", 315),
        ),
    )
    right = RouteExample(
        climb_id="right",
        source_layout="mirror",
        angle=45,
        grade="V5",
        holds=(
            HoldNode("start", 0.8, 0.1, "plastic:10", 315),
            HoldNode("finish", 0.3, 0.9, "plastic:20", 45),
        ),
    )
    splits = split_examples([*examples, left, right])
    locations = {
        split_index
        for split_index, split in enumerate(splits)
        for route in split
        if route.climb_id in {"left", "right"}
    }
    assert len(locations) == 1


def test_pretraining_softens_low_evidence_and_excludes_holdouts() -> None:
    vocabulary = Vocabulary.build(
        [
            example("grade-min", "V0", ("0", "100")),
            example("grade-max", "V8", ("8", "108")),
        ]
    )
    low_evidence = example(
        "low-evidence",
        "V5",
        ("500", "501"),
        grade_value=5.25,
        ascents=1,
    )
    excluded = example("excluded", "V5", ("900", "901"), grade_value=5.0, ascents=3)
    excluded_copy = example(
        "excluded-copy",
        "V5",
        ("900", "901"),
        grade_value=5.0,
        ascents=1,
    )

    selected = select_pretraining_examples(
        [low_evidence, excluded_copy], vocabulary, [excluded]
    )
    assert selected == [low_evidence]
    batch = collate_routes(selected, vocabulary, uncertain_targets=True)
    assert batch["target_spreads"].item() == pytest.approx(1.0)
    assert batch["weights"].item() < 0.5
