from tension_grade.data import split_examples
from tension_grade.schema import HoldNode, RouteExample


def example(climb_id: str, grade: str, placements: tuple[str, ...]) -> RouteExample:
    return RouteExample(
        climb_id=climb_id,
        angle=40,
        grade=grade,
        holds=tuple(
            HoldNode(placement, "hand", index / 10, index / 10)
            for index, placement in enumerate(placements)
        ),
    )


def test_exact_copies_with_different_uuids_remain_in_one_split() -> None:
    examples = [
        example(f"ordinary-{index}", f"V{index % 4}", (str(index), str(index + 100)))
        for index in range(80)
    ]
    examples.extend(
        [
            example("copy-one", "V5", ("900", "901")),
            example("copy-two", "V5", ("900", "901")),
        ]
    )
    splits = split_examples(examples)
    copy_locations = {
        split_index
        for split_index, split in enumerate(splits)
        for route in split
        if route.climb_id.startswith("copy-")
    }
    assert len(copy_locations) == 1


def test_split_is_deterministic() -> None:
    examples = [
        example(f"climb-{index}", f"V{index % 6}", (str(index), str(index + 100)))
        for index in range(120)
    ]
    first = [[route.climb_id for route in split] for split in split_examples(examples)]
    second = [[route.climb_id for route in split] for split in split_examples(examples)]
    assert first == second
