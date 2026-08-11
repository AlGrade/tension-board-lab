from tension_grade.data import split_examples
from tension_grade.schema import HoldNode, RouteExample


def example(
    climb_id: str,
    grade: str,
    placements: tuple[str, ...],
    *,
    layout: str = "mirror",
) -> RouteExample:
    return RouteExample(
        climb_id=climb_id,
        layout=layout,
        angle=40,
        grade=grade,
        holds=tuple(
            HoldNode(
                placement_id=placement,
                role="hand",
                x=index / 10,
                y=index / 10,
                hold_id=placement,
                hold_family=f"wood:{placement}",
                variant="none",
                material="wood",
                orientation_degrees=0,
            )
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


def test_mirrored_copies_remain_in_one_split() -> None:
    examples = [
        example(f"ordinary-{index}", f"V{index % 4}", (str(index), str(index + 100)))
        for index in range(80)
    ]
    left = RouteExample(
        climb_id="left",
        layout="mirror",
        angle=40,
        grade="V5",
        holds=(
            HoldNode("a", "start", 0.2, 0.1, "10l", "plastic:10", "l", "plastic", 45),
            HoldNode("b", "finish", 0.7, 0.9, "20r", "plastic:20", "r", "plastic", 315),
        ),
    )
    right = RouteExample(
        climb_id="right",
        layout="mirror",
        angle=45,
        grade="V5",
        holds=(
            HoldNode("c", "start", 0.8, 0.1, "10r", "plastic:10", "r", "plastic", 315),
            HoldNode("d", "finish", 0.3, 0.9, "20l", "plastic:20", "l", "plastic", 45),
        ),
    )
    splits = split_examples(examples + [left, right])
    locations = {
        split_index
        for split_index, split in enumerate(splits)
        for route in split
        if route.climb_id in {"left", "right"}
    }
    assert len(locations) == 1


def test_layout_is_excluded_from_groups_when_it_is_not_a_model_input() -> None:
    examples = [
        example(f"ordinary-{index}", f"V{index % 4}", (str(index), str(index + 100)))
        for index in range(80)
    ]
    examples.extend(
        (
            example("mirror-copy", "V5", ("900", "901"), layout="mirror"),
            example("spray-copy", "V5", ("900", "901"), layout="spray"),
        )
    )
    splits = split_examples(examples, include_layout=False)
    locations = {
        split_index
        for split_index, split in enumerate(splits)
        for route in split
        if route.climb_id in {"mirror-copy", "spray-copy"}
    }
    assert len(locations) == 1
