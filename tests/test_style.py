import math

import pytest

from tension_board_lab.schema import HoldNode, RouteExample
from tension_board_lab.style import (
    BUCKET_EDGES,
    FEATURE_NAMES,
    PRESETS,
    bucket_count,
    compute_style_features,
    hand_path,
    move_lengths,
    style_buckets,
    style_distance,
)


def problem(*holds: tuple[str, float, float]) -> RouteExample:
    return RouteExample(
        climb_id="styled",
        source_layout="mirror",
        angle=40,
        grade="V5",
        holds=tuple(
            HoldNode(role=role, x=x, y=y, hold_type="wood:FT", orientation_degrees=0.0)
            for role, x, y in holds
        ),
    )


# A ladder with two 0.3-high hand moves, one foot, spanning y 0.1 to 0.7.
LADDER = problem(
    ("start", 0.5, 0.1),
    ("hand", 0.5, 0.4),
    ("finish", 0.5, 0.7),
    ("foot", 0.2, 0.2),
)


def test_features_match_hand_computed_values() -> None:
    features = compute_style_features(LADDER)
    assert features.hand_count == 1.0
    assert features.foot_count == 1.0
    assert features.mean_move_length == pytest.approx(0.3)
    assert features.max_move_length == pytest.approx(0.3)
    assert features.move_length_variance == pytest.approx(0.0)
    assert features.height_span == pytest.approx(0.6)
    # One foot over a three-hold hand path.
    assert features.foot_to_hand_ratio == pytest.approx(1 / 3)


def test_hand_path_excludes_feet_and_runs_bottom_to_top() -> None:
    path = hand_path(LADDER.holds)
    assert [hold.role for hold in path] == ["start", "hand", "finish"]
    assert [hold.y for hold in path] == [0.1, 0.4, 0.7]


def test_move_lengths_are_euclidean_between_consecutive_hand_holds() -> None:
    diagonal = problem(("start", 0.1, 0.1), ("finish", 0.4, 0.5))
    assert move_lengths(diagonal.holds) == pytest.approx((math.dist((0.1, 0.1), (0.4, 0.5)),))


def test_variance_reflects_uneven_moves() -> None:
    even = problem(("start", 0.5, 0.0), ("hand", 0.5, 0.2), ("finish", 0.5, 0.4))
    uneven = problem(("start", 0.5, 0.0), ("hand", 0.5, 0.05), ("finish", 0.5, 0.4))
    assert compute_style_features(even).move_length_variance == pytest.approx(0.0)
    assert compute_style_features(uneven).move_length_variance > 0.0


def test_features_ignore_hold_type_and_orientation() -> None:
    other = RouteExample(
        climb_id="styled",
        source_layout="mirror",
        angle=40,
        grade="V5",
        holds=tuple(
            HoldNode(
                role=hold.role,
                x=hold.x,
                y=hold.y,
                hold_type="plastic:99",
                orientation_degrees=315.0,
            )
            for hold in LADDER.holds
        ),
    )
    assert compute_style_features(other) == compute_style_features(LADDER)


def test_bucket_edges_are_strictly_increasing() -> None:
    for feature in FEATURE_NAMES:
        edges = BUCKET_EDGES[feature]
        assert list(edges) == sorted(set(edges)), feature


def test_buckets_cover_every_slot_and_stay_in_range() -> None:
    buckets = style_buckets(compute_style_features(LADDER))
    assert len(buckets) == len(FEATURE_NAMES)
    for feature, bucket in zip(FEATURE_NAMES, buckets):
        assert 0 <= bucket < bucket_count(feature)


def test_bucket_index_rises_with_the_value() -> None:
    short = problem(("start", 0.5, 0.0), ("finish", 0.5, 0.1))
    long = problem(("start", 0.5, 0.0), ("finish", 0.5, 0.9))
    index = FEATURE_NAMES.index("max_move_length")
    short_bucket = style_buckets(compute_style_features(short))[index]
    long_bucket = style_buckets(compute_style_features(long))[index]
    assert short_bucket < long_bucket


def test_preset_distance_is_zero_inside_the_range_and_grows_outside() -> None:
    # Four big hand moves: power wants at most 4 hand holds and long moves.
    powerful = problem(
        ("start", 0.5, 0.0),
        ("hand", 0.2, 0.35),
        ("hand", 0.8, 0.7),
        ("finish", 0.5, 1.0),
    )
    assert style_distance(compute_style_features(powerful), "power") == 0.0

    crowded = problem(
        ("start", 0.5, 0.0),
        *[("hand", 0.5, 0.02 * step) for step in range(1, 10)],
        ("finish", 0.5, 0.3),
    )
    crowded_distance = style_distance(compute_style_features(crowded), "power")
    assert crowded_distance > 0.0

    # Further outside the range means a larger distance.
    tiny = problem(("start", 0.5, 0.0), ("finish", 0.5, 0.05))
    assert style_distance(compute_style_features(tiny), "power") > 0.0


def test_matches_agrees_with_zero_distance() -> None:
    features = compute_style_features(LADDER)
    for preset in PRESETS.values():
        assert preset.matches(features) == (preset.distance(features) == 0.0)


def test_conditioning_buckets_are_set_only_for_constrained_features() -> None:
    for preset in PRESETS.values():
        buckets = preset.conditioning_buckets()
        assert len(buckets) == len(FEATURE_NAMES)
        for feature, bucket in zip(FEATURE_NAMES, buckets):
            if feature in preset.bounds:
                assert bucket is not None and 0 <= bucket < bucket_count(feature)
            else:
                assert bucket is None


def test_features_require_a_hand_hold() -> None:
    with pytest.raises(ValueError):
        compute_style_features(problem(("foot", 0.2, 0.2), ("foot", 0.8, 0.2)))
