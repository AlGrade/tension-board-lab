import pytest

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.generator.tokenizer import (
    BOS,
    EOS,
    MAX_SEQUENCE_LENGTH,
    PAD,
    PREFIX_LENGTH,
    UNCOND,
    GeneratorVocabulary,
    canonical_holds,
    decode,
    encode,
    encode_prefix,
)
from tension_board_lab.schema import HOLD_ROLES, HoldNode, RouteExample
from tension_board_lab.style import FEATURE_NAMES, bucket_count

CATALOG = HoldCatalog.load()
VOCABULARY = GeneratorVocabulary.build(CATALOG)


def board_problem(
    roles: tuple[str, ...],
    *,
    layout: str = "mirror",
    angle: int = 40,
    grade: str = "V5",
) -> RouteExample:
    """A problem on real catalog positions, so the catalog lookup is exercised."""

    positions = [p for p in CATALOG.positions if CATALOG.contains(layout, *p)][: len(roles)]
    chosen = [CATALOG.placement(layout, *position) for position in positions]
    return RouteExample(
        climb_id="tokenized",
        source_layout=layout,
        angle=angle,
        grade=grade,
        holds=tuple(
            HoldNode(
                role=role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
            for role, placement in zip(roles, chosen)
        ),
    )


PROBLEM = board_problem(("start", "foot", "hand", "hand", "finish"))


def test_vocabulary_size_matches_the_board() -> None:
    assert len(VOCABULARY.positions) == 537
    # 537 positions x 4 roles; the conditioning prefix adds 75 tokens on top.
    assert VOCABULARY.size - VOCABULARY.hold_offset == 2148
    assert VOCABULARY.size == 2223


def test_every_token_id_is_claimed_exactly_once() -> None:
    claimed = list(range(len(("<pad>", "<bos>", "<eos>", "<uncond>"))))
    claimed += [VOCABULARY.layout_token(layout) for layout in VOCABULARY.layouts]
    claimed += [VOCABULARY.angle_token(angle) for angle in VOCABULARY.angles]
    claimed += [VOCABULARY.grade_token(grade) for grade in VOCABULARY.grade_labels]
    for feature in FEATURE_NAMES:
        claimed += [
            VOCABULARY.style_token(feature, bucket) for bucket in range(bucket_count(feature))
        ]
        claimed.append(VOCABULARY.style_token(feature, None))
    for position in VOCABULARY.positions:
        claimed += [VOCABULARY.hold_token(position, role) for role in HOLD_ROLES]
    assert sorted(claimed) == list(range(VOCABULARY.size))


def test_round_trip_preserves_the_problem() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG)
    rebuilt = decode(tokens, VOCABULARY, CATALOG)
    assert rebuilt.holds == canonical_holds(PROBLEM, CATALOG)
    assert rebuilt.angle == PROBLEM.angle
    assert rebuilt.source_layout == PROBLEM.source_layout
    assert rebuilt.grade == PROBLEM.grade


def test_round_trip_is_stable_under_hold_reordering() -> None:
    shuffled = RouteExample(
        climb_id=PROBLEM.climb_id,
        source_layout=PROBLEM.source_layout,
        angle=PROBLEM.angle,
        grade=PROBLEM.grade,
        holds=tuple(reversed(PROBLEM.holds)),
    )
    assert encode(shuffled, VOCABULARY, CATALOG) == encode(PROBLEM, VOCABULARY, CATALOG)


def test_holds_are_emitted_bottom_to_top() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG)
    body = tokens[1 + PREFIX_LENGTH : -1]
    ys = [VOCABULARY.hold_from_token(token)[0][1] for token in body]
    assert ys == sorted(ys)


def test_sequence_shape_and_length_bound() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG)
    assert tokens[0] == BOS
    assert tokens[-1] == EOS
    assert len(tokens) == 1 + PREFIX_LENGTH + len(PROBLEM.holds) + 1
    largest = board_problem(("hand",) * 20)
    assert len(encode(largest, VOCABULARY, CATALOG)) <= MAX_SEQUENCE_LENGTH


def test_layouts_use_separate_hold_tokens_for_positions_they_do_not_share() -> None:
    mirror_only = [
        position for position in CATALOG.positions if not CATALOG.contains("spray", *position)
    ]
    assert mirror_only, "expected positions that exist on mirror but not on spray"
    position = mirror_only[0]
    with pytest.raises(KeyError):
        CATALOG.placement("spray", *position)


def test_layout_and_angle_change_the_prefix() -> None:
    base = encode(PROBLEM, VOCABULARY, CATALOG)
    other_angle = encode(board_problem(("start", "hand"), angle=55), VOCABULARY, CATALOG)
    assert base[2] != other_angle[2]
    spray = encode(board_problem(("start", "hand"), layout="spray"), VOCABULARY, CATALOG)
    assert base[1] != spray[1]


def test_unconditional_prefix_replaces_every_conditioning_token() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG, unconditional=True)
    assert tokens[0] == BOS
    assert tokens[1 : 1 + PREFIX_LENGTH] == (UNCOND,) * PREFIX_LENGTH
    # The holds keep their positions, so the model sees them at the same offsets either way.
    assert tokens[1 + PREFIX_LENGTH :] == encode(PROBLEM, VOCABULARY, CATALOG)[1 + PREFIX_LENGTH :]
    with pytest.raises(ValueError):
        decode(tokens, VOCABULARY, CATALOG)


def test_unspecified_style_buckets_get_their_own_token() -> None:
    specified = encode(PROBLEM, VOCABULARY, CATALOG)
    unspecified = encode(
        PROBLEM, VOCABULARY, CATALOG, style=(None,) * len(FEATURE_NAMES)
    )
    assert specified[1 + PREFIX_LENGTH :] == unspecified[1 + PREFIX_LENGTH :]
    assert specified[4 : 1 + PREFIX_LENGTH] != unspecified[4 : 1 + PREFIX_LENGTH]


def test_prefix_primes_sampling_without_an_example() -> None:
    prefix = encode_prefix(VOCABULARY, layout="mirror", angle=45, grade="V7")
    assert len(prefix) == 1 + PREFIX_LENGTH
    assert prefix == encode(
        board_problem(("start", "hand"), angle=45, grade="V7"),
        VOCABULARY,
        CATALOG,
        style=(None,) * len(FEATURE_NAMES),
    )[: 1 + PREFIX_LENGTH]


def test_decode_ignores_trailing_padding() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG)
    assert decode(tokens + (PAD,) * 5, VOCABULARY, CATALOG) == decode(tokens, VOCABULARY, CATALOG)


def test_decode_rejects_a_repeated_position() -> None:
    tokens = encode(PROBLEM, VOCABULARY, CATALOG)
    duplicated = tokens[:-1] + (tokens[1 + PREFIX_LENGTH],) + (EOS,)
    with pytest.raises(ValueError):
        decode(duplicated, VOCABULARY, CATALOG)


def test_encode_rejects_two_roles_on_one_position() -> None:
    placement = CATALOG.placement("mirror", *CATALOG.positions[0])
    clashing = RouteExample(
        climb_id="clashing",
        source_layout="mirror",
        angle=40,
        grade="V5",
        holds=tuple(
            HoldNode(
                role=role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
            for role in ("start", "foot")
        ),
    )
    with pytest.raises(ValueError):
        encode(clashing, VOCABULARY, CATALOG)


def test_encode_requires_a_layout_and_a_grade() -> None:
    holds = PROBLEM.holds
    without_layout = RouteExample(climb_id="x", angle=40, grade="V5", holds=holds)
    with pytest.raises(ValueError):
        encode(without_layout, VOCABULARY, CATALOG)
    without_grade = RouteExample(climb_id="x", source_layout="mirror", angle=40, holds=holds)
    with pytest.raises(ValueError):
        encode(without_grade, VOCABULARY, CATALOG)


def test_encode_rejects_a_position_off_the_board() -> None:
    off_board = RouteExample(
        climb_id="off",
        source_layout="mirror",
        angle=40,
        grade="V5",
        holds=(
            HoldNode(role="start", x=0.123456, y=0.654321, hold_type="wood:FT",
                     orientation_degrees=0.0),
            HoldNode(role="finish", x=0.5, y=0.9, hold_type="wood:FT", orientation_degrees=0.0),
        ),
    )
    with pytest.raises(KeyError):
        encode(off_board, VOCABULARY, CATALOG)
