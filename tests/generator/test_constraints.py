import pytest

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.generator.constraints import (
    MAX_START_HEIGHT,
    MIN_FINISH_HEIGHT,
    BoulderConstraints,
)
from tension_board_lab.generator.tokenizer import EOS, GeneratorVocabulary

CATALOG = HoldCatalog.load()
VOCABULARY = GeneratorVocabulary.build(CATALOG)


def constraints(**overrides) -> BoulderConstraints:
    return BoulderConstraints(VOCABULARY, CATALOG, "mirror", **overrides)


def token(position_index: int, role: str) -> int:
    return VOCABULARY.hold_token(VOCABULARY.positions[position_index], role)


def mirror_positions() -> list[tuple[float, float]]:
    return [p for p in VOCABULARY.positions if CATALOG.contains("mirror", *p)]


def first_position(role: str, predicate) -> tuple[float, float]:
    for position in mirror_positions():
        if predicate(position[1]):
            return position
    raise AssertionError(f"no mirror position for {role}")


def test_positions_outside_the_layout_are_never_allowed() -> None:
    spray_only = [p for p in VOCABULARY.positions if not CATALOG.contains("mirror", *p)]
    assert spray_only
    allowed = constraints().mask_for([])
    for position in spray_only:
        for role in ("start", "hand", "foot", "finish"):
            assert not allowed[VOCABULARY.hold_token(position, role)]


def test_start_and_finish_respect_their_height_bounds() -> None:
    allowed = constraints().mask_for([])
    high = first_position("start", lambda y: y > MAX_START_HEIGHT)
    assert not allowed[VOCABULARY.hold_token(high, "start")]
    # The same position is fine for a hand.
    assert allowed[VOCABULARY.hold_token(high, "hand")]
    low = first_position("finish", lambda y: y < MIN_FINISH_HEIGHT)
    assert not allowed[VOCABULARY.hold_token(low, "finish")]
    assert allowed[VOCABULARY.hold_token(low, "foot")]


def test_a_used_position_is_retired_for_every_role() -> None:
    positions = mirror_positions()
    index = VOCABULARY.position_to_index[positions[0]]
    chosen = [VOCABULARY.hold_token(positions[0], "start")]
    allowed = constraints().mask_for(chosen)
    for role in ("start", "hand", "foot", "finish"):
        assert not allowed[VOCABULARY.hold_token(positions[0], role)]
    # A different position is untouched.
    assert allowed[VOCABULARY.hold_token(positions[1], "hand")].item() is True
    assert index >= 0


def test_eos_is_blocked_until_the_problem_is_valid() -> None:
    rules = constraints()
    low = first_position("start", lambda y: y <= MAX_START_HEIGHT)
    high = first_position("finish", lambda y: y >= MIN_FINISH_HEIGHT)
    assert not rules.mask_for([])[EOS]

    only_start = [VOCABULARY.hold_token(low, "start")]
    assert not rules.mask_for(only_start)[EOS]

    with_finish = only_start + [VOCABULARY.hold_token(high, "finish")]
    assert rules.mask_for(with_finish)[EOS]
    assert rules.is_complete(with_finish)


def test_missing_roles_are_forced_before_the_hold_limit() -> None:
    rules = constraints(max_holds=4)
    positions = [p for p in mirror_positions() if MIN_FINISH_HEIGHT <= p[1] <= MAX_START_HEIGHT]
    chosen = [VOCABULARY.hold_token(positions[0], "start")]
    chosen += [VOCABULARY.hold_token(positions[index], "hand") for index in (1, 2)]
    # One slot left and no finish yet: only finishes may be sampled.
    allowed = rules.mask_for(chosen)
    assert not allowed[EOS]
    assert allowed[VOCABULARY.hold_token(positions[5], "finish")]
    assert not allowed[VOCABULARY.hold_token(positions[5], "hand")]


def test_a_problem_can_always_be_completed() -> None:
    """Whatever the sampler picks, some legal continuation still finishes the problem."""

    rules = constraints(max_holds=6)
    positions = [p for p in mirror_positions() if MIN_FINISH_HEIGHT <= p[1] <= MAX_START_HEIGHT]
    chosen: list[int] = []
    for step in range(6):
        allowed = rules.mask_for(chosen)
        assert allowed.any(), f"dead end after {step} holds"
        if allowed[EOS]:
            break
        candidates = [
            VOCABULARY.hold_token(position, role)
            for position in positions[: 40]
            for role in ("hand", "foot", "start", "finish")
        ]
        pick = next(t for t in candidates if allowed[t])
        chosen.append(pick)
    assert rules.is_complete(chosen)


def test_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValueError):
        BoulderConstraints(VOCABULARY, CATALOG, "nonexistent")
