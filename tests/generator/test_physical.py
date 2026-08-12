import math

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.generator.physical import (
    UNKNOWN_HOLD_TYPE,
    hold_type_index,
    physical_tables,
)
from tension_board_lab.generator.tokenizer import BOS, EOS, PAD, UNCOND, GeneratorVocabulary
from tension_board_lab.schema import HOLD_ROLES

CATALOG = HoldCatalog.load()
VOCABULARY = GeneratorVocabulary.build(CATALOG)
TYPE_IDS, ORIENTATIONS, INDEX = physical_tables(VOCABULARY, CATALOG)


def test_every_board_hold_type_gets_an_id() -> None:
    assert len(INDEX) == 106
    # Ids start at 1 so that 0 can mean "not a hold token".
    assert min(INDEX.values()) == 1
    assert max(INDEX.values()) == 106
    assert UNKNOWN_HOLD_TYPE == 0


def test_tables_cover_one_layout_each() -> None:
    assert len(TYPE_IDS) == len(VOCABULARY.layouts)
    assert len(ORIENTATIONS) == len(VOCABULARY.layouts)
    for table in TYPE_IDS:
        assert len(table) == VOCABULARY.size


def test_special_and_prefix_tokens_carry_no_hold() -> None:
    """Only hold tokens have something bolted behind them."""

    for layout_index in range(len(VOCABULARY.layouts)):
        for token in (PAD, BOS, EOS, UNCOND):
            assert TYPE_IDS[layout_index][token] == UNKNOWN_HOLD_TYPE
            assert ORIENTATIONS[layout_index][token] == [0.0, 0.0]
        for layout in VOCABULARY.layouts:
            assert TYPE_IDS[layout_index][VOCABULARY.layout_token(layout)] == UNKNOWN_HOLD_TYPE
        for angle in VOCABULARY.angles:
            assert TYPE_IDS[layout_index][VOCABULARY.angle_token(angle)] == UNKNOWN_HOLD_TYPE


def test_every_position_on_a_layout_resolves() -> None:
    for layout_index, layout in enumerate(VOCABULARY.layouts):
        resolved = sum(1 for value in TYPE_IDS[layout_index] if value != UNKNOWN_HOLD_TYPE)
        # 498 positions, four roles each, all sharing one physical hold.
        assert resolved == 498 * len(HOLD_ROLES)


def test_all_four_roles_of_a_position_share_one_hold() -> None:
    layout = VOCABULARY.layouts[0]
    for position in VOCABULARY.positions[:60]:
        if not CATALOG.contains(layout, *position):
            continue
        ids = {TYPE_IDS[0][VOCABULARY.hold_token(position, role)] for role in HOLD_ROLES}
        assert len(ids) == 1


def test_the_same_position_resolves_differently_per_layout() -> None:
    """The reason the layout has to be a model input rather than read off the tokens."""

    shared = [
        position
        for position in VOCABULARY.positions
        if all(CATALOG.contains(layout, *position) for layout in VOCABULARY.layouts)
    ]
    assert shared
    differing = 0
    for position in shared:
        token = VOCABULARY.hold_token(position, "hand")
        if TYPE_IDS[0][token] != TYPE_IDS[1][token]:
            differing += 1
    # Measured at 432 of 459; assert the property, not the exact count.
    assert differing > len(shared) * 0.8


def test_positions_absent_from_a_layout_stay_unknown() -> None:
    mirror, spray = VOCABULARY.layouts.index("mirror"), VOCABULARY.layouts.index("spray")
    missing = [p for p in VOCABULARY.positions if not CATALOG.contains("spray", *p)]
    assert missing
    for position in missing:
        token = VOCABULARY.hold_token(position, "hand")
        assert TYPE_IDS[spray][token] == UNKNOWN_HOLD_TYPE
        assert TYPE_IDS[mirror][token] != UNKNOWN_HOLD_TYPE


def test_orientation_is_sin_cos_of_the_catalog_angle() -> None:
    """Same order as the critic's featurization; swapping them is silent and wrong."""

    layout = VOCABULARY.layouts[0]
    for position in VOCABULARY.positions[:40]:
        if not CATALOG.contains(layout, *position):
            continue
        placement = CATALOG.placement(layout, *position)
        radians = math.radians(placement.orientation_degrees)
        sine, cosine = ORIENTATIONS[0][VOCABULARY.hold_token(position, "hand")]
        assert sine == math.sin(radians)
        assert cosine == math.cos(radians)


def test_hold_type_ids_are_stable_across_builds() -> None:
    """The checkpoint stores these ids; a reshuffle would silently corrupt a trained model."""

    assert hold_type_index(CATALOG) == INDEX
    assert hold_type_index(HoldCatalog.load()) == INDEX
