"""Boulder problem to token sequence, and back.

A sequence is::

    [BOS] layout angle grade style x7 | hold hold ... hold | [EOS]

The conditioning prefix has a fixed width, so a given piece of conditioning always sits at the
same position. Classifier-free guidance replaces every prefix token with ``UNCOND`` rather than
collapsing the prefix to one token, which would shift the holds.

A hold token encodes only ``(position, role)``. Hold type and orientation are recovered from
the catalog, because within one layout a position identifies exactly one hold. That is what
keeps the vocabulary at roughly 2,150 tokens instead of one token per hold type combination.

Problems are sets, so holds are emitted in a deterministic ``(y, x)`` order: bottom to top,
which puts starts early and finishes late.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import HoldCatalog, Position, position_key
from ..grades import V_GRADES, normalize_v_grade
from ..schema import HOLD_ROLES, SUPPORTED_ANGLES, HoldNode, RouteExample
from ..style import FEATURE_NAMES, bucket_count, compute_style_features, style_buckets

PAD, BOS, EOS, UNCOND = 0, 1, 2, 3
SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<uncond>")

# The largest problem in the corpus uses 35 holds.
MAX_HOLDS = 35

# layout, angle, grade, and one bucket per style feature.
PREFIX_LENGTH = 3 + len(FEATURE_NAMES)

# [BOS] + prefix + holds + [EOS]
MAX_SEQUENCE_LENGTH = PREFIX_LENGTH + MAX_HOLDS + 2

ANGLES: tuple[int, ...] = tuple(sorted(SUPPORTED_ANGLES))
ROLE_TO_INDEX = {role: index for index, role in enumerate(HOLD_ROLES)}


@dataclass(frozen=True)
class GeneratorVocabulary:
    """Token ids for the conditioning prefix and every ``(position, role)`` pair."""

    positions: tuple[Position, ...]
    position_to_index: dict[Position, int]
    layouts: tuple[str, ...]
    angles: tuple[int, ...]
    grade_labels: tuple[str, ...]
    layout_offset: int
    angle_offset: int
    grade_offset: int
    style_offsets: tuple[int, ...]
    hold_offset: int

    @classmethod
    def build(
        cls,
        catalog: HoldCatalog,
        *,
        grade_labels: tuple[str, ...] = V_GRADES,
    ) -> GeneratorVocabulary:
        layout_offset = len(SPECIAL_TOKENS)
        angle_offset = layout_offset + len(catalog.layouts)
        grade_offset = angle_offset + len(ANGLES)

        offset = grade_offset + len(grade_labels)
        style_offsets: list[int] = []
        for feature in FEATURE_NAMES:
            style_offsets.append(offset)
            # One slot past the buckets holds "any", used when a preset leaves a feature free.
            offset += bucket_count(feature) + 1

        return cls(
            positions=catalog.positions,
            position_to_index={
                position: index for index, position in enumerate(catalog.positions)
            },
            layouts=catalog.layouts,
            angles=ANGLES,
            grade_labels=grade_labels,
            layout_offset=layout_offset,
            angle_offset=angle_offset,
            grade_offset=grade_offset,
            style_offsets=tuple(style_offsets),
            hold_offset=offset,
        )

    @property
    def size(self) -> int:
        return self.hold_offset + len(self.positions) * len(HOLD_ROLES)

    def as_dict(self) -> dict[str, object]:
        """Everything needed to rebuild this vocabulary without the catalog file."""

        return {
            "positions": [list(position) for position in self.positions],
            "layouts": list(self.layouts),
            "angles": list(self.angles),
            "grade_labels": list(self.grade_labels),
            "layout_offset": self.layout_offset,
            "angle_offset": self.angle_offset,
            "grade_offset": self.grade_offset,
            "style_offsets": list(self.style_offsets),
            "hold_offset": self.hold_offset,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> GeneratorVocabulary:
        positions = tuple((float(x), float(y)) for x, y in raw["positions"])
        return cls(
            positions=positions,
            position_to_index={position: index for index, position in enumerate(positions)},
            layouts=tuple(str(value) for value in raw["layouts"]),
            angles=tuple(int(value) for value in raw["angles"]),
            grade_labels=tuple(str(value) for value in raw["grade_labels"]),
            layout_offset=int(raw["layout_offset"]),
            angle_offset=int(raw["angle_offset"]),
            grade_offset=int(raw["grade_offset"]),
            style_offsets=tuple(int(value) for value in raw["style_offsets"]),
            hold_offset=int(raw["hold_offset"]),
        )

    # -- conditioning ------------------------------------------------------------------

    def layout_token(self, layout: str) -> int:
        try:
            return self.layout_offset + self.layouts.index(layout)
        except ValueError as error:
            raise ValueError(f"Unknown layout {layout!r}") from error

    def angle_token(self, angle: int) -> int:
        try:
            return self.angle_offset + self.angles.index(int(angle))
        except ValueError as error:
            raise ValueError(f"Unsupported angle {angle!r}") from error

    def grade_token(self, grade: str) -> int:
        label = normalize_v_grade(grade)
        try:
            return self.grade_offset + self.grade_labels.index(label)
        except ValueError as error:
            raise ValueError(f"Grade {label} is outside the generator vocabulary") from error

    def style_token(self, feature: str, bucket: int | None) -> int:
        index = FEATURE_NAMES.index(feature)
        slots = bucket_count(feature)
        # ``None`` means unspecified and maps to the slot just past the real buckets.
        resolved = slots if bucket is None else bucket
        if not 0 <= resolved <= slots:
            raise ValueError(f"Bucket {bucket} is out of range for style feature {feature!r}")
        return self.style_offsets[index] + resolved

    # -- holds -------------------------------------------------------------------------

    def hold_token(self, position: Position, role: str) -> int:
        try:
            position_index = self.position_to_index[position]
        except KeyError as error:
            raise ValueError(f"Position {position} is not on the board") from error
        return self.hold_offset + position_index * len(HOLD_ROLES) + ROLE_TO_INDEX[role]

    def is_hold_token(self, token: int) -> bool:
        return self.hold_offset <= token < self.size

    def hold_from_token(self, token: int) -> tuple[Position, str]:
        if not self.is_hold_token(token):
            raise ValueError(f"Token {token} is not a hold token")
        index = token - self.hold_offset
        position_index, role_index = divmod(index, len(HOLD_ROLES))
        return self.positions[position_index], HOLD_ROLES[role_index]


def canonical_holds(example: RouteExample, catalog: HoldCatalog) -> tuple[HoldNode, ...]:
    """The problem's holds in emission order, with type and orientation from the catalog.

    This is the form a problem survives a round trip as: the tokenizer keeps position and role,
    and the catalog supplies the rest.
    """

    layout = _require_layout(example)
    ordered = sorted(example.holds, key=lambda hold: (hold.y, hold.x))
    holds: list[HoldNode] = []
    for hold in ordered:
        placement = catalog.placement(layout, hold.x, hold.y)
        holds.append(
            HoldNode(
                role=hold.role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
        )
    return tuple(holds)


def encode_prefix(
    vocabulary: GeneratorVocabulary,
    *,
    layout: str,
    angle: int,
    grade: str,
    style: tuple[int | None, ...] | None = None,
    unconditional: bool = False,
) -> tuple[int, ...]:
    """``[BOS]`` plus the conditioning prefix, ready to prime sampling."""

    if unconditional:
        return (BOS,) + (UNCOND,) * PREFIX_LENGTH
    buckets = style if style is not None else (None,) * len(FEATURE_NAMES)
    if len(buckets) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} style buckets, got {len(buckets)}")
    return (
        BOS,
        vocabulary.layout_token(layout),
        vocabulary.angle_token(angle),
        vocabulary.grade_token(grade),
        *(
            vocabulary.style_token(feature, bucket)
            for feature, bucket in zip(FEATURE_NAMES, buckets)
        ),
    )


def encode(
    example: RouteExample,
    vocabulary: GeneratorVocabulary,
    catalog: HoldCatalog,
    *,
    style: tuple[int | None, ...] | None = None,
    unconditional: bool = False,
) -> tuple[int, ...]:
    """Encode a problem as a full training sequence.

    Style buckets are computed from the problem unless ``style`` overrides them; training
    passes a partly-``None`` tuple to teach the model to handle partial style requests.
    """

    layout = _require_layout(example)
    if example.grade is None:
        raise ValueError("Encoding a problem requires a grade")
    if len(example.holds) > MAX_HOLDS:
        raise ValueError(f"A problem may use at most {MAX_HOLDS} holds")

    holds = canonical_holds(example, catalog)
    positions = [position_key(hold.x, hold.y) for hold in holds]
    if len(set(positions)) != len(positions):
        raise ValueError("Two holds share one position; a position carries a single role")

    if style is None:
        style = style_buckets(compute_style_features(example))
    prefix = encode_prefix(
        vocabulary,
        layout=layout,
        angle=example.angle,
        grade=example.grade,
        style=style,
        unconditional=unconditional,
    )
    body = tuple(
        vocabulary.hold_token(position, hold.role)
        for position, hold in zip(positions, holds)
    )
    return prefix + body + (EOS,)


def decode(
    tokens: tuple[int, ...],
    vocabulary: GeneratorVocabulary,
    catalog: HoldCatalog,
    *,
    climb_id: str = "generated",
) -> RouteExample:
    """Rebuild a problem from a conditional sequence.

    ``climb_id``, ``grade_value``, and ``ascents`` are not carried in the sequence; everything
    the model sees—layout, angle, grade, and the holds—is recovered exactly.
    """

    sequence = list(tokens)
    while sequence and sequence[-1] == PAD:
        sequence.pop()
    if not sequence or sequence[0] != BOS:
        raise ValueError("A sequence must start with BOS")
    if len(sequence) < 1 + PREFIX_LENGTH:
        raise ValueError("Sequence is shorter than the conditioning prefix")

    prefix = sequence[1 : 1 + PREFIX_LENGTH]
    if UNCOND in prefix:
        raise ValueError("Cannot decode an unconditional sequence; its conditioning is dropped")

    layout = _lookup(vocabulary.layouts, prefix[0] - vocabulary.layout_offset, "layout")
    angle = _lookup(vocabulary.angles, prefix[1] - vocabulary.angle_offset, "angle")
    grade = _lookup(vocabulary.grade_labels, prefix[2] - vocabulary.grade_offset, "grade")

    body = sequence[1 + PREFIX_LENGTH :]
    if body and body[-1] == EOS:
        body.pop()
    if EOS in body or BOS in body:
        raise ValueError("Sequence contains a stray BOS or EOS token")

    holds: list[HoldNode] = []
    seen: set[Position] = set()
    for token in body:
        position, role = vocabulary.hold_from_token(token)
        if position in seen:
            raise ValueError(f"Position {position} appears twice in one problem")
        seen.add(position)
        placement = catalog.placement(layout, *position)
        holds.append(
            HoldNode(
                role=role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
        )
    return RouteExample(
        climb_id=climb_id,
        angle=angle,
        holds=tuple(holds),
        source_layout=layout,
        grade=grade,
    )


def _require_layout(example: RouteExample) -> str:
    if example.source_layout is None:
        raise ValueError("The generator needs a source layout to resolve hold positions")
    return example.source_layout


def _lookup(values: tuple, index: int, name: str):
    if not 0 <= index < len(values):
        raise ValueError(f"Prefix token is not a valid {name}")
    return values[index]
