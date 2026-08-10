"""Font bouldering-grade normalization and ordering."""

from __future__ import annotations

import re

FONT_GRADES: tuple[str, ...] = (
    "1A",
    "1B",
    "1C",
    "2A",
    "2B",
    "2C",
    "3A",
    "3B",
    "3C",
    "4A",
    "4B",
    "4C",
    "5A",
    "5B",
    "5C",
    "6A",
    "6A+",
    "6B",
    "6B+",
    "6C",
    "6C+",
    "7A",
    "7A+",
    "7B",
    "7B+",
    "7C",
    "7C+",
    "8A",
    "8A+",
    "8B",
    "8B+",
    "8C",
    "8C+",
    "9A",
    "9A+",
    "9B",
    "9B+",
    "9C",
    "9C+",
)

_INDEX = {grade: index for index, grade in enumerate(FONT_GRADES)}
_FONT_PATTERN = re.compile(r"^(?:FONT)?\s*([1-9])\s*([ABC])\s*(\+)?$", re.IGNORECASE)


def normalize_font_grade(value: str) -> str:
    """Return a canonical Font grade such as ``7A+``.

    Common decorations (``Font``, whitespace, lower-case letters) are accepted,
    but slash grades are deliberately rejected because they are ambiguous labels.
    """

    # Aurora stores combined labels such as ``7a/V6``; Font is the first part.
    compact = value.split("/", maxsplit=1)[0].strip()
    compact = re.sub(r"^fontainebleau\s*", "font ", compact, flags=re.IGNORECASE)
    match = _FONT_PATTERN.fullmatch(compact)
    if not match:
        raise ValueError(f"Unsupported or ambiguous Font grade: {value!r}")
    number, letter, plus = match.groups()
    grade = number + (letter.upper() if letter else "") + (plus or "")
    if grade not in _INDEX:
        raise ValueError(f"Font grade is outside the supported range: {grade}")
    return grade


def font_grade_index(value: str) -> int:
    return _INDEX[normalize_font_grade(value)]


def font_grade_span(min_index: int, max_index: int) -> tuple[str, ...]:
    if min_index < 0 or max_index >= len(FONT_GRADES) or min_index > max_index:
        raise ValueError("Invalid Font-grade index span")
    return FONT_GRADES[min_index : max_index + 1]
