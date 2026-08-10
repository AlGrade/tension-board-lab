"""Hueco V-grade normalization and ordering."""

from __future__ import annotations

import re

V_GRADES: tuple[str, ...] = tuple(f"V{number}" for number in range(23))

_INDEX = {grade: index for index, grade in enumerate(V_GRADES)}
_V_PATTERN = re.compile(r"^(?:HUECO\s*)?V\s*(\d{1,2})$", re.IGNORECASE)


def normalize_v_grade(value: str) -> str:
    """Return a canonical Hueco grade such as ``V7``.

    Aurora's combined labels (for example ``7a+/V7``), optional ``Hueco``
    prefixes, whitespace, and lower-case letters are accepted.
    """

    # Aurora stores combined labels such as ``7a/V6``; V is the second part.
    compact = value.rsplit("/", maxsplit=1)[-1].strip()
    match = _V_PATTERN.fullmatch(compact)
    if not match:
        raise ValueError(f"Unsupported V grade: {value!r}")
    grade = f"V{int(match.group(1))}"
    if grade not in _INDEX:
        raise ValueError(f"V grade is outside the supported range: {grade}")
    return grade


def v_grade_index(value: str) -> int:
    return _INDEX[normalize_v_grade(value)]


def v_grade_span(min_index: int, max_index: int) -> tuple[str, ...]:
    if min_index < 0 or max_index >= len(V_GRADES) or min_index > max_index:
        raise ValueError("Invalid V-grade index span")
    return V_GRADES[min_index : max_index + 1]
