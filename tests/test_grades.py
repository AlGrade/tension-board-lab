import pytest

from tension_grade.grades import font_grade_index, normalize_font_grade


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7a", "7A"),
        (" 7A+ ", "7A+"),
        ("Font 8b", "8B"),
        ("5c", "5C"),
        ("7a/V6", "7A"),
    ],
)
def test_normalize_font_grade(raw: str, expected: str) -> None:
    assert normalize_font_grade(raw) == expected


def test_grade_order_is_ordinal() -> None:
    assert font_grade_index("7A") < font_grade_index("7A+") < font_grade_index("7B")


def test_non_font_grade_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_font_grade("V6")
