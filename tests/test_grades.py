import pytest

from tension_board_lab.grades import normalize_v_grade, v_grade_index


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v6", "V6"),
        (" V7 ", "V7"),
        ("Hueco V13", "V13"),
        ("V0", "V0"),
        ("7a/V6", "V6"),
    ],
)
def test_normalize_v_grade(raw: str, expected: str) -> None:
    assert normalize_v_grade(raw) == expected


def test_grade_order_is_ordinal() -> None:
    assert v_grade_index("V6") < v_grade_index("V7") < v_grade_index("V8")


def test_non_v_grade_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_v_grade("7A")
