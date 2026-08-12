import pytest

from tension_board_lab.aurora import _continuous_v_grade


def test_aurora_difficulty_is_interpolated_on_v_grade_axis() -> None:
    difficulty_to_v_index = {23: 7, 24: 8, 25: 8, 26: 9}

    assert _continuous_v_grade(23.25, difficulty_to_v_index) == pytest.approx(7.25)
    assert _continuous_v_grade(24.5, difficulty_to_v_index) == pytest.approx(8.0)
