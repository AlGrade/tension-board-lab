import pytest

from tension_grade.schema import RouteExample


def route_data() -> dict:
    return {
        "climb_id": "42",
        "angle": 40,
        "grade": "v7",
        "holds": [
            {"placement_id": "10", "role": "start", "x": 0.2, "y": 0.1},
            {"placement_id": "99", "role": "finish", "x": 0.8, "y": 0.9},
        ],
    }


def test_route_is_normalized() -> None:
    route = RouteExample.from_dict(route_data(), require_grade=True)
    assert route.grade == "V7"
    assert route.angle == 40


def test_unsupported_angle_is_rejected() -> None:
    raw = route_data()
    raw["angle"] = 30
    with pytest.raises(ValueError):
        RouteExample.from_dict(raw)
