import pytest

from tension_grade.schema import RouteExample


def route_data() -> dict:
    return {
        "climb_id": "42",
        "layout": "mirror",
        "angle": 40,
        "grade": "v7",
        "holds": [
            {
                "placement_id": "10",
                "role": "start",
                "x": 0.2,
                "y": 0.1,
                "hold_id": "61s",
                "hold_family": "plastic:61",
                "variant": "s",
                "material": "plastic",
                "orientation_degrees": 180,
            },
            {
                "placement_id": "99",
                "role": "finish",
                "x": 0.8,
                "y": 0.9,
                "hold_id": "CBI",
                "hold_family": "wood:CBI",
                "variant": "none",
                "material": "wood",
                "orientation_degrees": 0,
            },
        ],
    }


def test_route_is_normalized() -> None:
    route = RouteExample.from_dict(route_data(), require_grade=True)
    assert route.grade == "V7"
    assert route.angle == 40
    assert route.layout == "mirror"
    assert route.holds[0].hold_family == "plastic:61"


def test_unsupported_angle_is_rejected() -> None:
    raw = route_data()
    raw["angle"] = 30
    with pytest.raises(ValueError):
        RouteExample.from_dict(raw)
