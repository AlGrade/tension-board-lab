import pytest

from tension_grade.schema import RouteExample


def route_data() -> dict:
    return {
        "climb_id": "42",
        "source_layout": "mirror",
        "angle": 40,
        "grade": "v7",
        "holds": [
            {
                "role": "start",
                "x": 0.2,
                "y": 0.1,
                "hold_type": "plastic:61",
                "orientation_degrees": 180,
            },
            {
                "role": "finish",
                "x": 0.8,
                "y": 0.9,
                "hold_type": "wood:CBI",
                "orientation_degrees": 0,
            },
        ],
    }


def test_route_is_normalized() -> None:
    route = RouteExample.from_dict(route_data(), require_grade=True)
    assert route.grade == "V7"
    assert route.angle == 40
    assert route.source_layout == "mirror"
    assert route.holds[0].hold_type == "plastic:61"


def test_prediction_route_does_not_require_layout_or_removed_hold_fields() -> None:
    raw = route_data()
    raw.pop("source_layout")
    route = RouteExample.from_dict(raw)
    assert route.source_layout is None


def test_unsupported_angle_is_rejected() -> None:
    raw = route_data()
    raw["angle"] = 30
    with pytest.raises(ValueError):
        RouteExample.from_dict(raw)
