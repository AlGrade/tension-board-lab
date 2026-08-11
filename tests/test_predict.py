from tension_grade.model import ModelConfig
from tension_grade.predict import route_for_model


def test_essential_prediction_does_not_require_removed_fields() -> None:
    config = ModelConfig(
        num_hold_families=10,
        num_variants=4,
        num_grades=8,
        use_variant=False,
        use_material=False,
        use_layout=False,
    )
    route = route_for_model(
        {
            "angle": 40,
            "holds": [
                {
                    "placement_id": "1",
                    "role": "start",
                    "x": 0.2,
                    "y": 0.1,
                    "hold_id": "20D",
                    "hold_family": "wood:20D",
                    "orientation_degrees": 45,
                },
                {
                    "placement_id": "2",
                    "role": "finish",
                    "x": 0.8,
                    "y": 0.9,
                    "hold_id": "21l",
                    "hold_family": "plastic:21",
                    "orientation_degrees": 180,
                },
            ],
        },
        config,
    )
    assert route.layout == "mirror"
    assert all(hold.material == "wood" for hold in route.holds)
    assert all(hold.variant == "none" for hold in route.holds)
