import json
from pathlib import Path

import pytest
import torch

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.data import Vocabulary, collate_routes
from tension_board_lab.export.artifacts import (
    board_artifact,
    critic_artifact,
    generator_artifact,
    parity_fixtures,
    select_fixture_examples,
)
from tension_board_lab.generator.tokenizer import GeneratorVocabulary
from tension_board_lab.grade.model import ModelConfig, TensionGradeTransformer
from tension_board_lab.schema import RouteExample

CATALOG = HoldCatalog.load()
CRITIC_CHECKPOINT = Path("checkpoints/grade/tb2_12x12.pt")
DATASET = Path("data/processed/tb2_12x12.jsonl")

needs_critic = pytest.mark.skipif(
    not CRITIC_CHECKPOINT.exists(), reason="critic checkpoint not built"
)


def test_board_artifact_covers_every_placement() -> None:
    board = board_artifact(CATALOG)
    assert sorted(board["layouts"]) == ["mirror", "spray"]
    for layout, holds in board["layouts"].items():
        assert len(holds) == 498, layout
        ys = [hold["y"] for hold in holds]
        assert ys == sorted(ys), "holds must be ordered bottom to top"


def test_board_artifact_inverts_the_coordinate_normalization() -> None:
    """`raw_x = x * 128 - 64` and `raw_y = y * 136 + 4`, the inverse of the import SQL."""

    board = board_artifact(CATALOG)
    for hold in board["layouts"]["mirror"][:50]:
        assert hold["raw_x"] == round(hold["x"] * 128.0 - 64.0)
        assert hold["raw_y"] == round(hold["y"] * 136.0 + 4.0)


def test_board_artifact_carries_auroras_role_colors() -> None:
    colors = board_artifact(CATALOG)["role_colors"]
    assert colors == {
        "start": "00DD00",
        "hand": "0066FF",
        "finish": "FF0000",
        "foot": "FF00FF",
    }


@needs_critic
def test_critic_artifact_records_the_input_contract() -> None:
    payload = torch.load(CRITIC_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    artifact = critic_artifact(payload, vocabulary)
    assert artifact["input_order"] == [
        "hold_type_ids",
        "orientations",
        "roles",
        "coordinates",
        "mask",
        "angles",
    ]
    assert artifact["role_to_index"] == {"start": 0, "hand": 1, "foot": 2, "finish": 3}
    assert artifact["temperature"] == pytest.approx(1.67, abs=1e-3)
    assert len(artifact["hold_type_to_index"]) == 106


def test_generator_artifact_rebuilds_its_vocabulary() -> None:
    vocabulary = GeneratorVocabulary.build(CATALOG)
    artifact = generator_artifact(vocabulary)
    assert artifact["size"] == vocabulary.size
    assert artifact["special_tokens"] == {"pad": 0, "bos": 1, "eos": 2, "uncond": 3}
    restored = GeneratorVocabulary.from_dict(json.loads(json.dumps(artifact["vocabulary"])))
    assert restored.size == vocabulary.size
    assert restored.positions == vocabulary.positions
    assert restored.hold_offset == vocabulary.hold_offset


@needs_critic
@pytest.mark.skipif(not DATASET.exists(), reason="processed dataset not built")
def test_fixtures_reproduce_the_pytorch_logits() -> None:
    """The fixtures are the contract the TypeScript featurizer is tested against."""

    payload = torch.load(CRITIC_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    model = TensionGradeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()

    examples = select_fixture_examples(DATASET, 4)
    cases = json.loads(json.dumps(parity_fixtures(examples, model, vocabulary)))
    assert len(cases) == 4
    for case in cases:
        problem = RouteExample.from_dict(case["problem"])
        batch = collate_routes([problem], vocabulary)
        assert batch["hold_type_ids"][0].tolist() == case["tensors"]["hold_type_ids"]
        assert batch["roles"][0].tolist() == case["tensors"]["roles"]
        assert [int(v) for v in batch["mask"][0].tolist()] == case["tensors"]["mask"]
        # Orientation is [sin, cos] in that order; a swap here is a silent wrong answer.
        for row, (sin_value, cos_value) in zip(
            batch["orientations"][0].tolist(), case["tensors"]["orientations"]
        ):
            assert row[0] == pytest.approx(sin_value, abs=1e-6)
            assert row[1] == pytest.approx(cos_value, abs=1e-6)


@pytest.mark.skipif(not DATASET.exists(), reason="processed dataset not built")
def test_fixture_selection_spans_short_and_long_problems() -> None:
    examples = select_fixture_examples(DATASET, 8)
    sizes = [len(example.holds) for example in examples]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]
