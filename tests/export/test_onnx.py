"""ONNX parity: the exported graphs must agree with PyTorch at shapes they were not traced at."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.data import Vocabulary, collate_routes
from tension_board_lab.export.onnx import (
    CRITIC_INPUTS,
    export_critic,
    export_generator,
    quantize,
)
from tension_board_lab.generator.model import BoulderGenerator, GeneratorConfig
from tension_board_lab.generator.tokenizer import GeneratorVocabulary
from tension_board_lab.grade.model import ModelConfig, TensionGradeTransformer
from tension_board_lab.schema import RouteExample

ort = pytest.importorskip("onnxruntime", reason="export extra not installed")

CRITIC_CHECKPOINT = Path("checkpoints/grade/tb2_12x12.pt")
GENERATOR_CHECKPOINT = Path("checkpoints/generator/tb2_12x12.pt")

needs_critic = pytest.mark.skipif(
    not CRITIC_CHECKPOINT.exists(), reason="critic checkpoint not built"
)
needs_generator = pytest.mark.skipif(
    not GENERATOR_CHECKPOINT.exists(), reason="generator checkpoint not built"
)


def session(path: Path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_critic() -> tuple[TensionGradeTransformer, Vocabulary, float]:
    payload = torch.load(CRITIC_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary.from_dict(payload["vocabulary"])
    model = TensionGradeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model.eval(), vocabulary, float(payload["temperature"])


def critic_batch(vocabulary: Vocabulary, batch: int, nodes: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(batch * 100 + nodes)
    return (
        torch.randint(0, len(vocabulary.hold_type_to_index), (batch, nodes)),
        torch.randn(batch, nodes, 2),
        torch.randint(0, 4, (batch, nodes)),
        torch.rand(batch, nodes, 2),
        torch.ones(batch, nodes),
        torch.full((batch,), 40.0),
    )


@pytest.fixture(scope="module")
def critic_onnx(tmp_path_factory) -> Path:
    destination = tmp_path_factory.mktemp("onnx") / "grade.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_critic(CRITIC_CHECKPOINT, destination)
    return destination


@pytest.fixture(scope="module")
def generator_onnx(tmp_path_factory) -> Path:
    destination = tmp_path_factory.mktemp("onnx") / "generator.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_generator(GENERATOR_CHECKPOINT, destination)
    return destination


@needs_critic
# The export traces at batch=2, nodes=10; every other shape proves the axes stayed dynamic.
@pytest.mark.parametrize("batch,nodes", [(1, 2), (1, 10), (1, 35), (4, 7), (3, 35), (8, 2)])
def test_critic_matches_pytorch_at_any_shape(critic_onnx, batch, nodes) -> None:
    model, vocabulary, _ = load_critic()
    inputs = critic_batch(vocabulary, batch, nodes)
    with torch.no_grad():
        expected = model(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4] > 0.5, inputs[5]
        ).numpy()
    actual = session(critic_onnx).run(
        ["logits"], {name: tensor.numpy() for name, tensor in zip(CRITIC_INPUTS, inputs)}
    )[0]
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-4


@needs_generator
# Traced at batch=2, length=12; MAX_SEQUENCE_LENGTH is the upper bound.
@pytest.mark.parametrize("batch,length", [(1, 5), (1, 40), (4, 20), (24, 13), (2, 40)])
def test_generator_matches_pytorch_at_any_shape(generator_onnx, batch, length) -> None:
    payload = torch.load(GENERATOR_CHECKPOINT, map_location="cpu", weights_only=False)
    vocabulary = GeneratorVocabulary.from_dict(payload["vocabulary"])
    model = BoulderGenerator(GeneratorConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    torch.manual_seed(length)
    tokens = torch.randint(0, vocabulary.size, (batch, length))
    layouts = torch.zeros(batch, dtype=torch.long)
    with torch.no_grad():
        expected = model(tokens, layouts).numpy()
    actual = session(generator_onnx).run(
        ["logits"], {"tokens": tokens.numpy(), "layouts": layouts.numpy()}
    )[0]
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-4


@needs_critic
def test_exported_critic_reproduces_the_documented_prediction(critic_onnx) -> None:
    _, vocabulary, temperature = load_critic()
    route = RouteExample.from_dict(
        json.loads(Path("examples/route.json").read_text()), require_grade=False
    )
    batch = collate_routes([route], vocabulary)
    feed = {
        name: (batch[name].float() if name == "mask" else batch[name]).numpy()
        for name in CRITIC_INPUTS
    }
    logits = session(critic_onnx).run(["logits"], feed)[0][0]
    distribution = torch.tensor(logits / temperature).softmax(-1)
    index = int(distribution.argmax())
    assert vocabulary.grade_labels[index] == "V9"
    assert float(distribution[index]) == pytest.approx(0.3832, abs=5e-4)


@needs_critic
def test_int8_quantization_keeps_the_critic_usable(critic_onnx, tmp_path) -> None:
    quantized = tmp_path / "grade.int8.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize(critic_onnx, quantized)
    assert quantized.stat().st_size < critic_onnx.stat().st_size / 2

    _, vocabulary, _ = load_critic()
    full, small = session(critic_onnx), session(quantized)
    agreements = 0
    trials = 20
    for index in range(trials):
        inputs = critic_batch(vocabulary, 1, 5 + index)
        feed = {name: tensor.numpy() for name, tensor in zip(CRITIC_INPUTS, inputs)}
        agreements += int(
            full.run(["logits"], feed)[0].argmax() == small.run(["logits"], feed)[0].argmax()
        )
    assert agreements >= trials * 0.8


@needs_critic
def test_the_mask_input_is_float_not_bool(critic_onnx) -> None:
    """The web side sends floats; a bool input would be awkward to produce from JS."""

    inputs = {tensor.name: tensor.type for tensor in session(critic_onnx).get_inputs()}
    assert inputs["mask"] == "tensor(float)"
    assert inputs["hold_type_ids"] == "tensor(int64)"


@needs_critic
def test_masking_actually_changes_the_result(critic_onnx) -> None:
    """A padded node must not influence the prediction, or the mask is being ignored."""

    _, vocabulary, _ = load_critic()
    inputs = list(critic_batch(vocabulary, 1, 6))
    inputs[4] = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    run = session(critic_onnx)
    feed = {name: tensor.numpy() for name, tensor in zip(CRITIC_INPUTS, inputs)}
    before = run.run(["logits"], feed)[0]
    # Scribble over the two masked nodes only.
    disturbed = list(inputs)
    disturbed[3] = inputs[3].clone()
    disturbed[3][0, 4:] = torch.rand(2, 2) * 10
    feed_after = {name: tensor.numpy() for name, tensor in zip(CRITIC_INPUTS, disturbed)}
    assert np.abs(run.run(["logits"], feed_after)[0] - before).max() < 1e-4


def test_catalog_positions_match_the_board() -> None:
    """Guards the assumption the whole export rests on."""

    catalog = HoldCatalog.load()
    assert len(catalog.positions) == 537
    for layout in catalog.layouts:
        assert len(catalog.placements[layout]) == 498
