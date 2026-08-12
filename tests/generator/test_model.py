import math

import pytest
import torch

from tension_board_lab.generator.model import BoulderGenerator, GeneratorConfig, sequence_loss
from tension_board_lab.generator.tokenizer import MAX_SEQUENCE_LENGTH, PAD

CONFIG = GeneratorConfig(
    vocabulary_size=64,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    num_hold_types=8,
    num_layouts=2,
    layers=2,
)
LAYOUTS = torch.zeros(1, dtype=torch.long)


def model() -> BoulderGenerator:
    torch.manual_seed(0)
    return BoulderGenerator(CONFIG).eval()


def test_initial_predictions_are_close_to_uniform() -> None:
    """Tied embeddings make the embedding scale the logit scale.

    PyTorch's default N(0, 1) embedding init put the starting loss near 57 against a uniform
    baseline of ln(vocabulary_size), so the model had to unlearn its own initialization.
    """

    network = model()
    tokens = torch.randint(0, CONFIG.vocabulary_size, (8, 16))
    with torch.no_grad():
        logits = network(tokens, torch.zeros(8, dtype=torch.long))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, CONFIG.vocabulary_size),
        torch.randint(0, CONFIG.vocabulary_size, (8 * 16,)),
    )
    assert float(loss) == pytest.approx(math.log(CONFIG.vocabulary_size), abs=0.5)


def test_output_shape_covers_the_vocabulary() -> None:
    logits = model()(
        torch.randint(0, CONFIG.vocabulary_size, (3, 12)), torch.zeros(3, dtype=torch.long)
    )
    assert logits.shape == (3, 12, CONFIG.vocabulary_size)


def test_attention_cannot_see_later_tokens() -> None:
    network = model()
    tokens = torch.randint(0, CONFIG.vocabulary_size, (1, 10))
    with torch.no_grad():
        original = network(tokens, LAYOUTS)
        changed = tokens.clone()
        changed[0, -1] = (changed[0, -1] + 1) % CONFIG.vocabulary_size
        after = network(changed, LAYOUTS)
    # Changing the last token may only move the last position's prediction.
    assert torch.allclose(original[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(original[:, -1], after[:, -1], atol=1e-6)


def test_padding_does_not_change_earlier_predictions() -> None:
    network = model()
    tokens = torch.randint(1, CONFIG.vocabulary_size, (1, 8))
    padded = torch.cat((tokens, torch.full((1, 4), PAD)), dim=1)
    with torch.no_grad():
        assert torch.allclose(
            network(tokens, LAYOUTS), network(padded, LAYOUTS)[:, :8], atol=1e-6
        )


def test_embeddings_are_tied() -> None:
    network = model()
    assert network.token_embedding.weight.shape == (CONFIG.vocabulary_size, CONFIG.width)
    parameter_ids = {id(parameter) for parameter in network.parameters()}
    assert id(network.token_embedding.weight) in parameter_ids
    # No separate output projection matrix exists.
    assert not hasattr(network, "output_projection")


def test_loss_ignores_padding() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, CONFIG.vocabulary_size)
    targets = torch.tensor([[1, 2, 3, PAD, PAD], [4, 5, PAD, PAD, PAD]])
    weights = torch.ones(2)
    baseline = sequence_loss(logits, targets, weights, PAD)
    # Changing predictions at padded positions must not move the loss.
    disturbed = logits.clone()
    disturbed[0, 3:] += 10.0
    disturbed[1, 2:] += 10.0
    assert torch.allclose(baseline, sequence_loss(disturbed, targets, weights, PAD))


def test_loss_weights_examples() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, CONFIG.vocabulary_size)
    targets = torch.tensor([[1, 2, PAD, PAD], [3, 4, PAD, PAD]])
    equal = sequence_loss(logits, targets, torch.ones(2), PAD)
    first_only = sequence_loss(logits, targets, torch.tensor([1.0, 0.0]), PAD)
    second_only = sequence_loss(logits, targets, torch.tensor([0.0, 1.0]), PAD)
    assert torch.allclose(equal, (first_only + second_only) / 2, atol=1e-6)


def test_sequences_longer_than_the_model_are_rejected() -> None:
    network = model()
    too_long = torch.zeros((1, MAX_SEQUENCE_LENGTH + 1), dtype=torch.long)
    try:
        network(too_long, LAYOUTS)
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an over-long sequence")
