import random

import torch

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.generator.tokenizer import (
    PAD,
    PREFIX_LENGTH,
    UNCOND,
    GeneratorVocabulary,
    encode,
)
from tension_board_lab.generator.train import (
    collate_sequences,
    encode_dataset,
    randomize_prefix,
    select_generator_examples,
)
from tension_board_lab.schema import HoldNode, RouteExample

CATALOG = HoldCatalog.load()
VOCABULARY = GeneratorVocabulary.build(CATALOG)


def problem(climb_id: str, roles: tuple[str, ...], *, layout: str = "mirror") -> RouteExample:
    positions = [p for p in CATALOG.positions if CATALOG.contains(layout, *p)][: len(roles)]
    return RouteExample(
        climb_id=climb_id,
        source_layout=layout,
        angle=40,
        grade="V5",
        ascents=5,
        holds=tuple(
            HoldNode(
                role=role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
            for role, placement in zip(
                roles, (CATALOG.placement(layout, *p) for p in positions)
            )
        ),
    )


PROBLEMS = [
    problem("a", ("start", "hand", "finish")),
    problem("b", ("start", "hand", "foot", "finish")),
]


def test_targets_skip_the_conditioning_prefix() -> None:
    batch = collate_sequences(encode_dataset(PROBLEMS, VOCABULARY, CATALOG))
    # The prefix is given, not predicted, so its target positions are ignored.
    assert (batch["targets"][:, :PREFIX_LENGTH] == PAD).all()
    assert (batch["targets"][:, PREFIX_LENGTH:] != PAD).any()


def test_inputs_and_targets_are_shifted_by_one() -> None:
    encoded = encode_dataset([PROBLEMS[0]], VOCABULARY, CATALOG)
    batch = collate_sequences(encoded)
    tokens = encoded[0].tokens
    assert batch["inputs"][0, : len(tokens) - 1].tolist() == list(tokens[:-1])
    predicted = batch["targets"][0, PREFIX_LENGTH : len(tokens) - 1].tolist()
    assert predicted == list(tokens[1 + PREFIX_LENGTH :])


def test_shorter_sequences_are_padded_not_truncated() -> None:
    encoded = encode_dataset(PROBLEMS, VOCABULARY, CATALOG)
    batch = collate_sequences(encoded)
    longest = max(len(problem.tokens) for problem in encoded)
    assert batch["inputs"].shape[1] == longest - 1
    shortest = min(range(len(encoded)), key=lambda index: len(encoded[index].tokens))
    tail = batch["inputs"][shortest, len(encoded[shortest].tokens) - 1 :]
    assert (tail == PAD).all()


def test_weights_follow_ascent_counts() -> None:
    thin = problem("thin", ("start", "hand", "finish"))
    thick = RouteExample(
        climb_id="thick",
        source_layout=thin.source_layout,
        angle=thin.angle,
        grade=thin.grade,
        ascents=500,
        holds=thin.holds,
    )
    encoded = encode_dataset([thin, thick], VOCABULARY, CATALOG)
    assert encoded[0].weight < encoded[1].weight


def test_guidance_dropout_blanks_the_whole_prefix() -> None:
    random.seed(0)
    tokens = list(encode(PROBLEMS[0], VOCABULARY, CATALOG))
    randomize_prefix(tokens, guidance_dropout=1.0)
    assert tokens[1 : 1 + PREFIX_LENGTH] == [UNCOND] * PREFIX_LENGTH


def test_no_dropout_leaves_the_prefix_alone() -> None:
    random.seed(0)
    original = list(encode(PROBLEMS[0], VOCABULARY, CATALOG))
    tokens = list(original)
    randomize_prefix(tokens, guidance_dropout=0.0)
    assert tokens == original


def test_critic_holdouts_are_dropped_from_the_generator_pool() -> None:
    held_out = problem("held-out", ("start", "hand", "finish"))
    renamed = RouteExample(
        climb_id="renamed-copy",
        source_layout=held_out.source_layout,
        angle=held_out.angle,
        grade=held_out.grade,
        holds=held_out.holds,
    )
    other = problem("other", ("start", "hand", "foot", "finish"))
    selected = select_generator_examples([renamed, other], [held_out])
    assert [example.climb_id for example in selected] == ["other"]


def test_a_batch_runs_through_the_model() -> None:
    from tension_board_lab.generator.model import (
        BoulderGenerator,
        GeneratorConfig,
        sequence_loss,
    )
    from tension_board_lab.generator.tokenizer import MAX_SEQUENCE_LENGTH

    torch.manual_seed(0)
    batch = collate_sequences(encode_dataset(PROBLEMS, VOCABULARY, CATALOG))
    model = BoulderGenerator(
GeneratorConfig(
            vocabulary_size=VOCABULARY.size,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            num_hold_types=106,
            num_layouts=len(VOCABULARY.layouts),
            width=32,
            heads=4,
            layers=2,
        )
    )
    loss = sequence_loss(
        model(batch["inputs"], batch["layouts"]), batch["targets"], batch["weights"], PAD
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
