import math

import pytest
import torch

from tension_board_lab.catalog import HoldCatalog
from tension_board_lab.generator.constraints import MAX_START_HEIGHT, MIN_FINISH_HEIGHT
from tension_board_lab.generator.model import BoulderGenerator, GeneratorConfig
from tension_board_lab.generator.sample import (
    nucleus_filter,
    per_token_likelihood,
    rank_candidates,
    sample_problems,
)
from tension_board_lab.generator.tokenizer import MAX_SEQUENCE_LENGTH, GeneratorVocabulary
from tension_board_lab.schema import HoldNode, RouteExample

CATALOG = HoldCatalog.load()
VOCABULARY = GeneratorVocabulary.build(CATALOG)


def untrained_model() -> BoulderGenerator:
    """An untrained model is the harshest test of masking: its choices are near-uniform."""

    torch.manual_seed(0)
    return BoulderGenerator(
GeneratorConfig(
            vocabulary_size=VOCABULARY.size,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            num_hold_types=106,
            num_layouts=len(VOCABULARY.layouts),
            width=32,
            heads=4,
            layers=2,
        )
    ).eval()


def sample(**overrides) -> list[tuple[RouteExample, float]]:
    options = {
        "layout": "mirror",
        "angle": 40,
        "grade": "V5",
        "count": 12,
        "guidance": 1.5,
        "generator": torch.Generator().manual_seed(0),
    }
    options.update(overrides)
    return sample_problems(untrained_model(), VOCABULARY, CATALOG, **options)


def test_every_sample_is_a_valid_problem() -> None:
    for problem, _ in sample():
        roles = [hold.role for hold in problem.holds]
        assert len(problem.holds) >= 2
        assert "start" in roles and "finish" in roles
        positions = {(hold.x, hold.y) for hold in problem.holds}
        assert len(positions) == len(problem.holds)
        for hold in problem.holds:
            assert CATALOG.contains("mirror", hold.x, hold.y)
            if hold.role == "start":
                assert hold.y <= MAX_START_HEIGHT
            if hold.role == "finish":
                assert hold.y >= MIN_FINISH_HEIGHT


def test_rows_that_finish_at_different_steps_still_decode() -> None:
    """Early finishers are padded with EOS while the batch runs on.

    Those fillers must not reach the decoder, which rejects a stray EOS. Several seeds,
    because whether rows finish together is exactly what varies.
    """

    for seed in range(6):
        problems = sample(
            count=16, max_holds=10, generator=torch.Generator().manual_seed(seed)
        )
        assert len({len(problem.holds) for problem, _ in problems}) >= 1
        for problem, _ in problems:
            assert 2 <= len(problem.holds) <= 10


def test_samples_carry_the_requested_conditioning() -> None:
    for problem, _ in sample(angle=45, grade="V8", layout="spray"):
        assert problem.angle == 45
        assert problem.grade == "V8"
        assert problem.source_layout == "spray"


def test_the_hold_limit_is_respected() -> None:
    for problem, _ in sample(max_holds=6):
        assert 2 <= len(problem.holds) <= 6


def test_log_likelihoods_are_negative_and_finite() -> None:
    for _, likelihood in sample():
        assert likelihood < 0.0
        assert not math.isnan(likelihood)


def test_sampling_is_reproducible_for_one_seed() -> None:
    first = sample(generator=torch.Generator().manual_seed(7))
    second = sample(generator=torch.Generator().manual_seed(7))
    assert [problem.holds for problem, _ in first] == [problem.holds for problem, _ in second]


def test_nucleus_filter_keeps_the_smallest_covering_set() -> None:
    row = torch.tensor([0.5, 0.3, 0.15, 0.05])
    filtered = nucleus_filter(row, 0.7)
    assert filtered[0] > 0 and filtered[1] > 0
    assert filtered[2] == 0 and filtered[3] == 0


def test_nucleus_filter_never_empties_the_set() -> None:
    row = torch.tensor([0.9, 0.05, 0.05])
    assert nucleus_filter(row, 0.1).sum() > 0


def test_ranking_prefers_the_requested_grade() -> None:
    problems = [problem for problem, _ in sample(count=3)]
    likelihoods = [-10.0, -10.0, -10.0]
    # Expected grades 5.0, 7.0, 3.0 against a target of 5.
    scores = [(5.0, 0.4), (7.0, 0.4), (3.0, 0.4)]
    ranked = rank_candidates(problems, likelihoods, scores, target_index=5.0)
    assert ranked[0].expected_grade == 5.0
    assert [candidate.score for candidate in ranked] == sorted(
        candidate.score for candidate in ranked
    )


def test_ranking_rewards_a_higher_likelihood_when_grades_tie() -> None:
    problems = [problem for problem, _ in sample(count=2)]
    ranked = rank_candidates(
        problems, [-30.0, -10.0], [(5.0, 0.4), (5.0, 0.4)], target_index=5.0
    )
    assert ranked[0].log_likelihood == -10.0


def board_problem(hold_count: int) -> RouteExample:
    positions = [p for p in CATALOG.positions if CATALOG.contains("mirror", *p)][:hold_count]
    roles = ["start"] + ["hand"] * (hold_count - 2) + ["finish"]
    return RouteExample(
        climb_id="x",
        angle=40,
        source_layout="mirror",
        grade="V5",
        holds=tuple(
            HoldNode(
                role=role,
                x=placement.x,
                y=placement.y,
                hold_type=placement.hold_type,
                orientation_degrees=placement.orientation_degrees,
            )
            for role, placement in zip(
                roles, (CATALOG.placement("mirror", *p) for p in positions)
            )
        ),
    )


def test_ranking_does_not_punish_a_problem_for_being_longer() -> None:
    """The likelihood term is per token, not a sum.

    Ranking on the sum preferred short problems for no other reason, and against 20
    hand-labelled problems it ordered good below bad more often than chance — worse than
    picking at random.
    """

    shorter, longer = board_problem(4), board_problem(12)
    # Equally plausible per token; the sums differ only because the lengths do.
    per_token = -3.0
    likelihoods = [
        per_token * (len(shorter.holds) + 1),
        per_token * (len(longer.holds) + 1),
    ]
    scores = [(5.0, 0.4), (5.0, 0.4)]
    ranked = rank_candidates([shorter, longer], likelihoods, scores, target_index=5.0)
    assert ranked[0].score == pytest.approx(ranked[1].score)

    # Ranking on the sum would have put the shorter one first by a wide margin.
    summed = [
        abs(5.0 - 5.0) - 0.05 * likelihood for likelihood in likelihoods
    ]
    assert summed[0] < summed[1] - 1.0


def test_per_token_likelihood_divides_by_holds_plus_eos() -> None:
    problem = board_problem(6)
    assert per_token_likelihood(problem, -14.0) == pytest.approx(-14.0 / 7)
