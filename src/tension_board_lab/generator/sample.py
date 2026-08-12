"""Sample boulder problems and rank them with the grade critic.

This is the offline twin of what the browser will do, so results can be judged before any
frontend exists. The sampling loop, the masking, and the ranking are the same; only the
runtime differs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..catalog import DEFAULT_CATALOG_PATH, HoldCatalog
from ..data import Vocabulary, collate_routes
from ..grade.model import ModelConfig, TensionGradeTransformer, probabilities
from ..grade.train import forward_batch, move_batch, select_device
from ..schema import RouteExample
from .constraints import BoulderConstraints
from .model import BoulderGenerator, GeneratorConfig
from .tokenizer import (
    EOS,
    MAX_HOLDS,
    PREFIX_LENGTH,
    GeneratorVocabulary,
    decode,
    encode_prefix,
)


def load_generator(
    checkpoint: Path, device: torch.device
) -> tuple[BoulderGenerator, GeneratorVocabulary]:
    """Rebuild a generator with its physical lookup tables restored."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocabulary = GeneratorVocabulary.from_dict(payload["vocabulary"])
    model = BoulderGenerator(GeneratorConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, vocabulary


@dataclass(frozen=True)
class Candidate:
    problem: RouteExample
    log_likelihood: float
    expected_grade: float
    confidence: float
    score: float


def nucleus_filter(probabilities_row: Tensor, top_p: float) -> Tensor:
    """Zero everything outside the smallest set of tokens covering ``top_p`` of the mass."""

    if top_p >= 1.0:
        return probabilities_row
    ordered, indices = probabilities_row.sort(descending=True)
    cumulative = ordered.cumsum(dim=-1)
    # Keep the token that crosses the threshold, so the set is never empty.
    keep = cumulative - ordered < top_p
    filtered = torch.zeros_like(probabilities_row)
    filtered[indices[keep]] = ordered[keep]
    return filtered


@torch.no_grad()
def sample_problems(
    model: BoulderGenerator,
    vocabulary: GeneratorVocabulary,
    catalog: HoldCatalog,
    *,
    layout: str,
    angle: int,
    grade: str,
    count: int = 24,
    temperature: float = 1.0,
    top_p: float = 0.92,
    guidance: float = 1.5,
    max_holds: int = MAX_HOLDS,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[RouteExample, float]]:
    """Sample ``count`` problems in one batch, returning each with its log-likelihood.

    Guidance above 1.0 interpolates away from the unconditional distribution, trading variety
    for fidelity to the requested grade.
    """

    device = device or next(model.parameters()).device
    model.eval()

    conditional = torch.tensor(
        encode_prefix(vocabulary, layout=layout, angle=angle, grade=grade),
        dtype=torch.long,
        device=device,
    ).repeat(count, 1)
    unconditional = torch.tensor(
        encode_prefix(vocabulary, layout=layout, angle=angle, grade=grade, unconditional=True),
        dtype=torch.long,
        device=device,
    ).repeat(count, 1)

    constraints = BoulderConstraints(vocabulary, catalog, layout, max_holds=max_holds)
    # The wall is the same whether or not the request is blanked, so both halves of the
    # guidance pair carry the real layout.
    layout_ids = torch.full(
        (count,), vocabulary.layouts.index(layout), dtype=torch.long, device=device
    )
    chosen: list[list[int]] = [[] for _ in range(count)]
    finished = [False] * count
    log_likelihoods = [0.0] * count

    for _ in range(max_holds + 1):
        logits = model(conditional, layout_ids)[:, -1, :]
        if guidance != 1.0:
            free_logits = model(unconditional, layout_ids)[:, -1, :]
            logits = free_logits + guidance * (logits - free_logits)
        logits = logits / max(temperature, 1e-4)

        masks = torch.stack(
            [constraints.mask_for(row) for row in chosen],
        ).to(device)
        logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
        distribution = logits.softmax(dim=-1)

        next_tokens = []
        for row in range(count):
            if finished[row]:
                next_tokens.append(EOS)
                continue
            filtered = nucleus_filter(distribution[row], top_p)
            total = filtered.sum()
            if total <= 0:
                # Nucleus filtering emptied the set; fall back to the masked distribution.
                filtered = distribution[row]
                total = filtered.sum()
            token = int(torch.multinomial(filtered / total, 1, generator=generator).item())
            log_likelihoods[row] += float(torch.log(distribution[row, token].clamp_min(1e-12)))
            next_tokens.append(token)
            if token == EOS:
                finished[row] = True
            else:
                chosen[row].append(token)

        step = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        conditional = torch.cat((conditional, step), dim=1)
        unconditional = torch.cat((unconditional, step), dim=1)
        if all(finished):
            break

    results = []
    for row in range(count):
        tokens = conditional[row].tolist()
        # Rows that finished early were padded with EOS while the rest ran on. Cut each
        # sequence at its own first EOS so those fillers never reach the decoder.
        body_start = 1 + PREFIX_LENGTH
        if EOS in tokens[body_start:]:
            tokens = tokens[: tokens.index(EOS, body_start) + 1]
        else:
            tokens.append(EOS)
        results.append((decode(tuple(tokens), vocabulary, catalog), log_likelihoods[row]))
    return results


@torch.no_grad()
def score_with_critic(
    problems: Sequence[RouteExample],
    critic: TensionGradeTransformer,
    vocabulary: Vocabulary,
    temperature: float,
    device: torch.device,
) -> list[tuple[float, float]]:
    """Expected grade index and top-class confidence for each problem, in one batch.

    The expectation is used rather than the argmax: at roughly 38% confidence the argmax is
    too noisy to rank candidates by.
    """

    batch = move_batch(collate_routes(list(problems), vocabulary), device)
    distribution = probabilities(forward_batch(critic, batch), temperature)
    axis = torch.arange(distribution.shape[-1], device=distribution.device, dtype=distribution.dtype)
    expectations = (distribution * axis).sum(dim=-1)
    confidences = distribution.max(dim=-1).values
    return [
        (float(expectation.item()), float(confidence.item()))
        for expectation, confidence in zip(expectations, confidences)
    ]


def rank_candidates(
    problems: Sequence[RouteExample],
    log_likelihoods: Sequence[float],
    critic_scores: Sequence[tuple[float, float]],
    *,
    target_index: float,
    grade_weight: float = 1.0,
    likelihood_weight: float = 0.05,
) -> list[Candidate]:
    """Lower scores are better: grade error minus plausibility."""

    candidates = []
    for problem, likelihood, (expectation, confidence) in zip(
        problems, log_likelihoods, critic_scores
    ):
        score = grade_weight * abs(expectation - target_index) - likelihood_weight * likelihood
        candidates.append(
            Candidate(
                problem=problem,
                log_likelihood=likelihood,
                expected_grade=expectation,
                confidence=confidence,
                score=score,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Generator checkpoint")
    parser.add_argument("--critic", type=Path, help="Grade critic checkpoint, for ranking")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--layout", default="mirror")
    parser.add_argument("--angle", type=int, default=40)
    parser.add_argument("--grade", default="V5")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--show", type=int, default=3, help="How many ranked problems to print")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    catalog = HoldCatalog.load(args.catalog)

    model, vocabulary = load_generator(args.checkpoint, device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        torch.manual_seed(args.seed)

    sampled = sample_problems(
        model,
        vocabulary,
        catalog,
        layout=args.layout,
        angle=args.angle,
        grade=args.grade,
        count=args.count,
        temperature=args.temperature,
        top_p=args.top_p,
        guidance=args.guidance,
        device=device,
        generator=generator,
    )
    problems = [problem for problem, _ in sampled]
    likelihoods = [likelihood for _, likelihood in sampled]

    if args.critic is None:
        for problem, likelihood in sampled[: args.show]:
            print(json.dumps({"log_likelihood": round(likelihood, 3), **problem.as_dict()}))
        return

    critic_checkpoint = torch.load(args.critic, map_location="cpu", weights_only=False)
    critic_vocabulary = Vocabulary.from_dict(critic_checkpoint["vocabulary"])
    critic = TensionGradeTransformer(ModelConfig(**critic_checkpoint["model_config"]))
    critic.load_state_dict(critic_checkpoint["model_state"])
    critic.to(device).eval()

    scores = score_with_critic(
        problems,
        critic,
        critic_vocabulary,
        float(critic_checkpoint.get("temperature", 1.0)),
        device,
    )
    target_index = critic_vocabulary.grade_labels.index(args.grade)
    ranked = rank_candidates(problems, likelihoods, scores, target_index=target_index)
    for candidate in ranked[: args.show]:
        print(
            json.dumps(
                {
                    "requested_grade": args.grade,
                    "expected_grade": round(
                        candidate.expected_grade + critic_vocabulary.grade_offset, 2
                    ),
                    "confidence": round(candidate.confidence, 4),
                    "log_likelihood": round(candidate.log_likelihood, 3),
                    "score": round(candidate.score, 4),
                    "problem": candidate.problem.as_dict(),
                }
            )
        )


if __name__ == "__main__":
    main()
