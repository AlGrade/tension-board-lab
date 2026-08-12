"""Measure the generator well enough to tell a real improvement from noise.

Ad-hoc sampling runs were too small to decide anything: two runs of about a hundred samples
each put the grade error at guidance 2.5 at 0.752 and 0.949. Every number here therefore comes
with a bootstrap confidence interval, and the seed is fixed so a run can be repeated exactly.

Written as one command so a before/after comparison is a diff of two JSON files.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

import torch

from ..catalog import DEFAULT_CATALOG_PATH, HoldCatalog
from ..data import Vocabulary, configuration_signature
from ..grade.model import ModelConfig, TensionGradeTransformer
from ..grade.train import select_device
from ..schema import HOLD_ROLES, RouteExample, load_jsonl
from .model import BoulderGenerator
from .sample import load_generator, sample_problems, score_with_critic
from .tokenizer import MAX_HOLDS, GeneratorVocabulary

# Measured over the 44,234 leakage-free training problems; the target the generator should
# reproduce if its problems look like the ones people actually set.
CORPUS_ROLES = {"start": 1.54, "hand": 4.52, "foot": 3.35, "finish": 1.06}
CORPUS_MEAN_HOLDS = 10.6

# Mirrors sample_problems' own defaults; recorded in the report so a comparison of two runs
# can tell whether anything but the checkpoint changed.
SAMPLING_DEFAULTS = {"temperature": 1.0, "top_p": 0.92, "max_holds": MAX_HOLDS}


def bootstrap(values: Sequence[float], rounds: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean with a 95% confidence interval, so small differences are not over-read."""

    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    generator = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(rounds):
        means.append(statistics.fmean(generator.choices(values, k=size)))
    means.sort()
    return {
        "mean": round(statistics.fmean(values), 4),
        "low": round(means[int(rounds * 0.025)], 4),
        "high": round(means[int(rounds * 0.975)], 4),
        "n": size,
    }


def hold_set(problem: RouteExample) -> frozenset[tuple[float, float, str]]:
    return frozenset((hold.x, hold.y, hold.role) for hold in problem.holds)


def diversity(problems: Sequence[RouteExample]) -> float:
    """Mean pairwise Jaccard distance within one batch of candidates.

    Twelve suggestions that are variations of one problem are worth less than twelve different
    ones, and nothing measured this before.

    Returns one number per candidate set rather than the individual pair distances. The pairs
    within a set are not independent — n problems give n(n-1)/2 of them — so bootstrapping the
    pairs would report an interval far narrower than the evidence supports.
    """

    sets = [hold_set(problem) for problem in problems]
    distances = []
    for index, first in enumerate(sets):
        for second in sets[index + 1 :]:
            union = len(first | second)
            distances.append(1.0 - (len(first & second) / union) if union else 0.0)
    return statistics.fmean(distances) if distances else 0.0


def is_valid(problem: RouteExample, catalog: HoldCatalog, layout: str) -> bool:
    roles = {hold.role for hold in problem.holds}
    positions = {(hold.x, hold.y) for hold in problem.holds}
    return (
        len(problem.holds) >= 2
        and "start" in roles
        and "finish" in roles
        and len(positions) == len(problem.holds)
        and all(catalog.contains(layout, hold.x, hold.y) for hold in problem.holds)
    )


def evaluate(
    model: BoulderGenerator,
    vocabulary: GeneratorVocabulary,
    catalog: HoldCatalog,
    critic: TensionGradeTransformer,
    critic_vocabulary: Vocabulary,
    temperature: float,
    *,
    grades: Sequence[str],
    guidances: Sequence[float],
    layout: str,
    angle: int,
    count: int,
    seed: int,
    corpus_signatures: set[str],
    device: torch.device,
) -> dict[str, object]:
    results = []
    for guidance in guidances:
        errors: list[float] = []
        feet: list[float] = []
        holds: list[float] = []
        role_counts: dict[str, list[float]] = {role: [] for role in HOLD_ROLES}
        novel: list[float] = []
        valid: list[float] = []
        spread: list[float] = []

        for grade in grades:
            sampled = sample_problems(
                model,
                vocabulary,
                catalog,
                layout=layout,
                angle=angle,
                grade=grade,
                count=count,
                guidance=guidance,
                device=device,
                # The sampler draws on the model's device, so the generator must live there.
                generator=torch.Generator(device=device).manual_seed(seed),
            )
            problems = [problem for problem, _ in sampled]
            scores = score_with_critic(
                problems, critic, critic_vocabulary, temperature, device
            )
            target = critic_vocabulary.grade_labels.index(grade)
            errors.extend(abs(expected - target) for expected, _ in scores)
            # One value per candidate set: the unit a user actually receives.
            spread.append(diversity(problems))
            for problem in problems:
                holds.append(len(problem.holds))
                counted = {role: 0 for role in HOLD_ROLES}
                for hold in problem.holds:
                    counted[hold.role] += 1
                for role, value in counted.items():
                    role_counts[role].append(value)
                feet.append(counted["foot"])
                novel.append(
                    0.0 if configuration_signature(problem) in corpus_signatures else 1.0
                )
                valid.append(1.0 if is_valid(problem, catalog, layout) else 0.0)

        results.append(
            {
                "guidance": guidance,
                "grade_mae": bootstrap(errors),
                "holds_per_problem": bootstrap(holds),
                "roles_per_problem": {
                    role: bootstrap(values) for role, values in role_counts.items()
                },
                "diversity": bootstrap(spread),
                "novel_fraction": bootstrap(novel),
                "valid_fraction": bootstrap(valid),
                "samples": len(holds),
            }
        )
    return {
        "corpus_reference": {"roles": CORPUS_ROLES, "mean_holds": CORPUS_MEAN_HOLDS},
        "settings": {
            "grades": list(grades),
            "layout": layout,
            "angle": angle,
            "count": count,
            "seed": seed,
            # Sampling defaults the run inherits; recorded so two reports can be told apart.
            "temperature": SAMPLING_DEFAULTS["temperature"],
            "top_p": SAMPLING_DEFAULTS["top_p"],
            "max_holds": SAMPLING_DEFAULTS["max_holds"],
        },
        "by_guidance": results,
    }


def summarize(report: dict[str, object]) -> None:
    print(
        f"corpus reference: feet {CORPUS_ROLES['foot']}, holds {CORPUS_MEAN_HOLDS}",
        flush=True,
    )
    for row in report["by_guidance"]:  # type: ignore[index]
        mae = row["grade_mae"]
        feet = row["roles_per_problem"]["foot"]
        holds = row["holds_per_problem"]
        print(
            f"  guidance {row['guidance']}: "
            f"MAE {mae['mean']:.3f} [{mae['low']:.3f}, {mae['high']:.3f}] · "
            f"feet {feet['mean']:.2f} [{feet['low']:.2f}, {feet['high']:.2f}] · "
            f"holds {holds['mean']:.1f} · "
            f"diversity {row['diversity']['mean']:.3f} · "
            f"valid {100 * row['valid_fraction']['mean']:.1f}% · "
            f"novel {100 * row['novel_fraction']['mean']:.1f}%",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=Path("checkpoints/generator/tb2_12x12.pt"))
    parser.add_argument("--critic", type=Path, default=Path("checkpoints/grade/tb2_12x12.pt"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--corpus", type=Path,
                        default=Path("data/processed/tb2_12x12_pretrain.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--grades", default="V2,V4,V6,V8,V10")
    parser.add_argument("--guidance", default="1.0,1.5,2.5,3.5")
    parser.add_argument("--layout", default="mirror")
    parser.add_argument("--angle", type=int, default=40)
    parser.add_argument("--count", type=int, default=48,
                        help="Candidates per grade; the default gives ~240 samples per scale")
    parser.add_argument("--seed", type=int, default=17)
    # CPU by default: a measurement has to be reproducible and comparable across machines,
    # and accelerator kernels do not guarantee identical floating point results.
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    catalog = HoldCatalog.load(args.catalog)

    model, vocabulary = load_generator(args.checkpoint, device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    critic_payload = torch.load(args.critic, map_location="cpu", weights_only=False)
    critic_vocabulary = Vocabulary.from_dict(critic_payload["vocabulary"])
    critic = TensionGradeTransformer(ModelConfig(**critic_payload["model_config"]))
    critic.load_state_dict(critic_payload["model_state"])
    critic.to(device).eval()

    corpus_signatures = {
        configuration_signature(example) for example in load_jsonl(args.corpus)
    }
    print(f"corpus signatures: {len(corpus_signatures)}", flush=True)

    report = evaluate(
        model,
        vocabulary,
        catalog,
        critic,
        critic_vocabulary,
        float(critic_payload.get("temperature", 1.0)),
        grades=args.grades.split(","),
        guidances=[float(value) for value in args.guidance.split(",")],
        layout=args.layout,
        angle=args.angle,
        count=args.count,
        seed=args.seed,
        corpus_signatures=corpus_signatures,
        device=device,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["model_config"] = payload.get("model_config")
    report["selected_epoch"] = payload.get("selected_epoch")
    report["test_metrics"] = payload.get("test_metrics")
    summarize(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"written": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
