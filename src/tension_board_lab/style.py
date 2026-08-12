"""Rule-based style features, the single source of truth for what a style means.

Style is defined here in code rather than learned from labels: every feature is computable
from a problem's ``(x, y, role)`` triples alone, so it costs no annotation and is available
for every training example. The features are used twice, from this one definition:

1. as part of the generator's conditioning prefix, discretized into buckets;
2. at sampling time, as ``style_distance`` for ranking candidates and as a hold-count bound
   for logit masking.

``web/src/style.ts`` mirrors this module and is tested for parity against it.

Crimpy, slopey, and pinchy are deliberately not expressible: they would need a hand-annotated
taxonomy of the 106 hold types. They can be added later without changing anything here.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise

from .schema import HoldNode, RouteExample

# Holds a climber pulls on, in the order they are used. Feet are excluded: move lengths are
# about where the hands travel.
HAND_PATH_ROLES = frozenset({"start", "hand", "finish"})

FEATURE_NAMES: tuple[str, ...] = (
    "hand_count",
    "foot_count",
    "mean_move_length",
    "max_move_length",
    "move_length_variance",
    "height_span",
    "foot_to_hand_ratio",
)

# Bucket edges and normalization scales below are percentiles of the 17,477 training-split
# problems, never of validation or test. A value falls in bucket ``i`` when it is below
# ``edges[i]``, and in the last bucket when it is at or above every edge. Edges are strictly
# increasing, so no bucket is unreachable.
BUCKET_EDGES: dict[str, tuple[float, ...]] = {
    "hand_count": (3.0, 4.0, 5.0, 6.0),
    "foot_count": (2.0, 3.0, 4.0, 6.0),
    "mean_move_length": (0.167, 0.190, 0.214, 0.244),
    "max_move_length": (0.244, 0.276, 0.313, 0.355),
    "move_length_variance": (0.0020, 0.0035, 0.0052, 0.0081),
    # Over 40% of problems span the full board, so the top bucket is "uses the whole wall".
    "height_span": (0.85, 0.95, 0.999),
    "foot_to_hand_ratio": (0.286, 0.400, 0.500, 0.667),
}

# Spread (p90 - p10) per feature, so violations of different features are comparable.
FEATURE_SCALES: dict[str, float] = {
    "hand_count": 3.0,
    "foot_count": 5.0,
    "mean_move_length": 0.119,
    "max_move_length": 0.180,
    "move_length_variance": 0.010,
    "height_span": 0.235,
    "foot_to_hand_ratio": 0.611,
}


@dataclass(frozen=True)
class StyleFeatures:
    hand_count: float
    foot_count: float
    mean_move_length: float
    max_move_length: float
    move_length_variance: float
    height_span: float
    foot_to_hand_ratio: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in FEATURE_NAMES)


def hand_path(holds: tuple[HoldNode, ...]) -> tuple[HoldNode, ...]:
    """The holds the hands use, bottom to top.

    Problems are sets, not sequences, so the true order is unknown. Sorting by ``(y, x)`` is
    the same deterministic order the generator emits holds in, and it approximates the order a
    climber reaches them.
    """

    return tuple(
        sorted((hold for hold in holds if hold.role in HAND_PATH_ROLES), key=lambda h: (h.y, h.x))
    )


def move_lengths(holds: tuple[HoldNode, ...]) -> tuple[float, ...]:
    """Distances between consecutive hand holds."""

    path = hand_path(holds)
    return tuple(math.dist((a.x, a.y), (b.x, b.y)) for a, b in pairwise(path))


def compute_style_features(example: RouteExample) -> StyleFeatures:
    holds = example.holds
    if not holds:
        raise ValueError("Cannot compute style features for a problem without holds")
    path = hand_path(holds)
    if not path:
        raise ValueError("A problem needs at least one hand, start, or finish hold")
    moves = move_lengths(holds)
    foot_count = sum(1 for hold in holds if hold.role == "foot")
    ys = [hold.y for hold in holds]
    return StyleFeatures(
        hand_count=float(sum(1 for hold in holds if hold.role == "hand")),
        foot_count=float(foot_count),
        mean_move_length=statistics.fmean(moves) if moves else 0.0,
        max_move_length=max(moves) if moves else 0.0,
        move_length_variance=statistics.pvariance(moves) if len(moves) > 1 else 0.0,
        height_span=max(ys) - min(ys),
        # Divided by the hand path rather than by ``hand_count``, which can be zero on short
        # problems that are only a start and a finish.
        foot_to_hand_ratio=foot_count / len(path),
    )


def bucket_count(feature: str) -> int:
    return len(BUCKET_EDGES[feature]) + 1


def feature_bucket(feature: str, value: float) -> int:
    return bisect_right(BUCKET_EDGES[feature], value)


def style_buckets(features: StyleFeatures) -> tuple[int, ...]:
    """One bucket index per feature, in ``FEATURE_NAMES`` order."""

    return tuple(feature_bucket(name, value) for name, value in features.as_dict().items())


@dataclass(frozen=True)
class StylePreset:
    """An inclusive range per constrained feature; unlisted features are unconstrained."""

    name: str
    bounds: dict[str, tuple[float | None, float | None]]

    def matches(self, features: StyleFeatures) -> bool:
        return self.distance(features) == 0.0

    def distance(self, features: StyleFeatures) -> float:
        """Mean scale-normalized violation across the constrained features; 0.0 inside them."""

        values = features.as_dict()
        violations = []
        for feature, (low, high) in self.bounds.items():
            value = values[feature]
            below = low - value if low is not None and value < low else 0.0
            above = value - high if high is not None and value > high else 0.0
            violations.append((below + above) / FEATURE_SCALES[feature])
        return statistics.fmean(violations) if violations else 0.0

    def conditioning_buckets(self) -> tuple[int | None, ...]:
        """Buckets to condition the generator on, with ``None`` where the preset is silent.

        A preset pins two or three of the seven features. Inventing values for the rest would
        over-constrain sampling, so they stay unspecified and the tokenizer emits its
        "any" bucket for them.
        """

        buckets: list[int | None] = []
        for feature in FEATURE_NAMES:
            bounds = self.bounds.get(feature)
            if bounds is None:
                buckets.append(None)
                continue
            low, high = bounds
            edges = BUCKET_EDGES[feature]
            lowest = feature_bucket(feature, low) if low is not None else 0
            highest = feature_bucket(feature, high) if high is not None else len(edges)
            buckets.append((lowest + highest) // 2)
        return tuple(buckets)


# Thresholds are training-split percentiles: "high" is p70 or above, "short"/"low" is p30 or
# below, and dyno's cutoff is p90. Corpus coverage is small by construction—these describe
# distinctive problems, not the average one.
PRESETS: dict[str, StylePreset] = {
    # 12.1% of the training split.
    "power": StylePreset(
        name="power",
        bounds={
            "hand_count": (None, 4.0),
            "mean_move_length": (0.227, None),
            "max_move_length": (0.334, None),
        },
    ),
    # 1.3%: long problems are genuinely rare on this board, so the UI should say so.
    "endurance": StylePreset(
        name="endurance",
        bounds={
            "hand_count": (8.0, None),
            "mean_move_length": (None, 0.179),
            "height_span": (0.95, None),
        },
    ),
    # 7.2%.
    "dyno": StylePreset(
        name="dyno",
        bounds={"max_move_length": (0.401, None), "hand_count": (3.0, 6.0)},
    ),
    # 12.1%.
    "technical": StylePreset(
        name="technical",
        bounds={"foot_to_hand_ratio": (0.571, None), "mean_move_length": (None, 0.179)},
    ),
}


def style_distance(features: StyleFeatures, preset: str | StylePreset) -> float:
    resolved = PRESETS[preset] if isinstance(preset, str) else preset
    return resolved.distance(features)


def max_hold_count(preset: str | StylePreset) -> int | None:
    """Upper bound on total holds for logit masking, or ``None`` when unbounded.

    Derived from the preset's own hand-count ceiling plus the corpus-maximum foot count, so
    masking never contradicts ``matches``.
    """

    resolved = PRESETS[preset] if isinstance(preset, str) else preset
    hand_high = resolved.bounds.get("hand_count", (None, None))[1]
    if hand_high is None:
        return None
    foot_high = resolved.bounds.get("foot_count", (None, None))[1]
    # Starts and finishes are not bounded by any preset; the corpus never exceeds 4 of each.
    return int(hand_high + (foot_high if foot_high is not None else 24) + 8)
