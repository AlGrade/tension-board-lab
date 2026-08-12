"""What is physically bolted to the wall behind each token.

A hold token is only ``(position, role)``, so on its own the model can learn what a position
affords only by memorizing that position — separately for each of the 537 of them. Hold type
generalizes where position cannot: 56 of the 106 types are used for one role more than 70% of
the time, ``plastic:7`` is a foot 99.4% of the time, and 70.3% of all foot placements sit on
types that are mostly feet. Handing the model the type lets it share that across every position
carrying the same hold.

The same position carries a *different* hold on mirror than on spray, so these tables are per
layout and the layout is a model input rather than something derived from the tokens. Guidance
blanks the request, not the wall.
"""

from __future__ import annotations

import math

from ..catalog import HoldCatalog
from ..schema import HOLD_ROLES
from .tokenizer import GeneratorVocabulary

# Index 0 is reserved for "not a hold token", matching the critic's unknown-type slot.
UNKNOWN_HOLD_TYPE = 0


def hold_type_index(catalog: HoldCatalog) -> dict[str, int]:
    """Stable ids for every hold type on the board, 1-based like the critic's vocabulary."""

    types = sorted(
        {placement.hold_type for layout in catalog.placements.values() for placement in layout.values()}
    )
    return {hold_type: index + 1 for index, hold_type in enumerate(types)}


def physical_tables(
    vocabulary: GeneratorVocabulary, catalog: HoldCatalog
) -> tuple[list[list[int]], list[list[list[float]]], dict[str, int]]:
    """Per layout, the hold type and orientation behind every token.

    Returns ``(type_ids, orientations, hold_type_to_index)`` shaped ``[layouts, vocabulary]``
    and ``[layouts, vocabulary, 2]``. Non-hold tokens and positions absent from a layout get
    the unknown type and a zero orientation.
    """

    index = hold_type_index(catalog)
    type_ids: list[list[int]] = []
    orientations: list[list[list[float]]] = []

    for layout in vocabulary.layouts:
        types = [UNKNOWN_HOLD_TYPE] * vocabulary.size
        angles = [[0.0, 0.0] for _ in range(vocabulary.size)]
        for position in vocabulary.positions:
            if not catalog.contains(layout, *position):
                continue
            placement = catalog.placement(layout, *position)
            radians = math.radians(placement.orientation_degrees)
            # Every role of a position sits on the same physical hold.
            for role in HOLD_ROLES:
                token = vocabulary.hold_token(position, role)
                types[token] = index[placement.hold_type]
                angles[token] = [math.sin(radians), math.cos(radians)]
        type_ids.append(types)
        orientations.append(angles)

    return type_ids, orientations, index
