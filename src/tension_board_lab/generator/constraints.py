"""Hard rules on what a problem may contain, applied as logit masks while sampling.

Masking during sampling rather than filtering afterwards is what makes generation cheap: a
rejected candidate costs a whole forward pass, a masked token costs nothing. It also
guarantees validity instead of hoping for it.

Thresholds are deliberately loose. They exist to rule out nonsense, not to encode taste, so
each one is set where it excludes a negligible slice of the real corpus—the percentages below
are measured over the 44,234 leakage-free training problems. ``web/src/generate/`` mirrors this
module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ..catalog import HoldCatalog
from ..schema import HOLD_ROLES
from .tokenizer import EOS, MAX_HOLDS, GeneratorVocabulary

# A start above this height excludes 0.027% of real problems; a finish below the other
# excludes 0.30%. Both are far weaker than "the finish is the topmost hold", which holds for
# only 96.6% and would reject genuine problems.
MAX_START_HEIGHT = 0.9
MIN_FINISH_HEIGHT = 0.2

# Every problem in the corpus has at least one start and one finish, and the schema already
# requires two holds.
MIN_HOLDS = 2

ROLE_INDEX = {role: index for index, role in enumerate(HOLD_ROLES)}


class BoulderConstraints:
    """Legal next-token masks for one layout.

    Built once per sampling run; ``mask_for`` is then a few tensor operations per step.
    """

    def __init__(
        self,
        vocabulary: GeneratorVocabulary,
        catalog: HoldCatalog,
        layout: str,
        *,
        max_holds: int = MAX_HOLDS,
        max_start_height: float = MAX_START_HEIGHT,
        min_finish_height: float = MIN_FINISH_HEIGHT,
        min_holds: int = MIN_HOLDS,
    ) -> None:
        if layout not in vocabulary.layouts:
            raise ValueError(f"Unknown layout {layout!r}")
        if max_holds < min_holds:
            raise ValueError("max_holds cannot be below min_holds")
        self.vocabulary = vocabulary
        self.layout = layout
        self.max_holds = max_holds
        self.min_holds = min_holds

        roles = len(HOLD_ROLES)
        allowed = torch.zeros(vocabulary.size, dtype=torch.bool)
        # Token ids for one position are contiguous, one per role.
        position_tokens = torch.arange(len(vocabulary.positions) * roles) + vocabulary.hold_offset
        self.position_tokens = position_tokens.view(-1, roles)

        for index, position in enumerate(vocabulary.positions):
            if not catalog.contains(layout, *position):
                continue
            y = position[1]
            for role in HOLD_ROLES:
                if role == "start" and y > max_start_height:
                    continue
                if role == "finish" and y < min_finish_height:
                    continue
                allowed[vocabulary.hold_token(position, role)] = True
        self.allowed_holds = allowed

        self.start_tokens = self._role_tokens("start")
        self.finish_tokens = self._role_tokens("finish")

    def _role_tokens(self, role: str) -> Tensor:
        tokens = self.position_tokens[:, ROLE_INDEX[role]]
        return tokens[self.allowed_holds[tokens]]

    def mask_for(self, chosen: Sequence[int]) -> Tensor:
        """Boolean mask over the vocabulary; ``True`` means the token may be sampled next."""

        allowed = self.allowed_holds.clone()
        starts = finishes = 0
        for token in chosen:
            position, role = self.vocabulary.hold_from_token(token)
            index = self.vocabulary.position_to_index[position]
            # One position carries a single role, so retire all of its tokens at once.
            allowed[self.position_tokens[index]] = False
            starts += role == "start"
            finishes += role == "finish"

        count = len(chosen)
        missing = (starts == 0) + (finishes == 0)
        complete = count >= self.min_holds and not missing

        # Leave room to still place whatever the problem is missing.
        if count + missing >= self.max_holds:
            required = torch.zeros_like(allowed)
            if starts == 0:
                required[self.start_tokens] = True
            if finishes == 0:
                required[self.finish_tokens] = True
            allowed &= required if missing else torch.zeros_like(allowed)

        allowed[EOS] = complete
        return allowed

    def is_complete(self, chosen: Sequence[int]) -> bool:
        return bool(self.mask_for(chosen)[EOS])
