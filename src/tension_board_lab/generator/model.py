"""Decoder-only transformer over boulder problem token sequences."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GeneratorConfig:
    vocabulary_size: int
    max_sequence_length: int
    width: int = 192
    heads: int = 8
    layers: int = 6
    expansion: int = 4
    dropout: float = 0.1

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


# Embeddings are tied to the output projection, so their scale sets the scale of the logits.
# PyTorch's default N(0, 1) embedding init would put logits around sqrt(width) ~ 14 and start
# training far worse than uniform; 0.02 starts it at roughly ln(vocabulary_size).
INITIALIZER_RANGE = 0.02


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)


class CausalBlock(nn.Module):
    """Pre-norm self-attention over earlier tokens only, then a feed-forward."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Model width must be divisible by the number of heads")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.norm_attention = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.width, config.width)
        self.output_dropout = nn.Dropout(config.dropout)
        self.norm_feed_forward = nn.LayerNorm(config.width)
        hidden = config.width * config.expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(config.width, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor, causal_mask: Tensor) -> Tensor:
        batch, length, width = tokens.shape
        normalized = self.norm_attention(tokens)
        qkv = self.qkv(normalized).view(batch, length, 3, self.heads, self.head_width)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(dim=2))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        # Same caveat as the critic: this constant overflows in fp16, so export as int8 or
        # fp32, or replace it with -1e4 before exporting.
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.output_dropout(self.output(attended))
        return tokens + self.feed_forward(self.norm_feed_forward(tokens))


class BoulderGenerator(nn.Module):
    """Predicts the next token of a problem sequence.

    Input and output embeddings are tied: the hold vocabulary is most of the parameters, and
    tying it halves that cost while keeping the two views of a token consistent.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.width)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(CausalBlock(config) for _ in range(config.layers))
        self.output_norm = nn.LayerNorm(config.width)
        self.output_bias = nn.Parameter(torch.zeros(config.vocabulary_size))
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.apply(_initialize)
        # Residual branches are scaled down so their sum stays unit-scale through the stack.
        residual_scale = INITIALIZER_RANGE / math.sqrt(2 * config.layers)
        for name, parameter in self.named_parameters():
            if name.endswith(("output.weight", "feed_forward.3.weight")):
                nn.init.normal_(parameter, mean=0.0, std=residual_scale)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence of {length} exceeds the model's {self.config.max_sequence_length}"
            )
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.input_dropout(hidden)
        mask = self.causal_mask[:length, :length]
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.output_norm(hidden)
        return torch.matmul(hidden, self.token_embedding.weight.t()) + self.output_bias


def sequence_loss(logits: Tensor, targets: Tensor, weights: Tensor, pad_token: int) -> Tensor:
    """Mean next-token cross-entropy per sequence, weighted by example confidence.

    Averaging within a sequence before weighting keeps long problems from dominating, so the
    loss reflects problems rather than tokens.
    """

    token_losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=pad_token,
        reduction="none",
    ).view(targets.shape)
    counted = (targets != pad_token).to(logits.dtype)
    per_sequence = (token_losses * counted).sum(dim=-1) / counted.sum(dim=-1).clamp_min(1.0)
    return (per_sequence * weights).sum() / weights.sum().clamp_min(1e-8)
