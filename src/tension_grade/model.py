"""Geometry-biased graph transformer for selected climbing holds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    num_placements: int
    num_grades: int
    num_roles: int = 4
    num_angles: int = 5
    width: int = 192
    heads: int = 8
    layers: int = 6
    expansion: int = 4
    dropout: float = 0.12

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class GraphAttentionBlock(nn.Module):
    """Self-attention with a learned bias for every relative hold geometry."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Model width must be divisible by the number of heads")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.norm_attention = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.geometry_bias = nn.Sequential(
            nn.Linear(6, config.width // 2),
            nn.GELU(),
            nn.Linear(config.width // 2, config.heads),
        )
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

    def forward(self, nodes: Tensor, coordinates: Tensor, mask: Tensor) -> Tensor:
        batch, node_count, width = nodes.shape
        normalized = self.norm_attention(nodes)
        qkv = self.qkv(normalized).view(batch, node_count, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        delta = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        dx, dy = delta.unbind(dim=-1)
        distance = torch.sqrt(dx.square() + dy.square() + 1e-8)
        geometry = torch.stack((dx, dy, dx.abs(), dy.abs(), distance, dy.sign()), dim=-1)
        bias = self.geometry_bias(geometry).permute(0, 3, 1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores + bias
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, node_count, width)
        nodes = nodes + self.output_dropout(self.output(attended))
        nodes = nodes + self.feed_forward(self.norm_feed_forward(nodes))
        return nodes * mask.unsqueeze(-1)


class TensionGradeTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.placement_embedding = nn.Embedding(config.num_placements + 1, config.width)
        self.role_embedding = nn.Embedding(config.num_roles, config.width)
        self.angle_embedding = nn.Embedding(config.num_angles, config.width)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(2, config.width), nn.GELU(), nn.Linear(config.width, config.width)
        )
        self.input_norm = nn.LayerNorm(config.width)
        self.blocks = nn.ModuleList(GraphAttentionBlock(config) for _ in range(config.layers))
        self.pool_gate = nn.Sequential(
            nn.LayerNorm(config.width), nn.Linear(config.width, 1, bias=False)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.width, config.num_grades),
        )

    def forward(
        self,
        placement_ids: Tensor,
        roles: Tensor,
        coordinates: Tensor,
        mask: Tensor,
        angles: Tensor,
    ) -> Tensor:
        angle_context = self.angle_embedding(angles).unsqueeze(1)
        nodes = (
            self.placement_embedding(placement_ids)
            + self.role_embedding(roles)
            + self.coordinate_embedding(coordinates)
            + angle_context
        )
        nodes = self.input_norm(nodes) * mask.unsqueeze(-1)
        for block in self.blocks:
            nodes = block(nodes, coordinates, mask)
        pool_scores = self.pool_gate(nodes).squeeze(-1)
        pool_scores = pool_scores.masked_fill(~mask, torch.finfo(pool_scores.dtype).min)
        pooled = (pool_scores.softmax(dim=-1).unsqueeze(-1) * nodes).sum(dim=1)
        return self.head(pooled + angle_context.squeeze(1))


def grade_loss(
    logits: Tensor, targets: Tensor, weights: Tensor, distance_weight: float = 0.15
) -> Tensor:
    """Categorical loss plus an ordinal penalty for far-away grade predictions."""

    categorical = F.cross_entropy(logits, targets, label_smoothing=0.04, reduction="none")
    probabilities = logits.softmax(dim=-1)
    grade_axis = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    distance = (grade_axis.unsqueeze(0) - targets.unsqueeze(1)).abs()
    ordinal = (probabilities * distance).sum(dim=-1)
    losses = categorical + distance_weight * ordinal
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


@torch.no_grad()
def probabilities(logits: Tensor, temperature: float = 1.0) -> Tensor:
    return (logits / max(temperature, 1e-4)).softmax(dim=-1)
