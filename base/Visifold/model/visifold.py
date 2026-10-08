from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention used by the official VisiFold implementation."""

    def __init__(self, model_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)

        query = self.q_proj(x).view(shape).transpose(1, 2)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        probabilities = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.model_dim
        )
        return self.out_proj(output)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 1024,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Keep the post-normalization order used by the authors' released code.
        x = self.norm1(x + self.dropout1(self.attention(x)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))


class VisiFold(nn.Module):
    """Temporal Folding Graph with node visibility for traffic forecasting.

    Input shape: ``[batch, in_steps, nodes, 3]``. Channels are standardized
    traffic flow/speed, normalized time-of-day, and integer day-of-week.
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 24,
        out_steps: int = 24,
        steps_per_day: int = 288,
        embedding_dim: int = 64,
        feed_forward_dim: int = 1024,
        num_heads: int = 4,
        num_layers: int = 1,
        mask_ratio: float = 0.2,
        subgraph_size: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if in_steps <= 0 or out_steps <= 0:
            raise ValueError("in_steps and out_steps must be positive")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")
        if subgraph_size <= 0:
            raise ValueError("subgraph_size must be positive")

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.mask_ratio = mask_ratio
        self.subgraph_size = subgraph_size
        self.model_dim = 4 * embedding_dim

        self.temporal_folding = nn.Sequential(
            nn.Linear(in_steps, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.time_of_day_embedding = nn.Embedding(steps_per_day, embedding_dim)
        self.day_of_week_embedding = nn.Embedding(7, embedding_dim)
        self.node_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))

        nn.init.xavier_normal_(self.time_of_day_embedding.weight)
        nn.init.xavier_normal_(self.day_of_week_embedding.weight)
        nn.init.xavier_uniform_(self.node_embedding)

        self.encoder = nn.ModuleList(
            SelfAttentionLayer(
                self.model_dim, feed_forward_dim, num_heads, dropout
            )
            for _ in range(num_layers)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(self.model_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, out_steps),
        )

    def _embed(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1:3] != (self.in_steps, self.num_nodes):
            raise ValueError(
                "Expected input shape "
                f"[batch, {self.in_steps}, {self.num_nodes}, channels], got {tuple(x.shape)}"
            )
        if x.shape[-1] < 3:
            raise ValueError("VisiFold input requires value, time-of-day and day-of-week")

        values = x[..., 0].transpose(1, 2)
        folded = self.temporal_folding(values)

        time_index = torch.clamp(
            (x[:, -1, :, 1] * self.steps_per_day).long(),
            min=0,
            max=self.steps_per_day - 1,
        )
        day_index = x[:, -1, :, 2].long().remainder(7)
        time_embedding = self.time_of_day_embedding(time_index)
        day_embedding = self.day_of_week_embedding(day_index)
        spatial_embedding = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        # Preserve the feature order in the authors' released implementation.
        return torch.cat(
            (folded, time_embedding, day_embedding, spatial_embedding), dim=-1
        ).contiguous()

    def _apply_node_mask(self, x: Tensor) -> tuple[Tensor, Tensor]:
        keep_count = max(1, int(self.num_nodes * (1.0 - self.mask_ratio)))
        visible_nodes = torch.randperm(self.num_nodes, device=x.device)[:keep_count]
        visible_nodes = visible_nodes.sort().values
        return x.index_select(1, visible_nodes), visible_nodes

    @staticmethod
    def _shuffle_nodes(x: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, node_count, feature_dim = x.shape
        permutation = torch.rand(batch_size, node_count, device=x.device).argsort(dim=1)
        index = permutation.unsqueeze(-1).expand(-1, -1, feature_dim)
        return x.gather(1, index), permutation

    def _split_subgraphs(self, x: Tensor) -> tuple[Tensor, int, int]:
        batch_size, node_count, feature_dim = x.shape
        if node_count <= self.subgraph_size:
            return x, 1, node_count

        group_count = math.ceil(node_count / self.subgraph_size)
        padded_count = group_count * self.subgraph_size
        if padded_count != node_count:
            padding = x.new_zeros(batch_size, padded_count - node_count, feature_dim)
            x = torch.cat((x, padding), dim=1)
        return (
            x.view(batch_size * group_count, self.subgraph_size, feature_dim),
            group_count,
            node_count,
        )

    @staticmethod
    def _recover_subgraphs(x: Tensor, group_count: int, node_count: int) -> Tensor:
        if group_count == 1:
            return x
        return x.view(-1, group_count * x.shape[1], x.shape[2])[:, :node_count]

    @staticmethod
    def _unshuffle_nodes(x: Tensor, permutation: Tensor) -> Tensor:
        inverse = permutation.argsort(dim=1)
        return x.gather(1, inverse.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        embeddings = self._embed(x)
        visible_nodes = None
        permutation = None
        group_count = 1
        node_count = embeddings.shape[1]

        if self.training:
            embeddings, visible_nodes = self._apply_node_mask(embeddings)
            embeddings, permutation = self._shuffle_nodes(embeddings)
            embeddings, group_count, node_count = self._split_subgraphs(embeddings)

        for layer in self.encoder:
            embeddings = layer(embeddings)
        prediction = self.prediction_head(embeddings)

        if self.training:
            prediction = self._recover_subgraphs(prediction, group_count, node_count)
            prediction = self._unshuffle_nodes(prediction, permutation)

        return prediction.transpose(1, 2), visible_nodes
