from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _init_conv(conv: nn.Conv2d) -> None:
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ChannelAlign(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.projection = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False) if c_in > c_out else None
        if self.projection is not None:
            _init_conv(self.projection)

    def forward(self, x: Tensor) -> Tensor:
        if self.projection is not None:
            x = self.projection(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        elif self.c_in < self.c_out:
            x = F.pad(x, (0, self.c_out - self.c_in))
        return x


class TemporalConvLayer(nn.Module):
    def __init__(self, kt: int, c_in: int, c_out: int, activation: str) -> None:
        super().__init__()
        self.kt = kt
        self.c_out = c_out
        self.activation = activation
        self.align = ChannelAlign(c_in, c_out)
        out_channels = 2 * c_out if activation == "glu" else c_out
        self.conv = nn.Conv2d(c_in, out_channels, kernel_size=(kt, 1))
        _init_conv(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.align(x)[:, self.kt - 1 :, :, :]
        x_conv = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        if self.activation == "glu":
            p, q = torch.split(x_conv, self.c_out, dim=-1)
            return (p + residual) * torch.sigmoid(q)
        if self.activation == "relu":
            return torch.relu(x_conv + residual)
        if self.activation == "sigmoid":
            return torch.sigmoid(x_conv)
        raise ValueError(f"Unsupported activation: {self.activation}")


class ChebyshevGraphConv(nn.Module):
    def __init__(self, ks: int, c_in: int, c_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ks, c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, cheb_polynomials: Tensor) -> Tensor:
        x_ks = torch.einsum("knm,btmc->btknc", cheb_polynomials, x)
        return torch.einsum("btknc,kco->btno", x_ks, self.weight) + self.bias


class SpatialConvLayer(nn.Module):
    def __init__(self, ks: int, c_in: int, c_out: int) -> None:
        super().__init__()
        self.align = ChannelAlign(c_in, c_out)
        self.graph_conv = ChebyshevGraphConv(ks, c_in, c_out)

    def forward(self, x: Tensor, cheb_polynomials: Tensor) -> Tensor:
        return torch.relu(self.graph_conv(x, cheb_polynomials) + self.align(x))


class STConvBlock(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        ks: int,
        kt: int,
        channels: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        c_in, c_temporal, c_out = channels
        self.temporal_in = TemporalConvLayer(kt, c_in, c_temporal, "glu")
        self.spatial = SpatialConvLayer(ks, c_temporal, c_temporal)
        self.temporal_out = TemporalConvLayer(kt, c_temporal, c_out, "relu")
        self.norm = nn.LayerNorm((num_nodes, c_out), eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, cheb_polynomials: Tensor) -> Tensor:
        x = self.temporal_in(x)
        x = self.spatial(x, cheb_polynomials)
        x = self.temporal_out(x)
        return self.dropout(self.norm(x))


class OutputLayer(nn.Module):
    def __init__(self, num_nodes: int, time_steps: int, channels: int) -> None:
        super().__init__()
        self.temporal = TemporalConvLayer(time_steps, channels, channels, "glu")
        self.norm = nn.LayerNorm((num_nodes, channels), eps=1e-6)
        self.sigmoid = TemporalConvLayer(1, channels, channels, "sigmoid")
        self.weight = nn.Parameter(torch.empty(channels, 1))
        self.node_bias = nn.Parameter(torch.zeros(1, 1, num_nodes, 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.temporal(x)
        x = self.norm(x)
        x = self.sigmoid(x)
        return (torch.matmul(x, self.weight) + self.node_bias)[:, 0, :, 0]


class STGCN(nn.Module):
    """Classic two-block STGCN with Chebyshev graph convolution."""

    def __init__(
        self,
        num_nodes: int,
        cheb_polynomials: Tensor,
        n_his: int = 12,
        ks: int = 3,
        kt: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if cheb_polynomials.shape != (ks, num_nodes, num_nodes):
            raise ValueError(
                "cheb_polynomials must have shape "
                f"({ks}, {num_nodes}, {num_nodes}), got {tuple(cheb_polynomials.shape)}"
            )

        blocks = ((1, 32, 64), (64, 32, 128))
        remaining_steps = n_his - len(blocks) * 2 * (kt - 1)
        if remaining_steps <= 1:
            raise ValueError(f"Output temporal kernel must be greater than 1, got {remaining_steps}")

        self.register_buffer("cheb_polynomials", cheb_polynomials.float())
        self.blocks = nn.ModuleList(
            [
                STConvBlock(num_nodes, ks, kt, channels, dropout)
                for channels in blocks
            ]
        )
        self.output = OutputLayer(num_nodes, remaining_steps, blocks[-1][-1])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 1:
            raise ValueError(f"Expected input [batch, time, nodes, 1], got {tuple(x.shape)}")
        for block in self.blocks:
            x = block(x, self.cheb_polynomials)
        return self.output(x)
