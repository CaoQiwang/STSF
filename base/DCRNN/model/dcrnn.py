from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _support_mm(support: Tensor, x: Tensor) -> Tensor:
    batch_size, num_nodes, channels = x.shape
    x_2d = x.permute(1, 2, 0).reshape(num_nodes, channels * batch_size)
    output = torch.sparse.mm(support, x_2d)
    return output.reshape(num_nodes, channels, batch_size).permute(2, 0, 1)


class DiffusionGraphConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_supports: int,
        max_diffusion_step: int,
        bias_start: float,
    ) -> None:
        super().__init__()
        self.max_diffusion_step = max_diffusion_step
        num_matrices = num_supports * max_diffusion_step + 1
        self.weight = nn.Parameter(
            torch.empty((input_dim + hidden_dim) * num_matrices, output_dim)
        )
        self.bias = nn.Parameter(torch.full((output_dim,), bias_start))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs: Tensor, state: Tensor, supports: tuple[Tensor, ...]) -> Tensor:
        x0 = torch.cat((inputs, state), dim=-1)
        diffusion_terms = [x0]

        if self.max_diffusion_step > 0:
            for support in supports:
                x1 = _support_mm(support, x0)
                diffusion_terms.append(x1)
                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2.0 * _support_mm(support, x1) - x0
                    diffusion_terms.append(x2)
                    x1, x0 = x2, x1

        x = torch.stack(diffusion_terms, dim=-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return torch.matmul(x, self.weight) + self.bias


class DCGRUCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_supports: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = DiffusionGraphConv(
            input_dim,
            hidden_dim,
            2 * hidden_dim,
            num_supports,
            max_diffusion_step,
            bias_start=1.0,
        )
        self.candidate = DiffusionGraphConv(
            input_dim,
            hidden_dim,
            hidden_dim,
            num_supports,
            max_diffusion_step,
            bias_start=0.0,
        )

    def forward(self, inputs: Tensor, state: Tensor, supports: tuple[Tensor, ...]) -> Tensor:
        reset_gate, update_gate = torch.chunk(
            torch.sigmoid(self.gates(inputs, state, supports)), 2, dim=-1
        )
        candidate = torch.tanh(self.candidate(inputs, reset_gate * state, supports))
        return update_gate * state + (1.0 - update_gate) * candidate


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_supports: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cells = nn.ModuleList(
            [
                DCGRUCell(
                    input_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    num_supports,
                    max_diffusion_step,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, inputs: Tensor, supports: tuple[Tensor, ...]) -> Tensor:
        batch_size, _, num_nodes, _ = inputs.shape
        states = [
            inputs.new_zeros((batch_size, num_nodes, self.hidden_dim))
            for _ in self.cells
        ]
        for time_step in range(inputs.shape[1]):
            output = inputs[:, time_step]
            for layer, cell in enumerate(self.cells):
                states[layer] = cell(output, states[layer], supports)
                output = states[layer]
        return torch.stack(states)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_supports: int,
        max_diffusion_step: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.cells = nn.ModuleList(
            [
                DCGRUCell(
                    output_dim if layer == 0 else hidden_dim,
                    hidden_dim,
                    num_supports,
                    max_diffusion_step,
                )
                for layer in range(num_layers)
            ]
        )
        self.projection = nn.Linear(hidden_dim, output_dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(
        self,
        inputs: Tensor,
        states: Tensor,
        supports: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        next_states = []
        output = inputs
        for layer, cell in enumerate(self.cells):
            output = cell(output, states[layer], supports)
            next_states.append(output)
        return self.projection(output), torch.stack(next_states)


class DCRNN(nn.Module):
    """DCRNN encoder-decoder with diffusion convolution and scheduled sampling."""

    def __init__(
        self,
        supports: tuple[Tensor, ...],
        input_dim: int = 2,
        output_dim: int = 1,
        rnn_units: int = 64,
        num_rnn_layers: int = 2,
        max_diffusion_step: int = 2,
        horizon: int = 12,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = True,
    ) -> None:
        super().__init__()
        if not supports:
            raise ValueError("At least one diffusion support is required")
        self.horizon = horizon
        self.output_dim = output_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning

        for index, support in enumerate(supports):
            self.register_buffer(f"support_{index}", support.coalesce())
        self.num_supports = len(supports)

        self.encoder = Encoder(
            input_dim,
            rnn_units,
            num_rnn_layers,
            self.num_supports,
            max_diffusion_step,
        )
        self.decoder = Decoder(
            output_dim,
            rnn_units,
            num_rnn_layers,
            self.num_supports,
            max_diffusion_step,
        )

    @property
    def supports(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, f"support_{index}") for index in range(self.num_supports))

    def sampling_threshold(self, batches_seen: int) -> float:
        return self.cl_decay_steps / (
            self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps)
        )

    def forward(
        self,
        inputs: Tensor,
        labels: Tensor | None = None,
        batches_seen: int = 0,
    ) -> Tensor:
        states = self.encoder(inputs, self.supports)
        batch_size, _, num_nodes, _ = inputs.shape
        decoder_input = inputs.new_zeros((batch_size, num_nodes, self.output_dim))
        outputs = []

        for time_step in range(self.horizon):
            decoder_output, states = self.decoder(decoder_input, states, self.supports)
            outputs.append(decoder_output)
            decoder_input = decoder_output

            if self.training and labels is not None:
                if self.use_curriculum_learning:
                    if torch.rand((), device=inputs.device).item() < self.sampling_threshold(batches_seen):
                        decoder_input = labels[:, time_step]
                else:
                    decoder_input = labels[:, time_step]

        return torch.stack(outputs, dim=1)
