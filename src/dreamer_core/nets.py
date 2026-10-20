"""Small network components used across the world model and the actor-critic."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """``[Linear -> LayerNorm -> SiLU] x layers``, optionally followed by a linear output.

    Hidden linears have no bias because the following LayerNorm would cancel it. The output
    layer can be zero-initialised: DreamerV3 does this for the reward and critic heads so that
    their initial predictions are exactly zero instead of a random, possibly large, value that
    the actor would immediately chase.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        layers: int,
        out_dim: int | None = None,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        dim = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(dim, hidden, bias=False), nn.LayerNorm(hidden), nn.SiLU()]
            dim = hidden
        self.trunk = nn.Sequential(*blocks)
        self.out: nn.Linear | None = None
        if out_dim is not None:
            self.out = nn.Linear(dim, out_dim)
            if zero_init_output:
                nn.init.zeros_(self.out.weight)
                nn.init.zeros_(self.out.bias)
        self.out_dim = out_dim if out_dim is not None else dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.trunk(x)
        return self.out(x) if self.out is not None else x


class LayerNormGRUCell(nn.Module):
    """GRU cell with a LayerNorm on the gate pre-activations and a negative update bias.

    This is the recurrent core used by DreamerV2/V3. The LayerNorm keeps gate
    pre-activations well scaled over long rollouts, and the ``update_bias = -1`` shifts the
    update gate towards *keeping* the previous state at initialisation, which makes the
    deterministic path a good memory before the network has learned what to overwrite.
    """

    def __init__(self, in_dim: int, hidden: int, update_bias: float = -1.0) -> None:
        super().__init__()
        self.hidden = hidden
        self.update_bias = update_bias
        self.linear = nn.Linear(in_dim + hidden, 3 * hidden, bias=False)
        self.norm = nn.LayerNorm(3 * hidden)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        reset, cand, update = parts.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update + self.update_bias)
        return update * cand + (1.0 - update) * h
