"""Pointwise multilayer-perceptron forward model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """x (d+1) -> y (6): Re/Im of S11, S21, S22 at a single frequency."""

    def __init__(self, in_dim: int, out_dim: int = 6, hidden: tuple[int, ...] = (256, 256, 256),
                 activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
