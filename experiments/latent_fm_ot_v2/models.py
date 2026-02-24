#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import torch
from torch import nn


class LatentVelocityMLP(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >=2, got {depth}")

        layers: List[nn.Module] = []
        in_dim = latent_dim + 1
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([x_t, t], dim=-1)
        return self.net(x)
