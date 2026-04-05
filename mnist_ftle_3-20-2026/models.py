from __future__ import annotations

import math

import torch
import torch.nn as nn


class TanhMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int, gain: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        self.gain = gain

        self.hidden = nn.ModuleList()
        fan_in = input_dim
        for _ in range(depth):
            layer = nn.Linear(fan_in, width)
            nn.init.normal_(layer.weight, mean=0.0, std=gain / math.sqrt(fan_in))
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            fan_in = width

        self.out = nn.Linear(fan_in, output_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=gain / math.sqrt(fan_in))
        nn.init.zeros_(self.out.bias)

    def hidden_map(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden_map(x))


def make_model(width: int, depth: int, gain: float) -> TanhMLP:
    return TanhMLP(input_dim=784, width=width, depth=depth, output_dim=10, gain=gain)
