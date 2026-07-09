"""LSTM trajectory predictor (Eq. 13.1): (batch, L, 4) -> (batch, H, 4).

Encoder LSTM consumes the length-L input window; the final hidden state is
decoded by a linear head into H*4 outputs reshaped to (H, 4). Channel order is
[x, y, vx, vy] throughout, matching the dataset contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    """Model hyperparameters with safe defaults (docs/07: L=8, H=12)."""
    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 1
    horizon: int = 12       # H
    output_size: int = 4


class LSTMPredictor(nn.Module):
    """Encoder-LSTM + linear decoder mapping (B, L, 4) -> (B, H, 4)."""

    def __init__(self, config: LSTMConfig | None = None):
        super().__init__()
        self.config = config or LSTMConfig()
        c = self.config
        self.lstm = nn.LSTM(
            input_size=c.input_size,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(c.hidden_size, c.horizon * c.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 4) -> (B, H, 4)."""
        if x.dim() != 3 or x.size(-1) != self.config.input_size:
            raise ValueError(
                f"expected (B, L, {self.config.input_size}), got {tuple(x.shape)}"
            )
        _out, (h_n, _c_n) = self.lstm(x)
        last_hidden = h_n[-1]                       # (B, hidden)
        flat = self.head(last_hidden)               # (B, H*4)
        return flat.view(-1, self.config.horizon, self.config.output_size)
