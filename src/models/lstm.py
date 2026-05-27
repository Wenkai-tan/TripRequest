"""LSTM baseline (no graph).

Each node's feature sequence is processed independently by a shared LSTM. The
input window is reshaped ``(B, T_in, N, F) -> (B*N, T_in, F)``; the last-layer
final hidden state becomes the per-node representation ``H: (B, N, hidden)``.

The adjacency `A` is accepted (for a uniform call signature) but ignored.
"""
from __future__ import annotations

import torch.nn as nn

from .base import BaseODModel


class LSTM_OD(BaseODModel):
    def __init__(self, n_features: int, n_nodes: int, T_in: int, T_out: int,
                 hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.0, output_mode: str = "two_head"):
        super().__init__(n_features, n_nodes, T_in, T_out,
                         hidden_dim=hidden_dim, output_mode=output_mode)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def encode(self, X, A=None):
        # X: (B, T_in, N, F)  ->  (B*N, T_in, F)
        B, T, N, Fdim = X.shape
        x = X.permute(0, 2, 1, 3).reshape(B * N, T, Fdim)
        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]                       # (B*N, hidden) — last layer
        return last.view(B, N, self.hidden_dim)
