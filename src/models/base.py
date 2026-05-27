"""Shared model interface for the OD forecasting ablation study.

Every model subclasses :class:`BaseODModel` and therefore obeys one input
contract and one output contract, regardless of backbone or output mode:

    input  : X (B, T_in, N, F)  and  A (N, N) or None  (A ignored by non-graph
             models, but always accepted so callers stay uniform)
    output : two_head -> (D_hat, outflow_hat, dest_dist_hat)
             direct   -> (D_hat, None, None)

`D_hat` has shape (B, T_out, N, N) in both modes, so evaluation metrics are
computed identically.

Subclasses implement :meth:`encode`, which maps the input window to a per-node
hidden representation ``H: (B, N, hidden_dim)``. The output head is attached by
the base class according to ``output_mode``.
"""
from __future__ import annotations

import torch.nn as nn

from .heads import DirectHead, TwoHead


class BaseODModel(nn.Module):
    def __init__(self, n_features: int, n_nodes: int, T_in: int, T_out: int,
                 hidden_dim: int = 64, output_mode: str = "two_head"):
        super().__init__()
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.T_in = T_in
        self.T_out = T_out
        self.hidden_dim = hidden_dim
        self.output_mode = output_mode

        if output_mode == "two_head":
            self.head = TwoHead(hidden_dim, n_nodes, T_out)
        elif output_mode == "direct":
            self.head = DirectHead(hidden_dim, n_nodes, T_out)
        else:
            raise ValueError(f"unknown output_mode: {output_mode}")

    # ----- to be implemented by subclasses --------------------------------
    def encode(self, X, A=None):
        """Map X (B, T_in, N, F) -> H (B, N, hidden_dim)."""
        raise NotImplementedError

    # ----- shared forward --------------------------------------------------
    def forward(self, X, A=None):
        """X: (B, T_in, N, F); A: (N, N) or None. Returns the output triple."""
        H = self.encode(X, A)
        return self.head(H)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
