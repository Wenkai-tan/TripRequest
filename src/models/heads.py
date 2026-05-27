"""Output heads shared by every backbone.

`TwoHead` decomposes the prediction into an outflow magnitude and a
destination distribution; `DirectHead` regresses the OD tensor directly. The
two are interchangeable so the `output_mode` ablation only swaps this module.

Parameter-count note
--------------------
The task spec sketches `DirectHead` as mean-pool -> ``Linear(hidden, T_out*N*N)``.
For N=263 that is ~17.7M parameters, dwarfing `TwoHead`'s destination head
(``Linear(hidden, T_out*N)``, ~68k) and confounding the output-mode comparison.

We therefore use the spec's documented alternative: a node-wise
``Linear(hidden, T_out*N)`` that predicts each origin's *outgoing OD row* from
that origin's hidden state. This matches `TwoHead`'s destination-head parameter
count almost exactly, so the two output modes are compared at equal model size.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class TwoHead(nn.Module):
    """Outflow head + destination-distribution head."""

    def __init__(self, hidden: int, n_nodes: int, T_out: int):
        super().__init__()
        self.outflow = nn.Linear(hidden, T_out)
        self.dest = nn.Linear(hidden, T_out * n_nodes)
        self.n_nodes = n_nodes
        self.T_out = T_out

    def forward(self, H):
        # H: (B, N, hidden)
        B, N, _ = H.shape
        # outflow >= 0 via softplus -> (B, T_out, N)
        outflow_hat = F.softplus(self.outflow(H)).permute(0, 2, 1)
        # destination distribution, softmax over destination axis
        dest_logits = self.dest(H).view(B, N, self.T_out, self.n_nodes)
        dest_dist_hat = F.softmax(dest_logits, dim=-1).permute(0, 2, 1, 3)
        # D_hat[b,t,v,u] = outflow[b,t,v] * dest_dist[b,t,v,u]
        D_hat = outflow_hat.unsqueeze(-1) * dest_dist_hat
        return D_hat, outflow_hat, dest_dist_hat


class DirectHead(nn.Module):
    """Single head: regress each origin's outgoing OD row directly."""

    def __init__(self, hidden: int, n_nodes: int, T_out: int):
        super().__init__()
        self.fc = nn.Linear(hidden, T_out * n_nodes)
        self.n_nodes = n_nodes
        self.T_out = T_out

    def forward(self, H):
        # H: (B, N, hidden) keyed by origin node
        B, N, _ = H.shape
        D = F.softplus(self.fc(H))                          # (B, N, T_out*N)
        D = D.view(B, N, self.T_out, self.n_nodes)          # (B, N, T_out, N)
        D_hat = D.permute(0, 2, 1, 3).contiguous()          # (B, T_out, N, N)
        return D_hat, None, None
