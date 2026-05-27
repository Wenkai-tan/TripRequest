"""Evaluation metrics for OD prediction.

All metrics operate on the final OD prediction ``D_hat`` and ground truth
``D_true``, both ``(B, T_out, N, N)``, so they are identical across output
modes. The :class:`MetricAccumulator` streams over batches to avoid holding
the whole test set in memory.

TCS
---
The spec lists ``TCS`` without a definition. We implement **Total Count
Similarity**: ``1 - |sum(D_hat) - sum(D_true)| / sum(D_true)``, i.e. how well
the model preserves the aggregate trip volume (1.0 = perfect). Revisit if the
project later pins down a different meaning.
"""
from __future__ import annotations

import torch

_EPS = 1e-8


def mae(pred, true) -> float:
    return (pred - true).abs().mean().item()


def rmse(pred, true) -> float:
    return torch.sqrt(((pred - true) ** 2).mean()).item()


def masked_mae(pred, true, mask=None) -> float:
    """MAE over entries where `mask` is True (defaults to ``true > 0``)."""
    if mask is None:
        mask = true > 0
    m = mask.to(pred.dtype)
    denom = m.sum().clamp_min(1.0)
    return ((pred - true).abs() * m).sum().item() / denom.item()


def topk_hit_rate(D_hat, D_true, k: int = 10) -> float:
    """Mean overlap between predicted and true top-K destinations per row."""
    hit, rows = _topk_hit_counts(D_hat, D_true, k)
    return hit / max(rows, 1)


def _topk_hit_counts(D_hat, D_true, k: int):
    """Streaming-friendly: returns ``(hit_sum, valid_row_count)``."""
    N = D_hat.shape[-1]
    k = min(k, N)
    row_sum = D_true.sum(dim=-1)                       # (B, T_out, N)
    valid = row_sum > 0
    th = D_hat.topk(k, dim=-1).indices                 # (B, T_out, N, k)
    tt = D_true.topk(k, dim=-1).indices
    # intersection size per row
    inter = (th.unsqueeze(-1) == tt.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    inter = inter.to(D_hat.dtype) / k                  # (B, T_out, N)
    return inter[valid].sum().item(), int(valid.sum().item())


def tcs(D_hat, D_true) -> float:
    """Total Count Similarity (see module docstring)."""
    sh = D_hat.sum()
    st = D_true.sum()
    return (1.0 - (sh - st).abs() / st.clamp_min(_EPS)).item()


class MetricAccumulator:
    """Streams batch-level statistics into the final metric dict."""

    def __init__(self, topk: int = 10):
        self.topk = topk
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0
        self.m_abs_sum = 0.0
        self.m_count = 0
        self.hit_sum = 0.0
        self.row_count = 0
        self.total_hat = 0.0
        self.total_true = 0.0

    @torch.no_grad()
    def update(self, D_hat, D_true):
        err = (D_hat - D_true).abs()
        self.abs_sum += err.sum().item()
        self.sq_sum += ((D_hat - D_true) ** 2).sum().item()
        self.count += D_hat.numel()

        mask = D_true > 0
        self.m_abs_sum += (err * mask).sum().item()
        self.m_count += int(mask.sum().item())

        hit, rows = _topk_hit_counts(D_hat, D_true, self.topk)
        self.hit_sum += hit
        self.row_count += rows

        self.total_hat += D_hat.sum().item()
        self.total_true += D_true.sum().item()

    def compute(self) -> dict:
        c = max(self.count, 1)
        return {
            "mae": self.abs_sum / c,
            "rmse": (self.sq_sum / c) ** 0.5,
            "masked_mae": self.m_abs_sum / max(self.m_count, 1),
            "topk_hit": self.hit_sum / max(self.row_count, 1),
            "tcs": 1.0 - abs(self.total_hat - self.total_true)
                         / max(self.total_true, _EPS),
        }
