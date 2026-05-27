"""Loss functions for the two output modes.

`two_head_loss` combines an outflow MAE, a destination-distribution KL term,
and a reconstructed-OD MAE. `direct_loss` is a plain OD MAE.

KL orientation
--------------
The spec writes ``KL(dest_dist_hat || dest_dist_true)``. The true distribution
has exact zeros (destinations with no trips), which makes that orientation
numerically infinite. We instead use ``KL(dest_dist_true || dest_dist_hat)``:
the model output is a softmax so it is strictly positive, true-side zeros
vanish (``0 * log 0 = 0``), and minimizing it still drives ``dest_dist_hat``
toward ``dest_dist_true``. This is the standard forward-KL training objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-8


def _masked_kl(dest_hat, dest_true, mask=None):
    """KL(dest_true || dest_hat), summed over destinations, averaged over rows.

    `mask`: (B, T_out, N) bool/float, True where the row should be counted
    (typically where ``outflow_true > 0``).
    """
    dest_hat = dest_hat.clamp_min(_EPS)
    dest_true_c = dest_true.clamp_min(_EPS)
    kl = (dest_true * (torch.log(dest_true_c) - torch.log(dest_hat))).sum(dim=-1)
    if mask is None:
        return kl.mean()
    m = mask.to(kl.dtype)
    return (kl * m).sum() / m.sum().clamp_min(1.0)


def two_head_loss(D_hat, outflow_hat, dest_dist_hat,
                  D_true, outflow_true, dest_dist_true,
                  lam1=1.0, lam2=1.0, lam3=0.5, mask=None):
    """Weighted sum of outflow MAE, destination KL, and reconstructed-OD MAE.

    Returns ``(total_loss, components)`` where `components` holds detached
    scalars for logging.
    """
    mae_outflow = F.l1_loss(outflow_hat, outflow_true)
    kl_dest = _masked_kl(dest_dist_hat, dest_dist_true, mask)
    mae_od = F.l1_loss(D_hat, D_true)
    total = lam1 * mae_outflow + lam2 * kl_dest + lam3 * mae_od
    # components are detached tensors (not Python floats): the trainer
    # accumulates them on-device and syncs once per epoch, so the training
    # loop never blocks on a per-batch .item() GPU sync.
    components = {
        "mae_outflow": mae_outflow.detach(),
        "kl_dest": kl_dest.detach(),
        "mae_od": mae_od.detach(),
        "total": total.detach(),
    }
    return total, components


def direct_loss(D_hat, D_true):
    """Plain OD MAE. Returns ``(total_loss, components)``."""
    mae_od = F.l1_loss(D_hat, D_true)
    return mae_od, {"mae_od": mae_od.detach(), "total": mae_od.detach()}


def build_loss(output_mode: str, lam1=1.0, lam2=1.0, lam3=0.5):
    """Return a closure ``loss_fn(preds, targets) -> (loss, components)``.

    `preds`   = (D_hat, outflow_hat, dest_dist_hat)
    `targets` = (D_true, outflow_true, dest_dist_true)
    """
    if output_mode == "two_head":
        def loss_fn(preds, targets):
            D_hat, outflow_hat, dest_dist_hat = preds
            D_true, outflow_true, dest_dist_true = targets
            mask = outflow_true > 0
            return two_head_loss(D_hat, outflow_hat, dest_dist_hat,
                                 D_true, outflow_true, dest_dist_true,
                                 lam1=lam1, lam2=lam2, lam3=lam3, mask=mask)
        return loss_fn

    if output_mode == "direct":
        def loss_fn(preds, targets):
            D_hat = preds[0]
            D_true = targets[0]
            return direct_loss(D_hat, D_true)
        return loss_fn

    raise ValueError(f"unknown output_mode: {output_mode}")
