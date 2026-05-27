"""Training driver: epoch loop, early stopping, checkpointing, evaluation.

`Trainer` is backbone- and output-mode-agnostic. It receives an already-built
model, batch iterators, and a loss closure; the same instance trains an LSTM
in `two_head` or `direct` mode unchanged.

Performance notes
-----------------
* Loaders yield ``(X, Y_OD)`` only. `Y_outflow` / `Y_dest_dist` are derived
  from `Y_OD` on the compute device, and only for `two_head` mode — this
  avoids transferring a second (B, T_out, N, N) tensor across the PCIe bus.
* Loss components are kept as on-device tensors and accumulated per epoch;
  the loop syncs to the host (`.item()`) once per epoch, not once per batch.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .metrics import MetricAccumulator


class Trainer:
    def __init__(self, model, loaders, loss_fn, config, run_dir, device,
                 adjacency=None):
        """
        loaders : dict with at least 'train' and 'val' iterators yielding
                  ``(X, Y_OD)`` batches (see WindowLoader).
        loss_fn : closure (preds, targets) -> (loss, components).
        config  : parsed config dict.
        adjacency : (N, N) tensor or None, passed to model.forward.
        """
        self.device = device
        self.model = model.to(device)
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.adj = None if adjacency is None else adjacency.to(device)
        self.two_head = config["model"].get("output_mode") == "two_head"

        tr = config["training"]
        self.max_epochs = tr["max_epochs"]
        self.patience = tr["patience"]
        self.grad_clip = tr["grad_clip"]
        self.opt = AdamW(model.parameters(), lr=tr["lr"],
                         weight_decay=tr["weight_decay"])
        self.sched = CosineAnnealingLR(self.opt, T_max=self.max_epochs)

        self.topk = config.get("eval", {}).get("topk", 10)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.ckpt_path = self.run_dir / "best.ckpt"

    # ----- internals -------------------------------------------------------
    def _to_device(self, t):
        return t.to(self.device, non_blocking=True)

    def _derive_targets(self, Y_OD):
        """Compute (outflow, dest_dist) from the OD tensor on-device.

        Zero-outflow rows fall back to a uniform destination distribution.
        Returns ``(None, None)`` in direct mode (the loss ignores them).
        """
        if not self.two_head:
            return None, None
        Y_outflow = Y_OD.sum(dim=-1)                       # (B, T_out, N)
        denom = Y_outflow.unsqueeze(-1)                    # (B, T_out, N, 1)
        uniform = Y_OD.new_full((), 1.0 / Y_OD.shape[-1])
        Y_dest = torch.where(denom > 0,
                             Y_OD / denom.clamp_min(1e-8),
                             uniform)
        return Y_outflow, Y_dest

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total = torch.zeros((), device=self.device)
        comps: dict[str, torch.Tensor] = {}
        n = 0
        torch.set_grad_enabled(train)
        for X, Y_OD in loader:
            X = self._to_device(X)
            Y_OD = self._to_device(Y_OD)
            Y_outflow, Y_dest = self._derive_targets(Y_OD)

            preds = self.model(X, self.adj)
            loss, comp = self.loss_fn(preds, (Y_OD, Y_outflow, Y_dest))
            if train:
                self.opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

            bs = X.size(0)
            n += bs
            total = total + loss.detach() * bs
            for k, v in comp.items():
                comps[k] = comps.get(k, torch.zeros((), device=self.device)) \
                    + v * bs
        torch.set_grad_enabled(True)
        n = max(n, 1)
        # single host sync for the whole epoch
        return (total / n).item(), {k: (v / n).item() for k, v in comps.items()}

    def _log(self, record: dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ----- public API ------------------------------------------------------
    def fit(self):
        best_val = math.inf
        bad_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss, train_comp = self._run_epoch(self.loaders["train"], True)
            val_loss, val_comp = self._run_epoch(self.loaders["val"], False)
            self.sched.step()

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.opt.param_groups[0]["lr"],
                "epoch_time_sec": round(time.time() - t0, 2),
                **{f"train_{k}": v for k, v in train_comp.items()},
                **{f"val_{k}": v for k, v in val_comp.items()},
            }
            self._log(record)
            print(f"  epoch {epoch:3d} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | {record['epoch_time_sec']}s")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad_epochs = 0
                torch.save({"model_state": self.model.state_dict(),
                            "epoch": epoch, "val_loss": val_loss},
                           self.ckpt_path)
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"  early stop at epoch {epoch} "
                          f"(best val {best_val:.4f})")
                    break
        return best_val

    def load_best(self):
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
        return self.model

    @torch.no_grad()
    def evaluate(self, loader):
        """Streaming metrics on `D_hat` vs `D_true` (same for both modes)."""
        self.model.eval()
        acc = MetricAccumulator(topk=self.topk)
        for X, Y_OD in loader:
            X = self._to_device(X)
            Y_OD = self._to_device(Y_OD)
            D_hat = self.model(X, self.adj)[0]
            acc.update(D_hat, Y_OD)
        return acc.compute()
