"""Single-run training entry point.

    python scripts/train.py --config configs/ablation/lstm_stat_twohead.yaml --seed 42

Each run writes into ``experiments/runs/<run_id>/``:
    config.yaml, feature_engineer.npz, scaler.json, metrics.jsonl,
    best.ckpt, metrics.json
and appends one row to ``experiments/results.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# make `src` importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.adjacency import build_adjacency, normalize_adjacency
from src.data.dataset import build_window_loaders
from src.data.splits import split_from_dates
from src.models import build_model
from src.training import Trainer, build_loss
from src.utils import RunLogger, append_results_row, load_config, set_seed


def load_tensor_and_splits(cfg):
    """Load the dense OD tensor + slot index, and the chronological splits."""
    root = Path(cfg["project_root"])
    npz = np.load(root / cfg["data"]["tensor_path"])
    tensor, slot_starts = npz["tensor"], npz["slot_starts"]

    splits_path = root / cfg["data"]["splits_path"]
    if splits_path.exists():
        meta = json.loads(Path(splits_path).read_text())
        splits = {k: tuple(v) for k, v in meta["split_idx"].items()}
    else:
        print(f"[warn] {splits_path} not found; recomputing from default dates")
        splits = split_from_dates(slot_starts)
    return tensor, slot_starts, splits


def run_training(cfg: dict, seed: int):
    """Execute one run end to end; returns ``(metrics, results_row)``."""
    root = Path(cfg["project_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    print(f"device={device}  seed={seed}")

    # ---- data ------------------------------------------------------------
    tensor, slot_starts, splits = load_tensor_and_splits(cfg)
    n_zones = cfg["data"]["n_zones"]
    fc = cfg["forecast"]
    loader_device = cfg["data"].get("loader_device", "cpu")
    loaders, fe, scaler = build_window_loaders(
        tensor, slot_starts, splits,
        feature_set=cfg["feature_set"],
        T_in=fc["T_in"], T_out=fc["T_out"],
        batch_size=cfg["training"]["batch_size"],
        n_nodes=n_zones, stride=fc["stride"],
        device=loader_device,
    )
    print(f"  loader_device={loader_device}")
    for name, ld in loaders.items():
        print(f"  {name:5s}: {ld.n_windows} windows")

    # ---- adjacency (fit on train range; LSTM ignores it) -----------------
    A_t = None
    if cfg["adjacency"] in ("flow", "geo", "hybrid"):
        try:
            A = build_adjacency(cfg["adjacency"], tensor=tensor,
                                train_range=splits["train"])
            A_t = torch.from_numpy(normalize_adjacency(A))
        except Exception as exc:                       # geo needs a shapefile
            print(f"[warn] adjacency '{cfg['adjacency']}' unavailable: {exc}")

    # ---- model + loss ----------------------------------------------------
    model = build_model(cfg, n_features=fe.n_features, n_nodes=n_zones)
    n_params = model.count_parameters()
    print(f"  model={cfg['model']['name']} "
          f"output_mode={cfg['model']['output_mode']} "
          f"F={fe.n_features} params={n_params:,}")
    lw = cfg["loss"]
    loss_fn = build_loss(cfg["model"]["output_mode"],
                         lam1=lw["lam1"], lam2=lw["lam2"], lam3=lw["lam3"])

    # ---- run directory ---------------------------------------------------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (f"{cfg['model']['name']}_{cfg['feature_set']}_"
              f"{cfg['model']['output_mode']}_s{seed}_{stamp}")
    run_dir = root / cfg["experiments_dir"] / "runs" / run_id
    logger = RunLogger(run_dir)
    logger.save_yaml("config.yaml", {k: v for k, v in cfg.items()
                                     if k != "project_root"})
    scaler.save(run_dir / "scaler.json")
    fe.save(run_dir / "feature_engineer.npz")

    # ---- train -----------------------------------------------------------
    trainer = Trainer(model, loaders, loss_fn, cfg, run_dir, device,
                      adjacency=A_t)
    t0 = time.time()
    best_val = trainer.fit()
    train_time = time.time() - t0
    trainer.load_best()

    # ---- evaluate --------------------------------------------------------
    val_m = trainer.evaluate(loaders["val"])
    test_m = trainer.evaluate(loaders["test"])
    ood_m = trainer.evaluate(loaders["ood"])
    metrics = {"best_val_loss": best_val, "train_time_sec": train_time,
               "n_params": n_params, "val": val_m, "test": test_m,
               "ood": ood_m}
    logger.save_json("metrics.json", metrics)

    row = {
        "run_id": run_id, "model": cfg["model"]["name"],
        "feature_set": cfg["feature_set"],
        "output_mode": cfg["model"]["output_mode"], "seed": seed,
        "val_mae": val_m["mae"], "val_rmse": val_m["rmse"],
        "test_mae": test_m["mae"], "test_rmse": test_m["rmse"],
        "test_masked_mae": test_m["masked_mae"],
        "test_topK_hit": test_m["topk_hit"], "test_tcs": test_m["tcs"],
        "ood_mae": ood_m["mae"], "ood_rmse": ood_m["rmse"],
        "train_time_sec": round(train_time, 1), "n_params": n_params,
    }
    append_results_row(root / cfg["experiments_dir"] / "results.csv", row)

    print(f"\n[done] {run_id}")
    print(f"  test  MAE={test_m['mae']:.4f} RMSE={test_m['rmse']:.4f} "
          f"topK={test_m['topk_hit']:.3f} TCS={test_m['tcs']:.3f}")
    print(f"  ood   MAE={ood_m['mae']:.4f} RMSE={ood_m['rmse']:.4f}")
    return metrics, row


def main():
    ap = argparse.ArgumentParser(description="Single-run OD forecasting trainer")
    ap.add_argument("--config", required=True, help="path to a yaml config")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides config['seed']")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    run_training(cfg, seed)


if __name__ == "__main__":
    main()
