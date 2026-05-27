"""Run logging: per-run directory artifacts and the aggregated results CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Column order for experiments/results.csv (one row per run).
RESULTS_COLUMNS = [
    "run_id", "model", "feature_set", "output_mode", "seed",
    "val_mae", "val_rmse",
    "test_mae", "test_rmse", "test_masked_mae", "test_topK_hit", "test_tcs",
    "ood_mae", "ood_rmse",
    "train_time_sec", "n_params",
]


class RunLogger:
    """Writes config / metrics artifacts into one run directory."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, obj: dict):
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        return path

    def save_yaml(self, name: str, obj: dict):
        import yaml
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, sort_keys=False)
        return path


def append_results_row(results_csv, row: dict):
    """Append one run's results row, writing the header if the file is new."""
    results_csv = Path(results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})
