"""Feature engineering for the OD forecasting ablation study.

Three *cumulative* feature sets, selected by the config key ``feature_set``:

    raw  : per-(t, v) outflow, inflow, and the flow toward this origin's
           top-K destinations (the top-K set is frozen on the training data).
    time : raw + calendar encoding (hour, day-of-week, month, US holidays).
    stat : time + long-term statistics fit on the training range only
           (historical averages, last-week lag, short rolling mean).

Leakage policy
--------------
`FeatureEngineer.fit()` looks at the *training slot range only*. The frozen
artifacts (top-K destinations, historical-average tables) are then applied to
every split by `transform()`. `FeatureScaler` likewise fits its mean/std on the
training range and is reused everywhere. Persist both with each run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

N_ZONES = 263
SLOTS_PER_WEEK = 96 * 7      # 672
TOP_K = 5


# --------------------------------------------------------------------------
# calendar / time encoding
# --------------------------------------------------------------------------
def time_encoding(slot_starts) -> np.ndarray:
    """Calendar features broadcast-ready over nodes.

    Returns ``(T, 12)``: ``[hour_sin, hour_cos, dow_one_hot(7),
    month_sin, month_cos, is_holiday]``.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(slot_starts)))
    T = len(ts)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    month = ts.month.to_numpy()

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_oh = np.zeros((T, 7), dtype=np.float32)
    dow_oh[np.arange(T), dow] = 1.0
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    cal = USFederalHolidayCalendar()
    hol = cal.holidays(start=ts.min().normalize(), end=ts.max().normalize())
    is_holiday = np.isin(ts.normalize().to_numpy(),
                         np.asarray(hol.to_numpy())).astype(np.float32)

    return np.column_stack([
        hour_sin, hour_cos, dow_oh, month_sin, month_cos, is_holiday,
    ]).astype(np.float32)


# --------------------------------------------------------------------------
# feature engineer
# --------------------------------------------------------------------------
class FeatureEngineer:
    """Builds the ``(T, N, F)`` feature tensor for a chosen feature set."""

    RAW_F = 2 + TOP_K     # outflow, inflow, top-K dest flow  -> 7
    TIME_F = 12
    STAT_F = 4

    def __init__(self, feature_set: str = "stat", n_nodes: int = N_ZONES,
                 top_k: int = TOP_K):
        if feature_set not in ("raw", "time", "stat"):
            raise ValueError(f"unknown feature_set: {feature_set}")
        self.feature_set = feature_set
        self.n_nodes = n_nodes
        self.top_k = top_k
        self._fitted = False
        # frozen, training-only artifacts:
        self.top_dest_idx = None      # (N, top_k) int64
        self.hist_dow_hour = None     # (7, 24, N) float32
        self.hist_month = None        # (12, N)    float32

    # ----- introspection ---------------------------------------------------
    @property
    def n_features(self) -> int:
        f = self.RAW_F
        if self.feature_set in ("time", "stat"):
            f += self.TIME_F
        if self.feature_set == "stat":
            f += self.STAT_F
        return f

    def continuous_indices(self) -> list[int]:
        """Column indices that `FeatureScaler` should standardize.

        Counts (raw block) and long-term stats (stat block) are standardized;
        cyclical / one-hot / binary calendar columns are left as-is.
        """
        idx = list(range(self.RAW_F))
        if self.feature_set == "stat":
            base = self.RAW_F + self.TIME_F
            idx += list(range(base, base + self.STAT_F))
        return idx

    def feature_names(self) -> list[str]:
        names = ["outflow", "inflow"]
        names += [f"top{ k+1 }_dest_flow" for k in range(self.top_k)]
        if self.feature_set in ("time", "stat"):
            names += ["hour_sin", "hour_cos"]
            names += [f"dow_{d}" for d in range(7)]
            names += ["month_sin", "month_cos", "is_holiday"]
        if self.feature_set == "stat":
            names += ["hist_avg_dow_hour", "hist_avg_month",
                      "last_week_same_slot", "rolling_mean_4"]
        return names

    # ----- fit / transform -------------------------------------------------
    def fit(self, tensor: np.ndarray, slot_starts, train_range):
        """Freeze training-only artifacts. `train_range` = ``(start, end)``."""
        s, e = train_range
        tr = tensor[s:e]                                   # (T_tr, N, N)

        # Top-K destinations per origin, by total training flow.
        od_total = tr.sum(axis=0).astype(np.int64)         # (N, N)
        order = np.argsort(-od_total, axis=1)
        self.top_dest_idx = order[:, :self.top_k].astype(np.int64)

        if self.feature_set == "stat":
            outflow = tr.sum(axis=2).astype(np.float64)    # (T_tr, N)
            ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(slot_starts)[s:e]))
            dow = ts.dayofweek.to_numpy()
            hour = ts.hour.to_numpy()
            month = ts.month.to_numpy()

            ssum = np.zeros((7, 24, self.n_nodes))
            scnt = np.zeros((7, 24, 1))
            np.add.at(ssum, (dow, hour), outflow)
            np.add.at(scnt, (dow, hour), 1.0)
            self.hist_dow_hour = (ssum / np.maximum(scnt, 1.0)).astype(np.float32)

            msum = np.zeros((12, self.n_nodes))
            mcnt = np.zeros((12, 1))
            np.add.at(msum, (month - 1,), outflow)
            np.add.at(mcnt, (month - 1,), 1.0)
            self.hist_month = (msum / np.maximum(mcnt, 1.0)).astype(np.float32)

        self._fitted = True
        return self

    def transform(self, tensor: np.ndarray, slot_starts) -> np.ndarray:
        """Build the ``(T, N, F)`` feature tensor for the given slot range."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called first")
        T, N, _ = tensor.shape

        outflow = tensor.sum(axis=2).astype(np.float32)    # (T, N)
        inflow = tensor.sum(axis=1).astype(np.float32)     # (T, N)
        # flow toward each origin's frozen top-K destinations -> (T, N, top_k)
        topk = tensor[:, np.arange(N)[:, None], self.top_dest_idx]
        topk = topk.astype(np.float32)

        blocks = [outflow[..., None], inflow[..., None], topk]
        feats = [np.concatenate(blocks, axis=-1)]          # (T, N, RAW_F)

        if self.feature_set in ("time", "stat"):
            enc = time_encoding(slot_starts)               # (T, 12)
            feats.append(np.broadcast_to(enc[:, None, :], (T, N, self.TIME_F)))

        if self.feature_set == "stat":
            ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(slot_starts)))
            dow = ts.dayofweek.to_numpy()
            hour = ts.hour.to_numpy()
            month = ts.month.to_numpy()
            hist_dh = self.hist_dow_hour[dow, hour]        # (T, N)
            hist_m = self.hist_month[month - 1]            # (T, N)

            last_week = np.zeros((T, N), dtype=np.float32)
            if T > SLOTS_PER_WEEK:
                last_week[SLOTS_PER_WEEK:] = outflow[:-SLOTS_PER_WEEK]

            rolling = np.zeros((T, N), dtype=np.float32)
            for lag in (1, 2, 3, 4):
                rolling[lag:] += outflow[:-lag]
            rolling /= 4.0

            stat = np.stack([hist_dh, hist_m, last_week, rolling], axis=-1)
            feats.append(stat.astype(np.float32))

        return np.concatenate(feats, axis=-1).astype(np.float32)

    def fit_transform(self, tensor, slot_starts, train_range) -> np.ndarray:
        return self.fit(tensor, slot_starts, train_range).transform(tensor, slot_starts)

    # ----- persistence -----------------------------------------------------
    def save(self, path):
        np.savez(
            path,
            feature_set=self.feature_set, n_nodes=self.n_nodes, top_k=self.top_k,
            top_dest_idx=self.top_dest_idx,
            hist_dow_hour=(self.hist_dow_hour if self.hist_dow_hour is not None
                           else np.zeros(0, np.float32)),
            hist_month=(self.hist_month if self.hist_month is not None
                        else np.zeros(0, np.float32)),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        fe = cls(feature_set=str(d["feature_set"]),
                 n_nodes=int(d["n_nodes"]), top_k=int(d["top_k"]))
        fe.top_dest_idx = d["top_dest_idx"]
        if d["hist_dow_hour"].size:
            fe.hist_dow_hour = d["hist_dow_hour"]
        if d["hist_month"].size:
            fe.hist_month = d["hist_month"]
        fe._fitted = True
        return fe


# --------------------------------------------------------------------------
# scaler
# --------------------------------------------------------------------------
class FeatureScaler:
    """Z-score standardization of selected columns, fit on training data only."""

    def __init__(self, continuous_idx):
        self.continuous_idx = list(continuous_idx)
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, X: np.ndarray, train_range):
        """Fit mean/std using only ``X[start:end]`` (the training slots)."""
        s, e = train_range
        sub = X[s:e][..., self.continuous_idx]
        sub = sub.reshape(-1, len(self.continuous_idx))
        self.mean = sub.mean(axis=0).astype(np.float32)
        self.std = (sub.std(axis=0) + 1e-6).astype(np.float32)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureScaler.fit() must be called first")
        out = X.copy()
        out[..., self.continuous_idx] = (
            (out[..., self.continuous_idx] - self.mean) / self.std)
        return out

    def fit_transform(self, X, train_range) -> np.ndarray:
        return self.fit(X, train_range).transform(X)

    def save(self, path):
        Path(path).write_text(json.dumps({
            "continuous_idx": self.continuous_idx,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        sc = cls(d["continuous_idx"])
        sc.mean = np.asarray(d["mean"], dtype=np.float32)
        sc.std = np.asarray(d["std"], dtype=np.float32)
        sc._fitted = True
        return sc
