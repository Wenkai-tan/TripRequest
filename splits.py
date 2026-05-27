"""Chronological train/val/test/OOD splits.

The OD problem is strictly causal: we never shuffle in time, and every
training-derived statistic must be fit on the training range only. This module
turns calendar dates into half-open index ranges `[start, end)` over the
continuous 15-minute slot index.

Split plan (from the task spec):

    Train: 2023-07 -> 2024-09   (~15 months)   index range starts at 0
    Val:   2024-10 -> 2024-11   (early stopping / model selection)
    Test:  2024-12              (in-distribution evaluation)
    OOD:   2025-03 -> 2025-04   (out-of-distribution robustness; reported apart)

2025-01/02 fall between Test and OOD. They are intentionally *not* assigned to
any split, but should still be present in the slot index so that the index
stays continuous and lag features (e.g. last-week) work across the OOD edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Exclusive upper bounds, expressed as month starts.
DEFAULT_SPLITS = {
    "train_start": "2023-07-01",
    "train_end":   "2024-10-01",   # Train covers ..2024-09
    "val_end":     "2024-12-01",   # Val   covers 2024-10..2024-11
    "test_end":    "2025-01-01",   # Test  covers 2024-12
    "ood_start":   "2025-03-01",
    "ood_end":     "2025-05-01",   # OOD   covers 2025-03..2025-04
}


def _to_epoch_ns(slot_starts) -> np.ndarray:
    """Coerce a slot index to a sorted int64 epoch-nanosecond array."""
    arr = np.asarray(slot_starts)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    # Fall back through pandas (handles DatetimeIndex / object arrays).
    return pd.DatetimeIndex(pd.to_datetime(arr)).asi8


def chronological_split(slot_starts,
                        train_end, val_end, test_end, ood_start, ood_end,
                        train_start=None):
    """Map split boundary dates to `[start, end)` index ranges.

    Parameters
    ----------
    slot_starts : array-like
        The continuous 15-minute slot index (int64 epoch-ns, datetime64, or a
        DatetimeIndex). Must be sorted ascending.
    train_end, val_end, test_end, ood_start, ood_end : date-like
        Split boundaries. All bounds are treated as half-open (exclusive end).
    train_start : date-like, optional
        Start of the training range. Defaults to slot index 0.

    Returns
    -------
    dict[str, tuple[int, int]]
        Keys ``train``, ``val``, ``test``, ``ood`` -> ``(start_idx, end_idx)``.
    """
    ss = _to_epoch_ns(slot_starts)
    if np.any(np.diff(ss) < 0):
        raise ValueError("slot_starts must be sorted ascending")

    def idx(date) -> int:
        return int(np.searchsorted(ss, pd.Timestamp(date).value, side="left"))

    train_s = 0 if train_start is None else idx(train_start)
    splits = {
        "train": (train_s,         idx(train_end)),
        "val":   (idx(train_end),  idx(val_end)),
        "test":  (idx(val_end),    idx(test_end)),
        "ood":   (idx(ood_start),  idx(ood_end)),
    }

    for name, (s, e) in splits.items():
        if e < s:
            raise ValueError(f"split '{name}' has negative span: [{s}, {e})")
        if e == s:
            # Not fatal (the month may simply be missing from the data), but
            # the caller almost certainly wants to know.
            import warnings
            warnings.warn(f"split '{name}' is empty: [{s}, {e})", stacklevel=2)
    return splits


def split_from_dates(slot_starts, split_dates=None):
    """Convenience wrapper using the :data:`DEFAULT_SPLITS` calendar."""
    d = dict(DEFAULT_SPLITS if split_dates is None else split_dates)
    return chronological_split(
        slot_starts,
        train_end=d["train_end"], val_end=d["val_end"], test_end=d["test_end"],
        ood_start=d["ood_start"], ood_end=d["ood_end"],
        train_start=d.get("train_start"),
    )
