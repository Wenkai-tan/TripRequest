"""Reduce OD matrices / tensors into map-ready quantities.

An OD matrix is ``(N, N)`` with rows = origin, columns = destination. The map
is a 2-D plane, so it can only carry a *scalar per zone* (choropleth) or a set
of *origin->destination flows* (flow map). This module produces both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import N_ZONES


def slice_window(tensor: np.ndarray, slots: pd.DatetimeIndex,
                 start, end, reduce: str = "sum") -> np.ndarray:
    """Collapse all slots in ``[start, end)`` into a single ``(N, N)`` matrix.

    ``reduce`` is ``"sum"`` (total requests in the window) or ``"mean"``
    (average requests per 15-min slot).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (slots >= start) & (slots < end)
    if not mask.any():
        raise ValueError(f"no slots in [{start}, {end})")
    sub = tensor[np.asarray(mask)]
    return sub.sum(axis=0) if reduce == "sum" else sub.mean(axis=0)


def outflow(mat: np.ndarray) -> np.ndarray:
    """Per-zone trip requests *originating* there (row sums)."""
    return np.asarray(mat).sum(axis=1)


def inflow(mat: np.ndarray) -> np.ndarray:
    """Per-zone trip requests *destined* there (column sums)."""
    return np.asarray(mat).sum(axis=0)


def netflow(mat: np.ndarray) -> np.ndarray:
    """Inflow minus outflow — positive = net sink, negative = net source."""
    return inflow(mat) - outflow(mat)


def hottest_zones(mat: np.ndarray, k: int = 15) -> list[int]:
    """LocationIDs of the ``k`` busiest zones by total activity (in + out).

    Used to auto-locate the densest part of the map for a zoomed flow view.
    """
    activity = inflow(mat) + outflow(mat)
    order = np.argsort(activity)[::-1][:k]
    return [int(i) + 1 for i in order]


def zone_series(values: np.ndarray) -> pd.Series:
    """Wrap a length-N array as a Series indexed by ``LocationID`` (1..N).

    This index aligns directly with the GeoDataFrame from ``load_zones``.
    """
    values = np.asarray(values)
    if values.shape[0] != N_ZONES:
        raise ValueError(f"expected {N_ZONES} values, got {values.shape[0]}")
    return pd.Series(values, index=pd.RangeIndex(1, N_ZONES + 1, name="LocationID"))


def top_flows(mat: np.ndarray, k: int = 200,
              drop_self: bool = True) -> pd.DataFrame:
    """Return the ``k`` strongest OD pairs as a tidy DataFrame.

    Columns: ``o_loc``, ``d_loc`` (1-based LocationIDs) and ``count``.
    ``drop_self`` removes intra-zone trips (origin == destination).
    """
    mat = np.asarray(mat)
    work = mat.astype(np.float64).copy()
    if drop_self:
        np.fill_diagonal(work, 0.0)

    flat = work.ravel()
    k = min(k, int((flat > 0).sum()))
    if k == 0:
        return pd.DataFrame(columns=["o_loc", "d_loc", "count"])

    top = np.argpartition(flat, -k)[-k:]
    o_idx, d_idx = np.unravel_index(top, work.shape)
    df = pd.DataFrame({
        "o_loc": o_idx + 1,
        "d_loc": d_idx + 1,
        "count": mat[o_idx, d_idx],
    })
    return df.sort_values("count", ascending=False, ignore_index=True)
