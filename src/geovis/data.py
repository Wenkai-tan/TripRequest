"""Loaders for the OD tensor and the TLC Taxi Zone geometry.

The dense tensor ``od_15min_tensor.npz`` is ``(T, N, N)`` int32 with
``N = 263``. Tensor axes are 0-based; the data convention is

    tensor index ``i``  <->  TLC ``LocationID = i + 1``

The shapefile ``taxi_zones.shp`` already contains exactly those 263 zones
(LocationID 1..263); the two "unknown" zones 264/265 are absent from both the
shapefile and the tensor, so no remapping table is required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

N_ZONES = 263

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSOR = _ROOT / "dataset" / "output" / "od_15min_tensor.npz"
DEFAULT_SHAPEFILE = _ROOT / "dataset" / "data" / "taxi_zones" / "taxi_zones.shp"

# EPSG:2263 — NY State Plane (feet); equal-distance/area, good for choropleths.
ZONE_CRS = 2263


def load_tensor(path: str | Path = DEFAULT_TENSOR) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return ``(tensor, slot_starts)``.

    ``tensor``      : ``(T, N, N)`` int32 OD counts.
    ``slot_starts`` : ``DatetimeIndex`` of length ``T``, one left-closed
                      15-min slot start per frame.
    """
    npz = np.load(path)
    tensor = npz["tensor"]
    slots = pd.to_datetime(npz["slot_starts"])
    return tensor, pd.DatetimeIndex(slots)


def load_zones(shapefile_path: str | Path = DEFAULT_SHAPEFILE,
               crs: int = ZONE_CRS):
    """Load the Taxi Zone polygons as a GeoDataFrame indexed by ``LocationID``.

    Rows are sorted by ``LocationID`` (1..263) so positional order matches
    tensor axis order. Requires geopandas.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError("geovis requires geopandas; install it first.") from exc

    gdf = gpd.read_file(shapefile_path).to_crs(epsg=crs)
    gdf = gdf.sort_values("LocationID").set_index("LocationID")
    return gdf


def find_slot(slots: pd.DatetimeIndex, timestamp: str | pd.Timestamp) -> int:
    """Return the integer index of ``timestamp`` within ``slots``.

    The timestamp is floored to the enclosing 15-min slot before lookup.
    """
    ts = pd.Timestamp(timestamp).floor("15min")
    idx = slots.get_indexer([ts])[0]
    if idx == -1:
        raise ValueError(
            f"slot {ts} is outside the tensor range "
            f"({slots[0]} .. {slots[-1]})"
        )
    return int(idx)
