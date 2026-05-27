"""Adjacency matrix builders for the spatial graph (N = 263 taxi zones).

Three variants, selected by the config key ``adjacency``:

    flow   : data-driven — aggregated training OD flow between zones.
    geo    : geographic — Gaussian kernel over zone-centroid distances.
    hybrid : a convex blend of `geo` and `flow`.

Only `flow` is computable from the OD tensor alone. `geo` needs zone centroids
(from the TLC Taxi Zone shapefile); `load_zone_centroids` is a thin optional
helper around geopandas.

All builders return a symmetric ``(N, N)`` float32 matrix. Non-graph models
ignore it; graph models should pass it through `normalize_adjacency` first.
"""
from __future__ import annotations

import numpy as np

N_ZONES = 263


# --------------------------------------------------------------------------
# normalization helpers
# --------------------------------------------------------------------------
def normalize_adjacency(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """Symmetric normalization ``D^-1/2 (A + I) D^-1/2`` (GCN-style)."""
    A = np.asarray(A, dtype=np.float64)
    if add_self_loops:
        A = A + np.eye(A.shape[0])
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    A_norm = dinv[:, None] * A * dinv[None, :]
    return A_norm.astype(np.float32)


def _row_normalize(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(axis=1, keepdims=True)
    return (A / np.maximum(deg, 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------
# flow adjacency
# --------------------------------------------------------------------------
def build_flow_adjacency(tensor: np.ndarray, train_range,
                         symmetric: bool = True,
                         log_scale: bool = True) -> np.ndarray:
    """Aggregate training-range OD flow into a zone-to-zone adjacency.

    Fit on the training range only — the graph structure is itself a learned
    artifact and must not see val/test/OOD data.
    """
    s, e = train_range
    flow = tensor[s:e].sum(axis=0).astype(np.float64)      # (N, N)
    np.fill_diagonal(flow, 0.0)
    if symmetric:
        flow = flow + flow.T
    if log_scale:
        flow = np.log1p(flow)
    return flow.astype(np.float32)


# --------------------------------------------------------------------------
# geographic adjacency
# --------------------------------------------------------------------------
def load_zone_centroids(shapefile_path, n_zones: int = N_ZONES) -> np.ndarray:
    """Load ``(N, 2)`` zone centroids from the TLC Taxi Zone shapefile.

    Requires geopandas. Centroids are projected to an equal-distance CRS
    (EPSG:2263, NY State Plane, feet) before centroid extraction.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            "geopandas is required for geographic adjacency; install it or "
            "use adjacency='flow'."
        ) from exc

    gdf = gpd.read_file(shapefile_path).to_crs(epsg=2263)
    gdf = gdf.sort_values("LocationID")
    cent = np.full((n_zones, 2), np.nan, dtype=np.float64)
    for _, row in gdf.iterrows():
        loc = int(row["LocationID"])
        if 1 <= loc <= n_zones:
            cent[loc - 1] = (row.geometry.centroid.x, row.geometry.centroid.y)
    return cent


def build_geo_adjacency(centroids: np.ndarray, k: int = 8,
                         sigma: float | None = None) -> np.ndarray:
    """Gaussian-kernel k-NN adjacency over zone centroids.

    `centroids` is ``(N, 2)``. Missing zones (NaN centroids) get no edges.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    N = centroids.shape[0]
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))               # (N, N)

    valid = np.isfinite(dist)
    finite = dist[valid & (dist > 0)]
    if sigma is None:
        sigma = float(np.median(finite)) if finite.size else 1.0

    A = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        order = np.argsort(np.where(np.isfinite(dist[i]), dist[i], np.inf))
        for j in order[1:k + 1]:
            if not np.isfinite(dist[i, j]):
                break
            A[i, j] = np.exp(-(dist[i, j] ** 2) / (2.0 * sigma ** 2))
    A = np.maximum(A, A.T)                                  # symmetrize
    return A.astype(np.float32)


# --------------------------------------------------------------------------
# hybrid adjacency
# --------------------------------------------------------------------------
def build_hybrid_adjacency(geo: np.ndarray, flow: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """Convex blend ``alpha * geo + (1 - alpha) * flow`` of row-normalized inputs."""
    g = _row_normalize(geo).astype(np.float64)
    f = _row_normalize(flow).astype(np.float64)
    return (alpha * g + (1.0 - alpha) * f).astype(np.float32)


def build_adjacency(kind: str, tensor=None, train_range=None,
                    centroids=None, alpha: float = 0.5, **kwargs) -> np.ndarray:
    """Dispatch helper driven by the config ``adjacency`` key."""
    if kind == "flow":
        return build_flow_adjacency(tensor, train_range, **kwargs)
    if kind == "geo":
        return build_geo_adjacency(centroids, **kwargs)
    if kind == "hybrid":
        geo = build_geo_adjacency(centroids, **kwargs)
        flow = build_flow_adjacency(tensor, train_range)
        return build_hybrid_adjacency(geo, flow, alpha=alpha)
    raise ValueError(f"unknown adjacency kind: {kind}")
