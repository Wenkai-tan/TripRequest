"""Matplotlib renderers: choropleth (per-zone scalar) and flow map (OD lines).

Both take the GeoDataFrame from ``geovis.data.load_zones`` (indexed by
``LocationID``) so zone values join positionally without ambiguity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def _as_series(values, zones) -> pd.Series:
    """Coerce values into a Series aligned to ``zones.index`` (LocationID)."""
    if isinstance(values, pd.Series):
        return values.reindex(zones.index)
    values = np.asarray(values)
    if values.shape[0] != len(zones):
        raise ValueError(
            f"values length {values.shape[0]} != n_zones {len(zones)}"
        )
    return pd.Series(values, index=zones.index)


def plot_choropleth(zones, values, *, ax=None, cmap: str = "YlOrRd",
                    log: bool = False, title: str | None = None,
                    legend_label: str = "trip requests", figsize=(9, 11),
                    edgecolor: str = "white", linewidth: float = 0.3):
    """Colour each taxi zone by a scalar ``values`` (length N or a Series).

    ``log=True`` switches to a logarithmic colour scale — recommended for
    request counts, which are heavily concentrated in Manhattan.

    ``edgecolor`` / ``linewidth`` control the zone borders. For a darker or
    lighter border, pass any matplotlib colour — e.g. greys from light to
    dark: ``"whitesmoke"``, ``"lightgrey"``, ``"grey"``, ``"dimgrey"``, or a
    hex code such as ``"#bbbbbb"``. Set ``linewidth=0`` to hide borders.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    gdf = zones.copy()
    gdf["_value"] = _as_series(values, zones)

    norm = None
    if log:
        positive = gdf["_value"][gdf["_value"] > 0]
        vmin = float(positive.min()) if len(positive) else 1.0
        vmax = float(gdf["_value"].max()) or 1.0
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        # LogNorm cannot render 0/NaN — draw those zones as the "missing" colour.
        gdf.loc[gdf["_value"] <= 0, "_value"] = np.nan

    gdf.plot(
        column="_value", ax=ax, cmap=cmap, norm=norm,
        legend=True, legend_kwds={"label": legend_label, "shrink": 0.6},
        edgecolor=edgecolor, linewidth=linewidth,
        missing_kwds={"color": "lightgrey", "label": "no requests"},
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)
    return ax


def _bezier(p0, p1, curvature: float, n: int = 24) -> np.ndarray:
    """Quadratic Bézier arc from p0 to p1; control point offset sideways."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    mid = (p0 + p1) / 2.0
    delta = p1 - p0
    normal = np.array([-delta[1], delta[0]])
    ctrl = mid + curvature * normal
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    return pts


def _zoom_bbox(zones, location_ids, pad: float):
    """Padded bounding box ``(minx, miny, maxx, maxy)`` covering the given zones."""
    ids = [i for i in location_ids if i in zones.index]
    if not ids:
        raise ValueError("zoom: none of the LocationIDs exist in `zones`")
    minx, miny, maxx, maxy = zones.loc[ids].total_bounds
    dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def _hot_bbox(zones, mat, n: int, pad: float):
    """Bounding box of the densest cluster among the ``n`` busiest zones.

    The busiest zones can include geographic outliers — e.g. JFK / LaGuardia
    rank near the top by request volume but sit far out in Queens, so their
    raw bounding box would span most of the city. Outliers are rejected with
    a median-absolute-deviation test on zone centroids, leaving the compact
    high-density core (typically Midtown Manhattan).
    """
    from .metrics import hottest_zones

    ids = [i for i in hottest_zones(mat, n) if i in zones.index]
    cent = zones.loc[ids].geometry.centroid
    xs, ys = cent.x.to_numpy(), cent.y.to_numpy()
    mx, my = np.median(xs), np.median(ys)
    madx = np.median(np.abs(xs - mx)) or 1.0
    mady = np.median(np.abs(ys - my)) or 1.0
    keep = (np.abs(xs - mx) <= 3.0 * madx) & (np.abs(ys - my) <= 3.0 * mady)
    core = [i for i, k in zip(ids, keep) if k] or ids
    return _zoom_bbox(zones, core, pad)


def plot_flow_map(zones, mat, *, k: int = 200, ax=None,
                   curvature: float = 0.15, max_linewidth: float = 6.0,
                   cmap: str = "viridis", title: str | None = None,
                   figsize=(9, 11), basemap_color: str = "whitesmoke",
                   edgecolor: str = "lightgrey", linewidth: float = 0.3,
                   zoom=None, extent=None, hot_zones: int = 15,
                   zoom_pad: float = 0.25):
    """Draw the ``k`` strongest origin->destination flows as curved arcs.

    Arc width and colour both scale with trip count. Zone polygons are drawn
    underneath as a basemap.

    ``basemap_color`` fills the zones; ``edgecolor`` / ``linewidth`` style the
    borders. Use light greys (``"whitesmoke"``, ``"lightgrey"``) for a subtle
    backdrop or darker ones (``"grey"``, ``"dimgrey"``) for sharper zones;
    ``linewidth=0`` hides borders entirely.

    Zoom — crops the *view* to declutter a dense area. The full basemap and
    all top-``k`` flows are still drawn (ranking stays global); arcs leaving
    the cropped view are simply clipped at its edge, so cross-boundary flows
    remain visible.

    ``zoom`` :
        ``"hot"`` — auto-zoom to the ``hot_zones`` busiest zones (most trip
            requests, in + out): the cluster carrying the densest flow.
        iterable of ``LocationID`` — zoom to those specific zones.
        ``None`` — no zoom (full-city view).
    ``extent`` : explicit ``(minx, miny, maxx, maxy)`` in the zones' CRS
        (EPSG:2263, feet); overrides ``zoom``.
    ``zoom_pad`` : fractional padding added around the zoom bounding box.
    """
    from .metrics import top_flows

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    zones.plot(ax=ax, color=basemap_color, edgecolor=edgecolor,
               linewidth=linewidth)
    ax.set_axis_off()

    # --- crop the view only; flows are NOT filtered ------------------------
    bbox = None
    if extent is not None:
        bbox = tuple(extent)
    elif zoom is not None:
        bbox = (_hot_bbox(zones, mat, hot_zones, zoom_pad)
                if isinstance(zoom, str)
                else _zoom_bbox(zones, list(zoom), zoom_pad))
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    flows = top_flows(mat, k=k, drop_self=True)
    if flows.empty:
        if title:
            ax.set_title(title, fontsize=13)
        return ax

    centroids = zones.geometry.centroid
    cmax = float(flows["count"].max())
    norm = Normalize(vmin=0.0, vmax=cmax)
    colormap = plt.get_cmap(cmap)

    for o, d, c in flows[["o_loc", "d_loc", "count"]].itertuples(index=False):
        if o not in centroids.index or d not in centroids.index:
            continue
        p0 = (centroids[o].x, centroids[o].y)
        p1 = (centroids[d].x, centroids[d].y)
        arc = _bezier(p0, p1, curvature)
        ax.plot(arc[:, 0], arc[:, 1],
                linewidth=0.5 + max_linewidth * (c / cmax),
                color=colormap(norm(c)), alpha=0.55, solid_capstyle="round")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(sm, ax=ax, shrink=0.6, label="trip requests (per OD pair)")
    if title:
        ax.set_title(title, fontsize=13)
    return ax
