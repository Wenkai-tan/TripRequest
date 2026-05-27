"""Geographic visualisation of the 15-min OD tensor on the NYC Taxi Zone map.

Two views, both keyed on TLC ``LocationID`` (1..263):

    choropleth  — per-zone scalar (outflow / inflow / netflow)
    flow map    — strongest origin->destination arcs

Typical use::

    from geovis import load_tensor, load_zones, find_slot
    from geovis import slice_window, outflow, plot_choropleth, plot_flow_map

    tensor, slots = load_tensor()
    zones = load_zones()
    mat = tensor[find_slot(slots, "2024-01-15 08:00")]
    plot_choropleth(zones, outflow(mat), log=True)
"""
from __future__ import annotations

from .data import (
    N_ZONES, ZONE_CRS, DEFAULT_TENSOR, DEFAULT_SHAPEFILE,
    load_tensor, load_zones, find_slot,
)
from .metrics import (
    slice_window, outflow, inflow, netflow, zone_series, top_flows,
    hottest_zones,
)
from .plot import plot_choropleth, plot_flow_map

__all__ = [
    "N_ZONES", "ZONE_CRS", "DEFAULT_TENSOR", "DEFAULT_SHAPEFILE",
    "load_tensor", "load_zones", "find_slot",
    "slice_window", "outflow", "inflow", "netflow", "zone_series", "top_flows",
    "hottest_zones",
    "plot_choropleth", "plot_flow_map",
]
