"""Data layer: chronological splits, feature engineering, windowing, adjacency.

Only lightweight (numpy/pandas) modules are re-exported here. `dataset` is left
out on purpose so that importing this package does not pull in torch — the
dataset-preparation notebook only needs `splits`.
"""
from .splits import chronological_split, DEFAULT_SPLITS  # noqa: F401

# Shared constants (the prompt fixes the OD problem geometry).
N_ZONES = 263            # NYC taxi zones 1..263; 264/265 = "Unknown" are dropped.
SLOT_MINUTES = 15
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES      # 96
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7           # 672
T_IN = 8                 # input window  = 2 h of history
T_OUT = 4                # output horizon = 1 h ahead

__all__ = [
    "chronological_split", "DEFAULT_SPLITS",
    "N_ZONES", "SLOT_MINUTES", "SLOTS_PER_DAY", "SLOTS_PER_WEEK", "T_IN", "T_OUT",
]
