"""YAML config loading with single-level ``extends`` inheritance.

An ablation config carries ``extends: configs/base.yaml`` (a path relative to
the project root). The base is loaded first and the child is deep-merged on
top. Project root is auto-detected as the nearest ancestor containing both
``src/`` and ``configs/``.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

_REQUIRED_KEYS = ["model", "feature_set", "forecast", "training"]


def find_project_root(start=None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "src").is_dir() and (d / "configs").is_dir():
            return d
    return p


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` onto a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_one(path: Path, root: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    ext = raw.pop("extends", None)
    if ext:
        base = _load_one((root / ext).resolve(), root)
        raw = deep_merge(base, raw)
    return raw


def validate_config(cfg: dict):
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config is missing required keys: {missing}")
    om = cfg["model"].get("output_mode", "two_head")
    if om not in ("two_head", "direct"):
        raise ValueError(f"invalid output_mode: {om}")
    if cfg["feature_set"] not in ("raw", "time", "stat"):
        raise ValueError(f"invalid feature_set: {cfg['feature_set']}")
    return cfg


def load_config(path, root=None) -> dict:
    """Load a config (resolving ``extends``) and validate it.

    The resolved project root is stored under ``cfg['project_root']`` so
    callers can resolve data paths consistently.
    """
    path = Path(path).resolve()
    root = Path(root).resolve() if root else find_project_root(path.parent)
    cfg = _load_one(path, root)
    cfg["project_root"] = str(root)
    cfg["config_path"] = str(path)
    return validate_config(cfg)
