"""Model zoo. Only the LSTM baseline is implemented so far; the remaining
backbones (transformer, stgcn, naive, hist_avg) are added during the ablation.
"""
from .base import BaseODModel          # noqa: F401
from .heads import DirectHead, TwoHead  # noqa: F401
from .lstm import LSTM_OD              # noqa: F401

_REGISTRY = {
    "lstm": LSTM_OD,
}


def build_model(config: dict, n_features: int, n_nodes: int = 263):
    """Construct a model from a parsed config dict."""
    m = dict(config["model"])
    fc = config["forecast"]
    name = m.pop("name")
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"model '{name}' is not implemented yet "
            f"(available: {sorted(_REGISTRY)})"
        )
    cls = _REGISTRY[name]
    return cls(
        n_features=n_features,
        n_nodes=n_nodes,
        T_in=fc["T_in"],
        T_out=fc["T_out"],
        hidden_dim=m.get("hidden_dim", 64),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("dropout", 0.0),
        output_mode=m.get("output_mode", "two_head"),
    )


__all__ = ["BaseODModel", "TwoHead", "DirectHead", "LSTM_OD", "build_model"]
