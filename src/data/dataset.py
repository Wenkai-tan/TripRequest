"""Sliding-window PyTorch dataset over the dense OD tensor.

A single sample is a window of `T_in` history slots and `T_out` target slots.
Windowing is strictly causal and confined to one chronological split: a window
is valid only if *both* its input and target ranges lie inside ``slot_range``.

`__getitem__` returns ``(X, Y_outflow, Y_dest_dist, Y_OD)``:

    X          : (T_in, N, F)   engineered + scaled features
    Y_outflow  : (T_out, N)     per-origin total outflow
    Y_dest_dist: (T_out, N, N)  destination distribution, rows sum to 1
                                 (zero-outflow rows fall back to uniform 1/N)
    Y_OD       : (T_out, N, N)  raw OD counts

`Y_outflow` / `Y_dest_dist` are produced even in ``direct`` output mode; the
loss simply ignores them there.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ODWindowDataset(Dataset):
    def __init__(self, tensor: np.ndarray, features: np.ndarray, slot_range,
                 T_in: int = 8, T_out: int = 4, stride: int = 1):
        """
        Parameters
        ----------
        tensor : (T, N, N) ndarray
            Dense OD counts for the full time span.
        features : (T, N, F) ndarray
            Engineered + scaled features for the full time span.
        slot_range : tuple[int, int]
            Half-open ``[start, end)`` index range for this split.
        T_in, T_out : int
            Input window length and output horizon (in slots).
        stride : int
            Step between consecutive window starts.
        """
        if tensor.shape[0] != features.shape[0]:
            raise ValueError("tensor and features must share the time axis")
        self.tensor = tensor
        self.features = features
        self.start, self.end = slot_range
        self.T_in = T_in
        self.T_out = T_out
        self.stride = stride
        self.n_nodes = tensor.shape[1]

        usable = (self.end - self.start) - (T_in + T_out)
        self.n_windows = 0 if usable < 0 else usable // stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int):
        if i < 0:
            i += self.n_windows
        if not 0 <= i < self.n_windows:
            raise IndexError(i)

        s = self.start + i * self.stride
        x0, x1 = s, s + self.T_in
        y0, y1 = x1, x1 + self.T_out

        X = self.features[x0:x1].astype(np.float32)            # (T_in, N, F)
        Y_OD = self.tensor[y0:y1].astype(np.float32)           # (T_out, N, N)
        Y_outflow = Y_OD.sum(axis=-1)                          # (T_out, N)

        denom = Y_outflow[..., None]                           # (T_out, N, 1)
        uniform = np.float32(1.0 / self.n_nodes)
        Y_dest = np.where(denom > 0,
                          Y_OD / np.maximum(denom, 1e-8),
                          uniform).astype(np.float32)

        return (
            torch.from_numpy(X),
            torch.from_numpy(Y_outflow),
            torch.from_numpy(Y_dest),
            torch.from_numpy(Y_OD),
        )

    def __repr__(self) -> str:
        return (f"ODWindowDataset(range=[{self.start},{self.end}), "
                f"T_in={self.T_in}, T_out={self.T_out}, "
                f"stride={self.stride}, n_windows={self.n_windows})")


def build_split_datasets(tensor, slot_starts, splits, feature_set="stat",
                         T_in=8, T_out=4, n_nodes=263, stride=1):
    """Fit features on the training range and window every split.

    The :class:`FeatureEngineer` and :class:`FeatureScaler` are both fit on the
    training slot range only (no leakage), then applied to the whole span. The
    same fitted scaler is returned so it can be persisted with the run.

    Returns ``(datasets, feature_engineer, scaler)`` where `datasets` maps
    ``train/val/test/ood`` to :class:`ODWindowDataset` instances.

    Note: training uses :func:`build_window_loaders` instead (much faster);
    this stays for unit tests and notebook inspection of the data contract.
    """
    from .features import FeatureEngineer, FeatureScaler

    train_range = splits["train"]
    fe = FeatureEngineer(feature_set=feature_set, n_nodes=n_nodes)
    X = fe.fit(tensor, slot_starts, train_range).transform(tensor, slot_starts)
    scaler = FeatureScaler(fe.continuous_indices())
    X = scaler.fit_transform(X, train_range)

    datasets = {
        name: ODWindowDataset(tensor, X, rng, T_in=T_in, T_out=T_out,
                              stride=stride)
        for name, rng in splits.items()
    }
    return datasets, fe, scaler


class WindowLoader:
    """Vectorized windowed batch iterator (the training-hot-path loader).

    A whole batch is assembled by a single advanced-index gather over the
    shared tensors — no per-sample Python, no per-sample numpy, and no
    DataLoader worker processes (on Windows each worker would `spawn`-copy the
    multi-GB dense tensor). Yields ``(X, Y_OD)``:

        X    : (B, T_in, N, F)   float32
        Y_OD : (B, T_out, N, N)  float32

    `Y_outflow` / `Y_dest_dist` are *not* materialized here; the trainer
    derives them from `Y_OD` on the compute device only when needed
    (`two_head` mode), which avoids the host->device transfer of a second
    (B, T_out, N, N) tensor.

    `tensor` and `features` must already be torch tensors on the desired
    device (CPU or CUDA); :func:`build_window_loaders` handles that and shares
    one copy across all splits.
    """

    def __init__(self, tensor, features, slot_range, T_in=8, T_out=4,
                 batch_size=64, stride=1, shuffle=False):
        self.tensor = tensor                       # (T, N, N) torch
        self.features = features                   # (T, N, F) torch float32
        self.device = tensor.device
        self.T_in = T_in
        self.T_out = T_out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_nodes = tensor.shape[1]

        start, end = slot_range
        usable = (end - start) - (T_in + T_out)
        n = 0 if usable < 0 else usable // stride + 1
        self.starts = start + torch.arange(n, device=self.device) * stride
        self._in_off = torch.arange(T_in, device=self.device)
        self._out_off = T_in + torch.arange(T_out, device=self.device)

    @property
    def n_windows(self) -> int:
        return int(self.starts.numel())

    def __len__(self) -> int:
        if self.n_windows == 0:
            return 0
        return (self.n_windows + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.n_windows
        if self.shuffle:
            order = torch.randperm(n, device=self.device)
        else:
            order = torch.arange(n, device=self.device)
        starts = self.starts[order]
        for i in range(0, n, self.batch_size):
            b = starts[i:i + self.batch_size]                  # (B,)
            in_idx = b[:, None] + self._in_off                 # (B, T_in)
            out_idx = b[:, None] + self._out_off               # (B, T_out)
            X = self.features[in_idx].float()                  # (B,T_in,N,F)
            Y_OD = self.tensor[out_idx].float()                # (B,T_out,N,N)
            yield X, Y_OD


def build_window_loaders(tensor, slot_starts, splits, feature_set="stat",
                         T_in=8, T_out=4, batch_size=64, n_nodes=263,
                         stride=1, device="cpu"):
    """Fit features (train-only) and build one :class:`WindowLoader` per split.

    `tensor` / scaled features are converted to torch tensors *once* and shared
    across all splits. With ``device='cuda'`` the whole tensor lives in VRAM
    (fastest, but needs enough memory — fine for month-scale pilots); with
    ``device='cpu'`` batches are gathered on the host and moved per-batch by
    the trainer.

    Returns ``(loaders, feature_engineer, scaler)``.
    """
    from .features import FeatureEngineer, FeatureScaler

    train_range = splits["train"]
    fe = FeatureEngineer(feature_set=feature_set, n_nodes=n_nodes)
    X = fe.fit(tensor, slot_starts, train_range).transform(tensor, slot_starts)
    scaler = FeatureScaler(fe.continuous_indices())
    X = scaler.fit_transform(X, train_range)

    dev = torch.device(device)
    tens = torch.as_tensor(tensor)                                  # int32
    feat = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    if dev.type != "cpu":
        tens = tens.to(dev)
        feat = feat.to(dev)

    loaders = {
        name: WindowLoader(tens, feat, rng, T_in=T_in, T_out=T_out,
                           batch_size=batch_size, stride=stride,
                           shuffle=(name == "train"))
        for name, rng in splits.items()
    }
    return loaders, fe, scaler
