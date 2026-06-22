from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class StandardScaler:
    """Standard scaler with scalar train-split statistics."""

    mean: np.float32
    std: np.float32

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrafficDataset(Dataset):
    """Dataset for pre-windowed traffic tensors."""

    def __init__(self, x: np.ndarray, y: np.ndarray, scaler: StandardScaler | None = None):
        if x.ndim != 4:
            raise ValueError(f"x must have shape [samples, time, nodes, features], got {x.shape}")
        if y.ndim != 4:
            raise ValueError(f"y must have shape [samples, horizon, nodes, features], got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must contain the same number of samples")
        if x.shape[2] != y.shape[2]:
            raise ValueError("x and y must contain the same number of nodes")

        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.scaler = scaler

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        x = np.array(self.x[index], dtype=np.float32, copy=True)
        y = np.array(self.y[index], dtype=np.float32, copy=True)
        if self.scaler is not None:
            x[..., 0] = self.scaler.transform(x[..., 0])
            y[..., 0] = self.scaler.transform(y[..., 0])
        return torch.from_numpy(x), torch.from_numpy(y)


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "x" not in data or "y" not in data:
        raise KeyError(f"{path} must contain 'x' and 'y' arrays")
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def _ensure_time_major(array: np.ndarray) -> np.ndarray:
    """Convert common DCRNN [B, T, N, F] tensors; reject unknown layouts."""

    if array.ndim != 4:
        raise ValueError(f"expected a 4D tensor, got shape {array.shape}")
    return array


def scale_speed_feature(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.float32(train_x[..., 0].mean())
    std = np.float32(train_x[..., 0].std())
    if float(std) == 0.0:
        raise ValueError("train speed feature has zero standard deviation")
    scaler = StandardScaler(mean=mean, std=std)

    def scale(array: np.ndarray) -> np.ndarray:
        scaled = array.astype(np.float32, copy=True)
        scaled[..., 0] = scaler.transform(scaled[..., 0])
        return scaled

    return scaler, scale(train_x), scale(val_x), scale(test_x)


load_npz_splits = SimpleNamespace(scale_speed_feature=scale_speed_feature)


def load_split_arrays(data_dir: str | Path) -> dict[str, np.ndarray | StandardScaler]:
    root = Path(data_dir)
    train_x, train_y = _load_npz(root / "train.npz")
    val_x, val_y = _load_npz(root / "val.npz")
    test_x, test_y = _load_npz(root / "test.npz")

    train_x = _ensure_time_major(train_x)
    val_x = _ensure_time_major(val_x)
    test_x = _ensure_time_major(test_x)
    train_y = _ensure_time_major(train_y)
    val_y = _ensure_time_major(val_y)
    test_y = _ensure_time_major(test_y)

    scaler, train_x, val_x, test_x = scale_speed_feature(train_x, val_x, test_x)
    train_y = train_y.astype(np.float32, copy=True)
    val_y = val_y.astype(np.float32, copy=True)
    test_y = test_y.astype(np.float32, copy=True)
    train_y[..., 0] = scaler.transform(train_y[..., 0])
    val_y[..., 0] = scaler.transform(val_y[..., 0])
    test_y[..., 0] = scaler.transform(test_y[..., 0])

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "scaler": scaler,
    }


def load_raw_split_arrays(data_dir: str | Path, include_test: bool = True) -> dict[str, np.ndarray | StandardScaler]:
    root = Path(data_dir)
    train_x, train_y = _load_npz(root / "train.npz")
    val_x, val_y = _load_npz(root / "val.npz")
    train_x = _ensure_time_major(train_x)
    train_y = _ensure_time_major(train_y)
    val_x = _ensure_time_major(val_x)
    val_y = _ensure_time_major(val_y)

    mean = np.float32(train_x[..., 0].mean())
    std = np.float32(train_x[..., 0].std())
    if float(std) == 0.0:
        raise ValueError("train speed feature has zero standard deviation")
    arrays: dict[str, np.ndarray | StandardScaler] = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "scaler": StandardScaler(mean=mean, std=std),
    }
    if include_test:
        test_x, test_y = _load_npz(root / "test.npz")
        arrays["test_x"] = _ensure_time_major(test_x)
        arrays["test_y"] = _ensure_time_major(test_y)
    return arrays


def make_dataloaders(
    arrays: Mapping[str, np.ndarray | StandardScaler],
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    scaler = arrays.get("scaler")
    loaders = {
        "train": DataLoader(
            TrafficDataset(arrays["train_x"], arrays["train_y"], scaler=scaler),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            TrafficDataset(arrays["val_x"], arrays["val_y"], scaler=scaler),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    if "test_x" in arrays and "test_y" in arrays:
        loaders["test"] = DataLoader(
            TrafficDataset(arrays["test_x"], arrays["test_y"], scaler=scaler),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders
