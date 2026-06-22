from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


X_OFFSETS = np.arange(-11, 1, dtype=np.int64).reshape(-1, 1)
Y_OFFSETS = np.arange(1, 13, dtype=np.int64).reshape(-1, 1)


def read_h5_speed(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read DCRNN-style pandas HDF5 speed data using h5py."""

    with h5py.File(path, "r") as f:
        if "df" in f:
            group = f["df"]
        elif "speed" in f:
            group = f["speed"]
        else:
            raise KeyError(f"{path} must contain a 'df' or 'speed' group")
        values = group["block0_values"][:].astype(np.float32)
        timestamps = group["axis1"][:].astype(np.int64)
    return values, timestamps


def time_of_day_feature(timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps_ns = timestamps_ns.astype(np.int64)
    day_ns = np.int64(24 * 60 * 60 * 1_000_000_000)
    return ((timestamps_ns % day_ns) / day_ns).astype(np.float32)


def generate_graph_seq2seq_io(
    values: np.ndarray,
    time_of_day: np.ndarray,
    x_offsets: np.ndarray = X_OFFSETS,
    y_offsets: np.ndarray = Y_OFFSETS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError(f"values must have shape [time, nodes], got {values.shape}")
    if time_of_day.shape[0] != values.shape[0]:
        raise ValueError("time_of_day length must match values time dimension")

    min_t = abs(int(x_offsets.min()))
    max_t = values.shape[0] - abs(int(y_offsets.max()))
    feature_time = np.broadcast_to(time_of_day[:, None], values.shape).astype(np.float32)
    data = np.stack([values.astype(np.float32), feature_time], axis=-1)

    x, y = [], []
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets.reshape(-1)])
        y.append(data[t + y_offsets.reshape(-1)])
    return np.stack(x, axis=0), np.stack(y, axis=0), x_offsets, y_offsets


def split_train_val_test(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sample_count = x.shape[0]
    train_count = round(sample_count * train_ratio)
    val_count = round(sample_count * val_ratio)
    return {
        "train": (x[:train_count], y[:train_count]),
        "val": (x[train_count : train_count + val_count], y[train_count : train_count + val_count]),
        "test": (x[train_count + val_count :], y[train_count + val_count :]),
    }


def generate_splits(h5_path: str | Path, output_dir: str | Path) -> dict[str, tuple[int, ...]]:
    values, timestamps = read_h5_speed(h5_path)
    x, y, x_offsets, y_offsets = generate_graph_seq2seq_io(values, time_of_day_feature(timestamps))
    splits = split_train_val_test(x, y)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for name, (split_x, split_y) in splits.items():
        np.savez_compressed(output / f"{name}.npz", x=split_x, y=split_y, x_offsets=x_offsets, y_offsets=y_offsets)
        shapes[name] = split_x.shape
    return shapes
