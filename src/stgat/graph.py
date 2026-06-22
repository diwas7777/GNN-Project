from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


def _to_numpy_adjacency(payload) -> np.ndarray:
    if isinstance(payload, (tuple, list)) and len(payload) == 3:
        payload = payload[2]
    adjacency = np.asarray(payload, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {adjacency.shape}")
    return adjacency


def _symmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    rowsum = adjacency.sum(axis=1)
    inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    inv_sqrt[rowsum == 0] = 0.0
    d_inv = np.diag(inv_sqrt.astype(np.float32))
    return (d_inv @ adjacency @ d_inv).astype(np.float32)


def _row_normalize(adjacency: np.ndarray) -> np.ndarray:
    rowsum = adjacency.sum(axis=1, keepdims=True)
    return np.divide(adjacency, rowsum, out=np.zeros_like(adjacency), where=rowsum != 0).astype(np.float32)


def load_adjacency(path: str | Path, adjacency_type: str = "raw") -> torch.Tensor:
    with Path(path).open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    adjacency = _to_numpy_adjacency(payload)

    if adjacency_type == "raw":
        normalized = adjacency
    elif adjacency_type == "identity":
        normalized = np.eye(adjacency.shape[0], dtype=np.float32)
    elif adjacency_type == "sym":
        normalized = _symmetric_normalize(adjacency)
    elif adjacency_type == "row":
        normalized = _row_normalize(adjacency)
    else:
        raise ValueError(f"unknown adjacency_type: {adjacency_type}")
    return torch.from_numpy(np.asarray(normalized, dtype=np.float32))
