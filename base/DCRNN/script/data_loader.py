from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        sample_start: int,
        sample_count: int,
        seq_len: int,
        horizon: int,
        mean: float,
        std: float,
    ) -> None:
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.sample_start = sample_start
        self.sample_count = sample_count
        self.seq_len = seq_len
        self.horizon = horizon
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.sample_start + index
        end = start + self.seq_len
        x_values = (self.values[start:end] - self.mean) / self.std
        y_values = (self.values[end : end + self.horizon] - self.mean) / self.std

        time_index = (np.arange(start, end, dtype=np.float32) % 288) / 288.0
        time_in_day = np.broadcast_to(time_index[:, None], x_values.shape)
        x = np.stack((x_values, time_in_day), axis=-1).astype(np.float32)
        y = y_values[:, :, None].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def _load_npz(data_path: Path, feature: int) -> np.ndarray:
    with np.load(data_path) as archive:
        values = archive["data"]
    if feature >= values.shape[-1]:
        raise ValueError(f"Feature index {feature} is out of range for shape {values.shape}")
    return values[:, :, feature].astype(np.float32)


def _load_pems_bay(data_path: Path) -> np.ndarray:
    with data_path.open("r", newline="") as file:
        column_count = len(next(csv.reader(file)))
    return np.loadtxt(
        data_path,
        delimiter=",",
        skiprows=1,
        usecols=range(1, column_count),
        dtype=np.float32,
    )


def _load_edge_graph(graph_path: Path, num_nodes: int) -> np.ndarray:
    edges = np.loadtxt(graph_path, delimiter=",", skiprows=1)
    distance = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    distance[src, dst] = edges[:, 2]
    finite_distance = distance[np.isfinite(distance)]
    std = finite_distance.std()
    adjacency = np.exp(-np.square(distance / std))
    adjacency[adjacency < 0.1] = 0.0
    return adjacency.astype(np.float32)


def _load_pickle_graph(graph_path: Path) -> np.ndarray:
    with graph_path.open("rb") as file:
        graph_data = pickle.load(file, encoding="latin1")
    return np.asarray(graph_data[2], dtype=np.float32)


def load_dataset(data_root: Path, dataset: str, feature: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = dataset.upper()
    dataset_dir = data_root / dataset
    if dataset in {"PEMS04", "PEMS08"}:
        values = _load_npz(dataset_dir / f"{dataset}.npz", feature)
        adjacency = _load_edge_graph(dataset_dir / f"{dataset}.csv", values.shape[1])
    elif dataset == "PEMS-BAY":
        values = _load_pems_bay(dataset_dir / "PEMS-BAY.csv")
        adjacency = _load_pickle_graph(dataset_dir / "adj_mx_bay.pkl")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if adjacency.shape != (values.shape[1], values.shape[1]):
        raise ValueError(
            f"Graph shape {adjacency.shape} does not match {values.shape[1]} sensors"
        )
    return values, adjacency


def build_datasets(
    values: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[TrafficDataset, TrafficDataset, TrafficDataset, float, float]:
    num_samples = len(values) - seq_len - horizon + 1
    if num_samples <= 0:
        raise ValueError("The sequence is shorter than seq_len + horizon")

    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_train - num_test

    train_windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=seq_len, axis=0
    )[:num_train]
    mean = float(train_windows.mean(dtype=np.float64))
    std = float(train_windows.std(dtype=np.float64))
    if std == 0.0:
        raise ValueError("Training data standard deviation is zero")

    train = TrafficDataset(values, 0, num_train, seq_len, horizon, mean, std)
    val = TrafficDataset(values, num_train, num_val, seq_len, horizon, mean, std)
    test = TrafficDataset(values, num_train + num_val, num_test, seq_len, horizon, mean, std)
    return train, val, test, mean, std


def _random_walk(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inverse_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inverse_degree[nonzero] = 1.0 / degree[nonzero]
    return inverse_degree[:, None] * adjacency


def _to_sparse_tensor(matrix: np.ndarray) -> torch.Tensor:
    row, col = np.nonzero(matrix)
    indices = torch.from_numpy(np.stack((row, col)).astype(np.int64))
    values = torch.from_numpy(matrix[row, col].astype(np.float32))
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def dual_random_walk_supports(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    forward = _random_walk(adjacency).T
    backward = _random_walk(adjacency.T).T
    return _to_sparse_tensor(forward), _to_sparse_tensor(backward)
