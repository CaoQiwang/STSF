from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, values: np.ndarray, n_his: int, n_pred: int) -> None:
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.n_his = n_his
        self.n_pred = n_pred
        self.length = len(values) - n_his - n_pred + 1
        if self.length <= 0:
            raise ValueError("The selected split is shorter than n_his + n_pred")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.n_his, :, None]
        y = self.values[index + self.n_his : index + self.n_his + self.n_pred]
        return torch.from_numpy(x), torch.from_numpy(y)


def _load_pems_npz(data_path: Path, feature: int) -> np.ndarray:
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


def _load_edge_graph(
    graph_path: Path,
    num_nodes: int,
    sigma2: float = 0.1,
    epsilon: float = 0.5,
) -> np.ndarray:
    edges = np.loadtxt(graph_path, delimiter=",", skiprows=1)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    distance = edges[:, 2] / 10000.0
    weight = np.exp(-(distance**2) / sigma2)
    weight[weight < epsilon] = 0.0
    adjacency[src, dst] = weight
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _load_pickle_graph(graph_path: Path) -> np.ndarray:
    with graph_path.open("rb") as file:
        graph_data = pickle.load(file, encoding="latin1")
    adjacency = np.asarray(graph_data[2], dtype=np.float32)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def load_dataset(
    data_root: Path,
    dataset: str,
    feature: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = dataset.upper()
    dataset_dir = data_root / dataset

    if dataset in {"PEMS04", "PEMS08"}:
        values = _load_pems_npz(dataset_dir / f"{dataset}.npz", feature)
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


def split_and_normalize(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    train_end = int(len(values) * 0.6)
    val_end = int(len(values) * 0.8)
    train = values[:train_end]
    val = values[train_end:val_end]
    test = values[val_end:]

    mean = float(train.mean())
    std = float(train.std())
    if std == 0.0:
        raise ValueError("Training data standard deviation is zero")
    return (train - mean) / std, (val - mean) / std, (test - mean) / std, mean, std


def chebyshev_polynomials(adjacency: np.ndarray, ks: int) -> torch.Tensor:
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = degree[nonzero] ** -0.5
    laplacian = np.eye(len(adjacency), dtype=np.float32) - (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    lambda_max = float(np.linalg.eigvalsh(laplacian).max())
    scaled_laplacian = 2.0 * laplacian / lambda_max - np.eye(len(adjacency), dtype=np.float32)

    polynomials = [np.eye(len(adjacency), dtype=np.float32)]
    if ks > 1:
        polynomials.append(scaled_laplacian)
    for _ in range(2, ks):
        polynomials.append(2.0 * scaled_laplacian @ polynomials[-1] - polynomials[-2])
    return torch.from_numpy(np.stack(polynomials).astype(np.float32))
