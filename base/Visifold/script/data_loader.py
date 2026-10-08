from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class StandardScaler:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


class TrafficDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        time_of_day: np.ndarray,
        day_of_week: np.ndarray,
        in_steps: int,
        out_steps: int,
        scaler: StandardScaler,
    ) -> None:
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.time_of_day = np.ascontiguousarray(time_of_day, dtype=np.float32)
        self.day_of_week = np.ascontiguousarray(day_of_week, dtype=np.int64)
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.scaler = scaler
        self.length = len(values) - in_steps - out_steps + 1
        if self.length <= 0:
            raise ValueError("A data split is shorter than in_steps + out_steps")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        history_end = index + self.in_steps
        target_end = history_end + self.out_steps

        values = self.scaler.transform(self.values[index:history_end])
        node_count = values.shape[1]
        time_of_day = np.broadcast_to(
            self.time_of_day[index:history_end, None], values.shape
        )
        day_of_week = np.broadcast_to(
            self.day_of_week[index:history_end, None], values.shape
        )
        history = np.stack((values, time_of_day, day_of_week), axis=-1).astype(
            np.float32, copy=False
        )
        target = self.values[history_end:target_end]
        if target.shape != (self.out_steps, node_count):
            raise RuntimeError("Unexpected target shape")
        return torch.from_numpy(history), torch.from_numpy(target)


def load_pems_bay(
    csv_path: Path, steps_per_day: int = 288
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"PEMS-BAY CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        first_row = next(reader)
    sensor_ids = header[1:]
    if not sensor_ids:
        raise ValueError(f"No sensor columns found in {csv_path}")
    first_timestamp = datetime.fromisoformat(first_row[0])

    values = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=1,
        usecols=range(1, len(header)),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[1] != len(sensor_ids):
        raise ValueError(f"Unexpected PEMS-BAY data shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("PEMS-BAY contains NaN or infinite values")

    minutes_per_step = 24 * 60 // steps_per_day
    first_step = (first_timestamp.hour * 60 + first_timestamp.minute) // minutes_per_step
    absolute_steps = first_step + np.arange(len(values), dtype=np.int64)
    time_of_day = (absolute_steps % steps_per_day).astype(np.float32) / steps_per_day
    day_of_week = (
        first_timestamp.weekday() + absolute_steps // steps_per_day
    ) % 7
    return values, time_of_day, day_of_week, sensor_ids


def load_visifold_npz(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load the paper authors' ``[flow, time-of-day, day-of-week]`` files."""

    if not data_path.is_file():
        raise FileNotFoundError(f"VisiFold data file not found: {data_path}")
    with np.load(data_path) as archive:
        data = archive["data"].astype(np.float32)
    if data.ndim != 3 or data.shape[-1] < 3:
        raise ValueError(
            f"Expected [timestamps, nodes, >=3] in {data_path}, got {data.shape}"
        )

    values = np.ascontiguousarray(data[..., 0])
    time_of_day = np.ascontiguousarray(data[:, 0, 1])
    day_of_week = np.ascontiguousarray(data[:, 0, 2], dtype=np.int64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite traffic values found in {data_path}")
    if not ((0.0 <= time_of_day).all() and (time_of_day < 1.0).all()):
        raise ValueError(f"Invalid time-of-day feature in {data_path}")
    if not ((0 <= day_of_week).all() and (day_of_week < 7).all()):
        raise ValueError(f"Invalid day-of-week feature in {data_path}")
    sensor_ids = [str(index) for index in range(values.shape[1])]
    return values, time_of_day, day_of_week, sensor_ids


def load_dataset(
    data_root: Path, dataset: str, steps_per_day: int = 288
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dataset = dataset.upper()
    if dataset == "PEMS-BAY":
        return load_pems_bay(
            data_root / "PEMS-BAY" / "PEMS-BAY.csv", steps_per_day
        )
    if dataset in {"PEMS04", "PEMS08"}:
        return load_visifold_npz(data_root / dataset / "data.npz")
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_datasets(
    values: np.ndarray,
    time_of_day: np.ndarray,
    day_of_week: np.ndarray,
    in_steps: int,
    out_steps: int,
) -> tuple[TrafficDataset, TrafficDataset, TrafficDataset, StandardScaler]:
    """Use the paper's chronological 60/20/20 split without boundary leakage."""

    train_end = int(len(values) * 0.6)
    validation_end = int(len(values) * 0.8)
    train_values = values[:train_end]
    scaler = StandardScaler(
        mean=float(train_values.mean(dtype=np.float64)),
        std=float(train_values.std(dtype=np.float64)),
    )
    if scaler.std == 0.0:
        raise ValueError("Training data standard deviation is zero")

    slices = (
        slice(0, train_end),
        slice(train_end, validation_end),
        slice(validation_end, len(values)),
    )
    datasets = tuple(
        TrafficDataset(
            values[data_slice],
            time_of_day[data_slice],
            day_of_week[data_slice],
            in_steps,
            out_steps,
            scaler,
        )
        for data_slice in slices
    )
    return datasets[0], datasets[1], datasets[2], scaler
