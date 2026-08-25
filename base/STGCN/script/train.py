from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

STGCN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STGCN_ROOT.parents[1]
sys.path.insert(0, str(STGCN_ROOT))

from model.stgcn import STGCN
from script.data_loader import (
    TrafficDataset,
    chebyshev_polynomials,
    load_dataset,
    split_and_normalize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classic STGCN model")
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS08", "PEMS-BAY"), default="PEMS04")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument("--n-his", type=int, default=12)
    parser.add_argument("--n-pred", type=int, default=9)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--kt", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_steps(model: STGCN, history: Tensor, n_pred: int) -> Tensor:
    predictions = []
    for _ in range(n_pred):
        prediction = model(history)
        predictions.append(prediction)
        history = torch.cat((history[:, 1:], prediction[:, None, :, None]), dim=1)
    return torch.stack(predictions, dim=1)


@torch.no_grad()
def evaluate(
    model: STGCN,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    absolute_error = None
    squared_error = None
    percentage_error = None
    count = 0

    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        prediction = predict_steps(model, history, target.shape[1])
        prediction = prediction * std + mean
        target = target * std + mean
        difference = (prediction - target).abs()

        batch_absolute = difference.sum(dim=(0, 2))
        batch_squared = difference.square().sum(dim=(0, 2))
        batch_percentage = (difference / (target.abs() + 1e-5)).sum(dim=(0, 2))
        absolute_error = batch_absolute if absolute_error is None else absolute_error + batch_absolute
        squared_error = batch_squared if squared_error is None else squared_error + batch_squared
        percentage_error = (
            batch_percentage if percentage_error is None else percentage_error + batch_percentage
        )
        count += target.shape[0] * target.shape[2]

    return absolute_error / count, torch.sqrt(squared_error / count), percentage_error / count


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    values, adjacency = load_dataset(args.data_root, args.dataset, args.feature)
    train_values, val_values, test_values, mean, std = split_and_normalize(values)

    train_data = TrafficDataset(train_values, args.n_his, 1)
    val_data = TrafficDataset(val_values, args.n_his, args.n_pred)
    test_data = TrafficDataset(test_values, args.n_his, args.n_pred)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    supports = chebyshev_polynomials(adjacency, args.ks)
    model = STGCN(
        num_nodes=values.shape[1],
        cheb_polynomials=supports,
        n_his=args.n_his,
        ks=args.ks,
        kt=args.kt,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    print(
        f"dataset={args.dataset} nodes={values.shape[1]} device={device} "
        f"mean={mean:.4f} std={std:.4f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for history, target in train_loader:
            history = history.to(device)
            target = target[:, 0].to(device)

            optimizer.zero_grad()
            prediction = model(history)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * history.shape[0]
            total_samples += history.shape[0]

        val_mae, val_rmse, val_mape = evaluate(model, val_loader, device, mean, std)
        horizon = args.n_pred - 1
        if val_mae[horizon].item() < best_val_mae:
            best_val_mae = val_mae[horizon].item()
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:03d} train_mse={total_loss / total_samples:.6f} "
            f"val_horizon={args.n_pred} mae={val_mae[horizon].item():.4f} "
            f"rmse={val_rmse[horizon].item():.4f} mape={val_mape[horizon].item() * 100:.2f}%"
        )
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rmse, test_mape = evaluate(model, test_loader, device, mean, std)

    report_steps = list(range(3, args.n_pred + 1, 3))
    if args.n_pred not in report_steps:
        report_steps.append(args.n_pred)
    for step in report_steps:
        index = step - 1
        print(
            f"test_horizon={step} mae={test_mae[index].item():.4f} "
            f"rmse={test_rmse[index].item():.4f} mape={test_mape[index].item() * 100:.2f}%"
        )


if __name__ == "__main__":
    train(parse_args())
