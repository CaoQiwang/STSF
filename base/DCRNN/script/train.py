from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

DCRNN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DCRNN_ROOT.parents[1]
sys.path.insert(0, str(DCRNN_ROOT))

from model.dcrnn import DCRNN
from script.data_loader import build_datasets, dual_random_walk_supports, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classic DCRNN model")
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS08", "PEMS-BAY"), default="PEMS-BAY")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--max-diffusion-step", type=int, default=2)
    parser.add_argument("--cl-decay-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=(20, 30, 40, 50))
    parser.add_argument("--lr-decay-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mae(prediction: Tensor, target: Tensor, mean: float, std: float) -> Tensor:
    prediction = prediction * std + mean
    target = target * std + mean
    mask = (target != 0.0).float()
    mask = mask / mask.mean()
    loss = torch.abs(prediction - target) * mask
    return torch.nan_to_num(loss).mean()


@torch.no_grad()
def evaluate(
    model: DCRNN,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    absolute_error = torch.zeros(model.horizon, device=device)
    squared_error = torch.zeros(model.horizon, device=device)
    percentage_error = torch.zeros(model.horizon, device=device)
    valid_count = torch.zeros(model.horizon, device=device)

    for history, target in loader:
        history = history.to(device)
        target = target.to(device)
        prediction = model(history)
        prediction = prediction * std + mean
        target = target * std + mean
        difference = prediction - target
        mask = target != 0.0

        absolute_error += (difference.abs() * mask).sum(dim=(0, 2, 3))
        squared_error += (difference.square() * mask).sum(dim=(0, 2, 3))
        percentage_error += (
            difference.abs() / torch.where(mask, target.abs(), torch.ones_like(target)) * mask
        ).sum(dim=(0, 2, 3))
        valid_count += mask.sum(dim=(0, 2, 3))

    mae = absolute_error / valid_count
    rmse = torch.sqrt(squared_error / valid_count)
    mape = percentage_error / valid_count
    return mae, rmse, mape


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    values, adjacency = load_dataset(args.data_root, args.dataset, args.feature)
    train_data, val_data, test_data, mean, std = build_datasets(
        values, args.seq_len, args.horizon
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = DCRNN(
        supports=dual_random_walk_supports(adjacency),
        input_dim=2,
        output_dim=1,
        rnn_units=args.rnn_units,
        num_rnn_layers=args.rnn_layers,
        max_diffusion_step=args.max_diffusion_step,
        horizon=args.horizon,
        cl_decay_steps=args.cl_decay_steps,
        use_curriculum_learning=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(args.lr_milestones), gamma=args.lr_decay_ratio
    )

    best_val_mae = float("inf")
    best_state = None
    wait = 0
    batches_seen = 0
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
            target = target.to(device)

            optimizer.zero_grad()
            prediction = model(history, target, batches_seen)
            loss = masked_mae(prediction, target, mean, std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * history.shape[0]
            total_samples += history.shape[0]
            batches_seen += 1

        val_mae, val_rmse, val_mape = evaluate(model, val_loader, device, mean, std)
        final_horizon = args.horizon - 1
        current_val_mae = val_mae[final_horizon].item()
        if current_val_mae <= best_val_mae:
            best_val_mae = current_val_mae
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        print(
            f"epoch={epoch:03d} train_mae={total_loss / total_samples:.4f} "
            f"val_horizon={args.horizon} mae={current_val_mae:.4f} "
            f"rmse={val_rmse[final_horizon].item():.4f} "
            f"mape={val_mape[final_horizon].item() * 100:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()
        if wait > args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rmse, test_mape = evaluate(model, test_loader, device, mean, std)
    for step in range(1, args.horizon + 1):
        index = step - 1
        print(
            f"test_horizon={step:02d} mae={test_mae[index].item():.4f} "
            f"rmse={test_rmse[index].item():.4f} mape={test_mape[index].item() * 100:.2f}%"
        )


if __name__ == "__main__":
    train(parse_args())
