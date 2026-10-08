from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

VISIFOLD_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = VISIFOLD_ROOT.parents[1]
sys.path.insert(0, str(VISIFOLD_ROOT))

from model.visifold import VisiFold  # noqa: E402
from script.data_loader import (  # noqa: E402
    StandardScaler,
    build_datasets,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VisiFold on a PEMS dataset")
    parser.add_argument(
        "--dataset",
        type=str.upper,
        choices=("PEMS04", "PEMS08", "PEMS-BAY"),
        default="PEMS-BAY",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=VISIFOLD_ROOT / "model" / "original_config.yaml",
    )
    parser.add_argument("--in-steps", type=int, default=24)
    parser.add_argument("--out-steps", type=int)
    parser.add_argument("--steps-per-day", type=int, default=288)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-milestones", type=int, nargs="+")
    parser.add_argument("--lr-decay-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--feed-forward-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--subgraph-size", type=int)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=VISIFOLD_ROOT / "output")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> str:
    """Fill unspecified arguments from the official config or PEMS-BAY defaults."""

    args.out_steps = args.in_steps if args.out_steps is None else args.out_steps
    preset_name = f"{args.dataset}-{args.in_steps}"
    fallback = {
        "lr": 1e-4,
        "milestones": [55],
        "lr_decay_rate": 0.1,
        "batch_size": 16,
        "max_epochs": 300,
        "early_stop": 10,
        "embedding_dim": 64,
        "feed_forward_dim": 1024,
        "num_heads": 4,
        "num_layers": 1,
        "mask_ratio": 0.2,
        "subgraph_size": 50,
    }

    presets = {}
    if args.config.is_file():
        with args.config.open("r", encoding="utf-8") as file:
            presets = yaml.safe_load(file) or {}
    if args.dataset in {"PEMS04", "PEMS08"} and preset_name not in presets:
        raise ValueError(
            f"No original preset named {preset_name} in {args.config}; "
            "use 24, 36, or 48 input steps"
        )
    preset = presets.get(preset_name, fallback)

    fields = {
        "lr": "lr",
        "lr_milestones": "milestones",
        "lr_decay_rate": "lr_decay_rate",
        "batch_size": "batch_size",
        "epochs": "max_epochs",
        "patience": "early_stop",
        "embedding_dim": "embedding_dim",
        "feed_forward_dim": "feed_forward_dim",
        "num_heads": "num_heads",
        "num_layers": "num_layers",
        "mask_ratio": "mask_ratio",
        "subgraph_size": "subgraph_size",
    }
    for argument, config_key in fields.items():
        if getattr(args, argument) is None:
            setattr(args, argument, preset[config_key])
    return preset_name if preset_name in presets else "PEMS-BAY-transferred"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def select_visible_targets(target: Tensor, visible_nodes: Tensor | None) -> Tensor:
    return target if visible_nodes is None else target.index_select(2, visible_nodes)


@torch.no_grad()
def evaluate(
    model: VisiFold,
    loader: DataLoader,
    device: torch.device,
    scaler: StandardScaler,
    criterion: nn.Module,
) -> tuple[float, float, float, float]:
    model.eval()
    loss_sum = 0.0
    element_count = 0
    absolute_error = torch.zeros((), device=device, dtype=torch.float64)
    squared_error = torch.zeros((), device=device, dtype=torch.float64)
    percentage_error = torch.zeros((), device=device, dtype=torch.float64)
    valid_count = torch.zeros((), device=device, dtype=torch.float64)

    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        normalized_prediction, _ = model(history)
        prediction = scaler.inverse_transform(normalized_prediction)

        loss_sum += criterion(prediction, target).item()
        element_count += target.numel()

        valid = torch.isfinite(target) & (target != 0.0)
        difference = (prediction - target).double()
        valid_double = valid.double()
        absolute_error += (difference.abs() * valid_double).sum()
        squared_error += (difference.square() * valid_double).sum()
        percentage_error += (
            difference.abs()
            / torch.where(valid, target.abs(), torch.ones_like(target)).double()
            * valid_double
        ).sum()
        valid_count += valid_double.sum()

    if element_count == 0 or valid_count.item() == 0:
        raise RuntimeError("Evaluation loader produced no valid targets")
    huber = loss_sum / element_count
    mae = (absolute_error / valid_count).item()
    rmse = torch.sqrt(squared_error / valid_count).item()
    mape = (percentage_error / valid_count).item() * 100.0
    return huber, rmse, mae, mape


def train_one_epoch(
    model: VisiFold,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: StandardScaler,
    criterion: nn.Module,
    clip_grad: float,
) -> float:
    model.train()
    loss_sum = 0.0
    element_count = 0

    for history, target in loader:
        history = history.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        normalized_prediction, visible_nodes = model(history)
        prediction = scaler.inverse_transform(normalized_prediction)
        target = select_visible_targets(target, visible_nodes)
        loss = criterion(prediction, target) / target.numel()
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        loss_sum += loss.item() * target.numel()
        element_count += target.numel()

    if element_count == 0:
        raise RuntimeError("Training loader produced no batches; reduce --batch-size")
    return loss_sum / element_count


def train(args: argparse.Namespace) -> None:
    preset_name = apply_preset(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    values, time_of_day, day_of_week, sensor_ids = load_dataset(
        args.data_root, args.dataset, args.steps_per_day
    )
    train_data, validation_data, test_data, scaler = build_datasets(
        values,
        time_of_day,
        day_of_week,
        args.in_steps,
        args.out_steps,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_loader(
        train_data, args.batch_size, True, args.num_workers, device, generator
    )
    validation_loader = build_loader(
        validation_data, args.batch_size, False, args.num_workers, device, generator
    )
    test_loader = build_loader(
        test_data, args.batch_size, False, args.num_workers, device, generator
    )

    model = VisiFold(
        num_nodes=len(sensor_ids),
        in_steps=args.in_steps,
        out_steps=args.out_steps,
        steps_per_day=args.steps_per_day,
        embedding_dim=args.embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mask_ratio=args.mask_ratio,
        subgraph_size=args.subgraph_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(args.lr_milestones),
        gamma=args.lr_decay_rate,
    )
    criterion = nn.HuberLoss(delta=1.0, reduction="sum")

    print(
        f"dataset={args.dataset} preset={preset_name} timestamps={len(values)} "
        f"nodes={len(sensor_ids)} split={len(train_data)}/{len(validation_data)}/"
        f"{len(test_data)} device={device}"
    )
    print(
        f"in_steps={args.in_steps} out_steps={args.out_steps} "
        f"mean={scaler.mean:.4f} std={scaler.std:.4f}"
    )

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            criterion,
            args.clip_grad,
        )
        validation_loss, validation_rmse, validation_mae, validation_mape = evaluate(
            model, validation_loader, device, scaler, criterion
        )
        print(
            f"epoch={epoch:03d} train_huber={train_loss:.5f} "
            f"val_huber={validation_loss:.5f} val_rmse={validation_rmse:.4f} "
            f"val_mae={validation_mae:.4f} val_mape={validation_mape:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        scheduler.step()
        if wait >= args.patience:
            print(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_loss, test_rmse, test_mae, test_mape = evaluate(
        model, test_loader, device, scaler, criterion
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / (
        f"visifold_{args.dataset.lower()}_{args.in_steps}_{args.out_steps}_seed{args.seed}.pt"
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "dataset": args.dataset,
            "preset": preset_name,
            "model_args": {
                "num_nodes": len(sensor_ids),
                "in_steps": args.in_steps,
                "out_steps": args.out_steps,
                "steps_per_day": args.steps_per_day,
                "embedding_dim": args.embedding_dim,
                "feed_forward_dim": args.feed_forward_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "mask_ratio": args.mask_ratio,
                "subgraph_size": args.subgraph_size,
                "dropout": args.dropout,
            },
            "scaler": {"mean": scaler.mean, "std": scaler.std},
            "sensor_ids": sensor_ids,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    elapsed = time.perf_counter() - started_at
    print(
        f"test_huber={test_loss:.5f} test_rmse={test_rmse:.4f} "
        f"test_mae={test_mae:.4f} test_mape={test_mape:.2f}%"
    )
    print(f"checkpoint={checkpoint_path} elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    train(parse_args())
