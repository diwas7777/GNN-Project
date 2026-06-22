from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgat.config import load_config
from stgat.data import load_raw_split_arrays, make_dataloaders
from stgat.engine import evaluate, save_checkpoint, should_use_amp, train_one_epoch
from stgat.graph import load_adjacency
from stgat.model import STGAT, STGATWithAdjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STGAT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug limit for smoke tests")
    return parser.parse_args()


def maybe_limit(loader, limit):
    if limit is None:
        return loader
    return list(loader)[:limit]


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    data_config = config["data"]
    model_config = config["model"]

    device = torch.device("cuda" if training.get("cuda", False) and torch.cuda.is_available() else "cpu")
    arrays = load_raw_split_arrays(project_path(data_config["data_dir"]), include_test=False)
    dataloaders = make_dataloaders(arrays, batch_size=training["batch_size"], num_workers=training.get("num_workers", 0))
    train_loader = maybe_limit(dataloaders["train"], args.limit_batches)
    val_loader = maybe_limit(dataloaders["val"], args.limit_batches)

    adjacency = load_adjacency(project_path(data_config["adjacency_path"]), data_config.get("adjacency_type", "raw"))
    model = STGAT(
        num_nodes=model_config["num_nodes"],
        input_features=model_config["input_features"],
        input_steps=model_config["input_steps"],
        output_steps=model_config["output_steps"],
        hidden_channels=model_config["hidden_channels"],
        attention_heads=tuple(model_config["attention_heads"]),
        blocks=model_config["blocks"],
        dropout=model_config["dropout"],
    ).to(device)
    adjacency_for_engine = adjacency
    requested_amp = training.get("amp", True)
    if device.type == "cuda" and training.get("multi_gpu", False) and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(STGATWithAdjacency(model, adjacency.to(device)))
        adjacency_for_engine = None
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        use_amp = should_use_amp(model, device, requested_amp)
        if requested_amp and not use_amp:
            print("Disabling CUDA automatic mixed precision with DataParallel")
    else:
        print(f"Using device: {device}")
        use_amp = should_use_amp(model, device, requested_amp)
    if use_amp:
        print("Using CUDA automatic mixed precision")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )

    epochs = args.epochs if args.epochs is not None else training["epochs"]
    best_val_mae = float("inf")
    checkpoint_path = project_path(training["checkpoint_path"])
    for epoch in range(1, epochs + 1):
        train_mae = train_one_epoch(
            model,
            train_loader,
            adjacency_for_engine,
            optimizer,
            arrays["scaler"],
            device,
            null_value=training.get("null_value", 0.0),
            grad_clip=training.get("grad_clip", 5.0),
            use_amp=use_amp,
            accumulation_steps=training.get("accumulation_steps", 1),
        )
        val_metrics = evaluate(
            model,
            val_loader,
            adjacency_for_engine,
            arrays["scaler"],
            device,
            null_value=training.get("null_value", 0.0),
            use_amp=use_amp,
        )
        val_mae = val_metrics["average"]["mae"]
        print(f"epoch={epoch} train_mae={train_mae:.4f} val_mae={val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config, best_val_mae)
            print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
