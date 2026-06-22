from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgat.config import load_config
from stgat.data import load_split_arrays, make_dataloaders
from stgat.engine import evaluate, load_checkpoint
from stgat.graph import load_adjacency
from stgat.model import STGAT


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STGAT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.training
    data_config = config.data
    model_config = config.model
    device = torch.device("cuda" if training.cuda and torch.cuda.is_available() else "cpu")

    arrays = load_split_arrays(project_path(data_config.data_dir))
    dataloaders = make_dataloaders(
        arrays,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    adjacency = load_adjacency(project_path(data_config.adjacency_path), data_config.adjacency_type)
    model = STGAT(
        num_nodes=model_config.num_nodes,
        input_features=model_config.input_features,
        input_steps=model_config.input_steps,
        output_steps=model_config.output_steps,
        hidden_channels=model_config.hidden_channels,
        attention_heads=tuple(model_config.attention_heads),
        blocks=model_config.blocks,
        dropout=model_config.dropout,
    ).to(device)
    checkpoint = load_checkpoint(project_path(args.checkpoint), model, device)
    metrics = evaluate(
        model,
        dataloaders["test"],
        adjacency,
        arrays["scaler"],
        device,
        null_value=training.null_value,
        amp_dtype=training.amp_dtype,
    )

    print(f"loaded epoch={checkpoint.get('epoch')} best_val_mae={checkpoint.get('best_val_mae')}")
    for horizon in (3, 6, 12):
        item = metrics["horizons"][horizon - 1]
        print(
            f"horizon={horizon} mae={item['mae']:.4f} "
            f"mape={item['mape']:.4f} rmse={item['rmse']:.4f}"
        )
    avg = metrics["average"]
    print(f"average mae={avg['mae']:.4f} mape={avg['mape']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == "__main__":
    main()
