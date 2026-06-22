from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import StandardScaler
from .metrics import masked_mae, metric_tuple
from .model import STGATWithAdjacency


def inverse_speed(tensor: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    return tensor * float(scaler.std) + float(scaler.mean)


def _forward(model: torch.nn.Module, x: torch.Tensor, adjacency: torch.Tensor | None) -> torch.Tensor:
    if adjacency is None:
        return model(x)
    return model(x, adjacency)


def _checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if isinstance(model, STGATWithAdjacency):
        return model.model
    return model


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    adjacency: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    device: torch.device,
    null_value: float = 0.0,
    grad_clip: float | None = 5.0,
    use_amp: bool = False,
    accumulation_steps: int = 1,
) -> float:
    model.train()
    adjacency = adjacency.to(device) if adjacency is not None else None
    losses: list[float] = []
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(dataloader, start=1):
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred = _forward(model, x, adjacency)
            pred_speed = inverse_speed(pred[..., 0], scaler)
            y_speed = inverse_speed(y[..., 0], scaler)
            loss = masked_mae(pred_speed, y_speed, null_value)
            scaled_loss = loss / accumulation_steps
        scaler_amp.scale(scaled_loss).backward()
        if step % accumulation_steps == 0:
            if grad_clip is not None:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if len(losses) % accumulation_steps != 0:
        if grad_clip is not None:
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler_amp.step(optimizer)
        scaler_amp.update()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    adjacency: torch.Tensor | None,
    scaler: StandardScaler,
    device: torch.device,
    null_value: float = 0.0,
    use_amp: bool = False,
) -> dict[str, Any]:
    model.eval()
    adjacency = adjacency.to(device) if adjacency is not None else None
    predictions = []
    labels = []
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            pred_batch = _forward(model, x, adjacency)
        predictions.append(inverse_speed(pred_batch[..., 0], scaler).detach().cpu())
        labels.append(inverse_speed(y[..., 0], scaler).detach().cpu())
    pred = torch.cat(predictions, dim=0)
    true = torch.cat(labels, dim=0)
    horizons = []
    for index in range(pred.shape[1]):
        mae, mape, rmse = metric_tuple(pred[:, index], true[:, index], null_value)
        horizons.append({"horizon": index + 1, "mae": mae, "mape": mape, "rmse": rmse})
    mae, mape, rmse = metric_tuple(pred, true, null_value)
    return {"average": {"mae": mae, "mape": mape, "rmse": rmse}, "horizons": horizons}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    best_val_mae: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _checkpoint_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "best_val_mae": best_val_mae,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
