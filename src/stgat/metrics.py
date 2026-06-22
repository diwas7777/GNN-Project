from __future__ import annotations

import torch


def _masked_loss(loss: torch.Tensor, labels: torch.Tensor, null_value: float) -> torch.Tensor:
    if null_value != null_value:
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_value
    mask = mask.to(dtype=loss.dtype)
    mask_mean = mask.mean()
    if torch.isclose(mask_mean, torch.zeros_like(mask_mean)):
        return torch.zeros((), dtype=loss.dtype, device=loss.device)
    mask = mask / mask_mean
    mask = torch.nan_to_num(mask)
    return torch.nan_to_num(loss * mask).mean()


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_value: float = 0.0) -> torch.Tensor:
    return _masked_loss(torch.abs(preds - labels), labels, null_value)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_value: float = 0.0) -> torch.Tensor:
    return _masked_loss((preds - labels) ** 2, labels, null_value)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_value: float = 0.0) -> torch.Tensor:
    return torch.sqrt(masked_mse(preds, labels, null_value))


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_value: float = 0.0) -> torch.Tensor:
    percentage = torch.abs((preds - labels) / labels.clamp_min(1e-5))
    return _masked_loss(percentage, labels, null_value)


def metric_tuple(preds: torch.Tensor, labels: torch.Tensor, null_value: float = 0.0) -> tuple[float, float, float]:
    return (
        float(masked_mae(preds, labels, null_value).detach().cpu()),
        float(masked_mape(preds, labels, null_value).detach().cpu()),
        float(masked_rmse(preds, labels, null_value).detach().cpu()),
    )
