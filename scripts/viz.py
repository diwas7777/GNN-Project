"""
Visualize STGAT training metrics from the per-epoch CSV log.

Usage:
    python scripts/viz.py checkpoints/metr_la.csv              # save PNG next to CSV
    python scripts/viz.py checkpoints/metr_la.csv --no-save    # show interactively
    python scripts/viz.py checkpoints/metr_la.csv --output report.png

In a Kaggle / Jupyter notebook:
    from scripts.viz import plot_training
    plot_training("checkpoints/metr_la.csv")
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Styling — clean, readable, publication-friendly
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "train": "#2c7bb6",  # blue
    "val": "#d7191c",  # red
    "best": "#fdae61",  # orange
    "lr": "#5e3c99",  # purple
    "grid": "#e0e0e0",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Read the training CSV and return column arrays.

    Returns
    -------
    dict with keys: epoch, train_mae, val_mae, val_mape, val_rmse,
                    epoch_time_s, lr, gpu_gb, is_best
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")

    epochs = np.array([int(r["epoch"]) for r in rows], dtype=np.int32)
    train_mae = np.array([float(r["train_mae"]) for r in rows], dtype=np.float32)
    val_mae = np.array([float(r["val_mae"]) for r in rows], dtype=np.float32)
    val_mape = np.array([float(r["val_mape"]) for r in rows], dtype=np.float32)
    val_rmse = np.array([float(r["val_rmse"]) for r in rows], dtype=np.float32)
    epoch_time_s = np.array([float(r["epoch_time_s"]) for r in rows], dtype=np.float32)
    lr = np.array([float(r["lr"]) for r in rows], dtype=np.float32)
    gpu_gb = np.array([float(r["gpu_memory_allocated_gb"]) for r in rows], dtype=np.float32)
    is_best = np.array([int(float(r["is_best"])) for r in rows], dtype=np.int32)

    return {
        "epoch": epochs,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "val_mape": val_mape,
        "val_rmse": val_rmse,
        "epoch_time_s": epoch_time_s,
        "lr": lr,
        "gpu_gb": gpu_gb,
        "is_best": is_best,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def _summarize(data: dict[str, np.ndarray]) -> dict[str, Any]:
    best_idx = int(np.argmin(data["val_mae"]))
    total_hours = float(np.sum(data["epoch_time_s"]) / 3600)

    # Overfitting: how much worse is the final model vs the best model?
    # Overfit ratio > 1.0 means validation error *increased* after the best epoch
    # while training error kept dropping — a classic overfitting signature.
    best_val = float(data["val_mae"][best_idx])
    final_val = float(data["val_mae"][-1])
    best_train = float(data["train_mae"][best_idx])
    final_train = float(data["train_mae"][-1])

    # Generalization gap at best epoch
    best_gap = best_val / best_train if best_train > 0 else float("inf")
    # Generalization gap at final epoch
    final_gap = final_val / final_train if final_train > 0 else float("inf")
    # Overfit degradation: how much val MAE worsened since the best epoch
    overfit_degradation = (final_val - best_val) / best_val if best_val > 0 else 0.0

    return {
        "total_epochs": len(data["epoch"]),
        "total_time_h": total_hours,
        "best_epoch": int(data["epoch"][best_idx]),
        "best_val_mae": best_val,
        "best_val_mape": float(data["val_mape"][best_idx]),
        "best_val_rmse": float(data["val_rmse"][best_idx]),
        "final_train_mae": final_train,
        "final_val_mae": final_val,
        "best_gap": best_gap,
        "final_gap": final_gap,
        "overfit_degradation": overfit_degradation,
    }


def _format_summary(s: dict[str, Any]) -> str:
    degradation_str = (
        f"  val degradation: {s['overfit_degradation']:+.1%} since best epoch"
        if s["overfit_degradation"] > 0.05
        else "  (no significant overfitting)"
    )
    return (
        f"Epochs: {s['total_epochs']}  |  Total time: {s['total_time_h']:.1f} h\n"
        f"Best epoch {s['best_epoch']}:  "
        f"val_mae={s['best_val_mae']:.4f}  "
        f"val_mape={s['best_val_mape']:.4f}  "
        f"val_rmse={s['best_val_rmse']:.4f}  "
        f"gap={s['best_gap']:.2f}x\n"
        f"Final epoch {s['total_epochs']}:  "
        f"train_mae={s['final_train_mae']:.4f}  "
        f"val_mae={s['final_val_mae']:.4f}  "
        f"gap={s['final_gap']:.2f}x\n"
        f"{degradation_str}"
    )


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_training(
    csv_path: str | Path,
    output: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot training metrics from a CSV log file.

    Creates a 2×2 panel figure:
      - Top-left:  train vs validation MAE (with best-epoch markers)
      - Top-right: validation MAPE + RMSE
      - Bottom-left: learning rate schedule
      - Bottom-right: epoch duration + GPU memory

    Parameters
    ----------
    csv_path : str or Path
        Path to the ``* .csv`` file written by ``scripts/train.py``.
    output : str or Path, optional
        If given, save the figure to this path (PNG).  If ``None`` (default)
        the PNG is saved next to the CSV with the same stem and ``_viz.png``
        suffix — unless ``show=True`` in an interactive environment.
    show : bool
        If True, call ``plt.show()``.  In a Jupyter / Kaggle notebook the
        figure will render inline regardless of this flag.
    """
    csv_path = Path(csv_path)
    data = load_csv(csv_path)
    summary = _summarize(data)

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_loss, ax_metric), (ax_lr, ax_time)) = axes  # type: ignore[misc]
    best_mask = data["is_best"] == 1

    # ── Top-left: train vs val MAE ─────────────────────────────────
    ax_loss.plot(data["epoch"], data["train_mae"], color=PALETTE["train"], label="Train MAE")
    ax_loss.plot(data["epoch"], data["val_mae"], color=PALETTE["val"], label="Val MAE")
    ax_loss.scatter(
        data["epoch"][best_mask],
        data["val_mae"][best_mask],
        color=PALETTE["best"],
        edgecolors="#b35806",
        s=50,
        zorder=5,
        label="Best model",
    )
    best_epoch = summary["best_epoch"]
    best_val = summary["best_val_mae"]
    ax_loss.annotate(
        f"best: epoch {best_epoch}\nMAE={best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 2, best_val + 0.08 * best_val),
        arrowprops={"arrowstyle": "->", "color": "#b35806", "lw": 1.2},
        fontsize=9,
        color="#b35806",
        fontweight="bold",
    )
    ax_loss.set_ylabel("MAE (speed)")
    ax_loss.set_title("Train vs Validation Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.grid(color=PALETTE["grid"], alpha=0.6)
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Top-right: val MAPE + RMSE ─────────────────────────────────
    ax_m = ax_metric.twinx()
    line_mape = ax_metric.plot(data["epoch"], data["val_mape"], color="#2ca25f", label="Val MAPE")
    line_rmse = ax_m.plot(data["epoch"], data["val_rmse"], color="#8856a7", linestyle="--", label="Val RMSE")
    ax_metric.set_ylabel("MAPE", color="#2ca25f")
    ax_m.set_ylabel("RMSE", color="#8856a7")
    ax_metric.tick_params(axis="y", colors="#2ca25f")
    ax_m.tick_params(axis="y", colors="#8856a7")
    # Merge legends
    lines = line_mape + line_rmse
    labels = [line.get_label() for line in lines]
    ax_metric.legend(lines, labels, loc="upper right")
    ax_metric.set_title("Validation MAPE & RMSE")
    ax_metric.grid(color=PALETTE["grid"], alpha=0.6)
    ax_metric.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Bottom-left: learning rate ──────────────────────────────────
    ax_lr.plot(data["epoch"], data["lr"], color=PALETTE["lr"], marker=".", markersize=4)
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_title("Learning Rate Schedule")
    ax_lr.set_yscale("log")
    ax_lr.grid(color=PALETTE["grid"], alpha=0.6)
    ax_lr.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Annotate LR drops
    unique_lrs = np.unique(data["lr"])
    if len(unique_lrs) > 1:
        for lr_val in unique_lrs:
            ax_lr.axhline(y=lr_val, color=PALETTE["grid"], linestyle=":", linewidth=0.8)
            ax_lr.text(
                data["epoch"][-1] + 0.5,
                lr_val,
                f"{lr_val:.1e}",
                fontsize=7,
                va="center",
                color="#666666",
            )

    # ── Bottom-right: epoch time + GPU memory ───────────────────────
    ax_t = ax_time.twinx()
    ax_time.bar(
        data["epoch"],
        data["epoch_time_s"] / 60,
        color="#abd9e9",
        edgecolor="#74add1",
        linewidth=0.3,
        label="Epoch time (min)",
    )
    if np.max(data["gpu_gb"]) > 0:
        ax_t.plot(
            data["epoch"], data["gpu_gb"], color="#d73027", marker=".", markersize=4, label="GPU mem (GiB)"
        )
        ax_t.set_ylabel("GPU memory (GiB)", color="#d73027")
        ax_t.tick_params(axis="y", colors="#d73027")
    ax_time.set_ylabel("Minutes")
    ax_time.set_xlabel("Epoch")
    ax_time.set_title("Epoch Duration & GPU Memory")
    ax_time.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_time.grid(axis="y", color=PALETTE["grid"], alpha=0.6)

    # ── Suptitle with summary ───────────────────────────────────────
    fig.suptitle(
        f"STGAT Training — {csv_path.stem}\n{_format_summary(summary)}",
        fontsize=10,
        fontfamily="monospace",
        y=1.01,
    )
    fig.tight_layout()

    # ── Save / show ─────────────────────────────────────────────────
    if output is None and not show:
        output = csv_path.with_suffix(".png")
    if output is not None:
        fig.savefig(output)
        print(f"Saved: {output}")

    # In a notebook the figure renders automatically; show() only when
    # running as a plain script.
    if show or not _is_notebook():
        plt.show()
    else:
        plt.close(fig)  # prevent double-render in notebooks

    return fig


def _is_notebook() -> bool:
    """Detect whether we are running inside a Jupyter / Kaggle notebook."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize STGAT training CSV metrics")
    parser.add_argument("csv", type=str, help="Path to the training CSV log")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output PNG path (default: <csv>_viz.png)"
    )
    parser.add_argument("--no-save", action="store_true", help="Show interactively, do not save")
    args = parser.parse_args()

    output = None if args.no_save else args.output
    plot_training(args.csv, output=output, show=args.no_save)


if __name__ == "__main__":
    main()
