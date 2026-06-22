from __future__ import annotations

import argparse
import atexit
import csv
import logging
import random
import signal
import sys
import time
import typing
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes messages through tqdm.write() so progress bars stay in place."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgat.config import load_config  # noqa: E402
from stgat.data import load_raw_split_arrays, make_dataloaders  # noqa: E402
from stgat.engine import (  # noqa: E402
    evaluate,
    load_checkpoint,
    save_checkpoint,
    should_use_amp,
    train_one_epoch,
)
from stgat.graph import load_adjacency  # noqa: E402
from stgat.model import STGAT, STGATWithAdjacency  # noqa: E402

logger = logging.getLogger("stgat.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STGAT")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug limit for smoke tests")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def maybe_limit(loader, limit):
    if limit is None:
        return loader
    return list(loader)[:limit]


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _setup_logging(log_dir: Path, name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    root_logger = logging.getLogger("stgat")
    root_logger.setLevel(logging.DEBUG)

    # Console handler (INFO+) — uses tqdm.write() to avoid breaking progress bars
    console = TqdmLoggingHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S"))
    root_logger.addHandler(console)

    # File handler (DEBUG+)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s  %(message)s"))
    root_logger.addHandler(file_handler)


def _setup_csv_logger(
    log_dir: Path, name: str, fieldnames: list[str]
) -> tuple[Path, csv.DictWriter, typing.TextIO]:
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{name}.csv"
    csv_file = open(csv_path, "w", newline="")  # noqa: SIM115 — kept open for per‑epoch writes
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return csv_path, writer, csv_file


def _log_csv(writer, file_handle, row: dict) -> None:
    writer.writerow(row)
    file_handle.flush()


def _format_duration(seconds: float) -> str:
    delta = timedelta(seconds=int(seconds))
    if delta >= timedelta(hours=1):
        return str(delta)
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:2d}m{secs:02d}s"


def _gpu_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    return f"  GPU mem: {allocated:.1f}/{reserved:.1f} GiB (alloc/reserved)"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (deterministic cuDNN)", seed)


# ---------------------------------------------------------------------------
# Graceful shutdown: save an emergency checkpoint on SIGINT / SIGTERM
# ---------------------------------------------------------------------------
_shutdown_state: dict = {
    "model": None,
    "optimizer": None,
    "epoch": 0,
    "config_dict": None,
    "best_val_mae": float("inf"),
    "checkpoint_path": None,
}


def _emergency_save() -> None:
    """Save an emergency checkpoint if a model and path are registered."""
    model = _shutdown_state["model"]
    path = _shutdown_state["checkpoint_path"]
    if model is None or path is None:
        return
    emergency_path = Path(path).with_suffix(".emergency.pt")
    try:
        from stgat.engine import save_checkpoint

        save_checkpoint(
            emergency_path,
            model,
            _shutdown_state["optimizer"],
            _shutdown_state["epoch"],
            _shutdown_state["config_dict"],
            _shutdown_state["best_val_mae"],
        )
        logger.info("Emergency checkpoint saved to %s", emergency_path)
    except Exception as exc:
        logger.error("Failed to save emergency checkpoint: %s", exc)


def _signal_handler(signum: int, frame: object) -> None:  # noqa: ARG001
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s — saving emergency checkpoint before exit", sig_name)
    _emergency_save()
    sys.exit(128 + signum)


def _register_shutdown_handlers() -> None:
    """Register SIGINT/SIGTERM handlers and atexit fallback."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_emergency_save)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.training
    data_config = config.data
    model_config = config.model

    # --- Logging setup ---
    checkpoint_dir = project_path(training.checkpoint_path).parent
    run_name = project_path(args.config).stem  # e.g., "metr_la" or "pems_bay"
    _setup_logging(checkpoint_dir, run_name)
    csv_path, csv_writer, csv_handle = _setup_csv_logger(
        checkpoint_dir,
        run_name,
        fieldnames=[
            "epoch",
            "train_mae",
            "val_mae",
            "val_mape",
            "val_rmse",
            "epoch_time_s",
            "lr",
            "gpu_memory_allocated_gb",
            "is_best",
        ],
    )

    # --- Seed ---
    seed = args.seed if args.seed is not None else training.seed
    if seed is not None:
        _set_seed(seed)
    else:
        logger.info("No random seed set (use --seed or training.seed for reproducibility)")

    # --- Device ---
    device = torch.device("cuda" if training.cuda and torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Data ---
    arrays = load_raw_split_arrays(project_path(data_config.data_dir), include_test=False)
    dataloaders = make_dataloaders(
        arrays,
        batch_size=training.batch_size,
        num_workers=training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = maybe_limit(dataloaders["train"], args.limit_batches)
    val_loader = maybe_limit(dataloaders["val"], args.limit_batches)

    # --- Graph ---
    adjacency = load_adjacency(project_path(data_config.adjacency_path), data_config.adjacency_type)

    # --- Model ---
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

    # --- torch.compile (PyTorch 2.0+) ---
    if training.compile_model:
        if hasattr(torch, "compile"):
            logger.info("Enabling torch.compile for model acceleration")
            model = torch.compile(model)  # type: ignore[attr-defined]
        else:
            logger.warning("torch.compile requested but not available (requires PyTorch 2.0+)")

    adjacency_for_engine = adjacency
    requested_amp = training.amp
    if device.type == "cuda" and training.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(STGATWithAdjacency(model, adjacency.to(device)))
        adjacency_for_engine = None
        logger.info("Using %d GPUs with DataParallel", torch.cuda.device_count())
        use_amp = should_use_amp(model, device, requested_amp)
        if requested_amp and not use_amp:
            logger.info("Disabling CUDA automatic mixed precision with DataParallel")
    else:
        use_amp = should_use_amp(model, device, requested_amp)
    if use_amp:
        dtype_label = "bfloat16" if training.amp_dtype == "bfloat16" else "float16"
        logger.info("Using CUDA automatic mixed precision (%s)", dtype_label)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    # --- LR Scheduler ---
    scheduler = None
    if training.scheduler is not None:
        sched_cfg = training.scheduler
        if sched_cfg.type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=sched_cfg.factor,
                patience=sched_cfg.patience,
                min_lr=sched_cfg.min_lr,
            )
            logger.info(
                "Using ReduceLROnPlateau scheduler (factor=%.2f, patience=%d)",
                sched_cfg.factor,
                sched_cfg.patience,
            )
        elif sched_cfg.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training.epochs,
                eta_min=sched_cfg.min_lr,
            )
            logger.info("Using CosineAnnealingLR scheduler (eta_min=%.1e)", sched_cfg.min_lr)

    # --- Checkpoint & Resume ---
    start_epoch = 1
    best_val_mae = float("inf")
    checkpoint_path = project_path(training.checkpoint_path)
    patience_counter = 0
    early_stop_patience = training.early_stopping.patience
    early_stop_min_delta = training.early_stopping.min_delta
    periodic_checkpoint_epochs = training.checkpoint_every

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = load_checkpoint(project_path(args.resume), model, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_mae = checkpoint.get("best_val_mae", float("inf"))
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                logger.warning("Could not restore optimizer state; starting fresh optimizer")

    # --- Training ---
    epochs = args.epochs if args.epochs is not None else training.epochs
    logger.info(
        "Starting training: %d epochs, batch_size=%d, accumulation_steps=%d, lr=%.1e",
        epochs,
        training.batch_size,
        training.accumulation_steps,
        training.learning_rate,
    )
    logger.info("Checkpoint dir: %s  |  CSV log: %s", checkpoint_dir, csv_path)

    # --- Graceful shutdown setup ---
    _shutdown_state["checkpoint_path"] = checkpoint_path
    _shutdown_state["config_dict"] = config.to_dict()
    _register_shutdown_handlers()

    total_start = time.perf_counter()

    epoch_pbar = tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_pbar:
        # Keep emergency-save state current so Ctrl+C saves the latest progress
        _shutdown_state["model"] = model
        _shutdown_state["optimizer"] = optimizer
        _shutdown_state["epoch"] = epoch
        _shutdown_state["best_val_mae"] = best_val_mae

        epoch_start = time.perf_counter()

        # --- Validation frequency: skip validation on non-validation epochs ---
        do_validation = epoch % training.val_every_n_epochs == 0

        train_mae = train_one_epoch(
            model,
            train_loader,
            adjacency_for_engine,
            optimizer,
            arrays["scaler"],
            device,
            null_value=training.null_value,
            grad_clip=training.grad_clip,
            use_amp=use_amp,
            amp_dtype=training.amp_dtype,
            accumulation_steps=training.accumulation_steps,
            epoch=epoch,
            total_epochs=epochs,
        )

        if do_validation:
            val_metrics = evaluate(
                model,
                val_loader,
                adjacency_for_engine,
                arrays["scaler"],
                device,
                null_value=training.null_value,
                use_amp=use_amp,
                amp_dtype=training.amp_dtype,
                desc=f"Validation epoch {epoch}",
            )
        else:
            val_metrics = {"average": {"mae": float("nan"), "mape": float("nan"), "rmse": float("nan")}}

        epoch_time = time.perf_counter() - epoch_start
        val_mae = val_metrics["average"]["mae"]
        val_mape = val_metrics["average"]["mape"]
        val_rmse = val_metrics["average"]["rmse"]
        current_lr = optimizer.param_groups[0]["lr"]
        is_best = val_mae < best_val_mae - early_stop_min_delta

        if is_best:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Logging ---
        gpu_mem_str = _gpu_memory_summary(device)
        epoch_time_str = _format_duration(epoch_time)
        logger.info(
            "epoch=%3d  train_mae=%.4f  val_mae=%.4f  val_mape=%.4f  val_rmse=%.4f  lr=%.1e  time=%s%s",
            epoch,
            train_mae,
            val_mae,
            val_mape,
            val_rmse,
            current_lr,
            epoch_time_str,
            gpu_mem_str,
        )

        # --- CSV ---
        csv_row = {
            "epoch": epoch,
            "train_mae": f"{train_mae:.6f}",
            "val_mae": f"{val_mae:.6f}",
            "val_mape": f"{val_mape:.6f}",
            "val_rmse": f"{val_rmse:.6f}",
            "epoch_time_s": f"{epoch_time:.1f}",
            "lr": f"{current_lr:.2e}",
            "gpu_memory_allocated_gb": f"{torch.cuda.memory_allocated(device) / (1024**3):.2f}"
            if device.type == "cuda"
            else "0",
            "is_best": "1" if is_best else "0",
        }
        _log_csv(csv_writer, csv_handle, csv_row)

        # --- Checkpointing ---
        if is_best:
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config.to_dict(), best_val_mae)
            logger.info("  ✓ Best model saved (val_mae=%.4f) → %s", val_mae, checkpoint_path)

        if periodic_checkpoint_epochs > 0 and epoch % periodic_checkpoint_epochs == 0 and not is_best:
            periodic_path = checkpoint_path.with_suffix(f".epoch{epoch}.pt")
            save_checkpoint(periodic_path, model, optimizer, epoch, config.to_dict(), best_val_mae)

        # --- Scheduler step ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mae)
            else:
                scheduler.step()

        # --- Early stopping ---
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement (patience=%d, min_delta=%.1e)",
                patience_counter,
                early_stop_patience,
                early_stop_min_delta,
            )
            epoch_pbar.close()
            break

        # --- Update progress bar ---
        epoch_pbar.set_postfix(
            train_mae=f"{train_mae:.4f}",
            val_mae=f"{val_mae:.4f}" if do_validation else "skip",
            best=f"{best_val_mae:.4f}",
            patience=f"{patience_counter}/{early_stop_patience}" if early_stop_patience > 0 else "-",
        )

    # --- Final summary ---
    total_time = time.perf_counter() - total_start
    logger.info(
        "Training complete. Total time: %s  |  Best val_mae: %.4f  |  Checkpoint: %s",
        _format_duration(total_time),
        best_val_mae,
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
