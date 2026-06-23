"""
Live inference for a trained STGAT model.

Load a checkpoint and produce 12-step-ahead traffic speed forecasts
from the last 12 observations.

Usage (script):
    python scripts/inference.py \\
        --checkpoint checkpoints/metr_la_best.pt \\
        --data-dir data/METR-LA \\
        --adjacency data/METR-LA/adj_mx_dijsk.pkl

Usage (Kaggle / notebook):
    from scripts.inference import TrafficPredictor
    predictor = TrafficPredictor.from_checkpoint(
        checkpoint="checkpoints/metr_la_best.pt",
        data_dir="data/METR-LA",
        adjacency="data/METR-LA/adj_mx_dijsk.pkl",
    )
    forecast = predictor.predict(
        speeds=last_12_speeds,       # shape (12, 207)
        timestamps=last_12_timestamps,  # nanosecond unix timestamps
    )
    # forecast.shape → (12, 207)  — predicted speeds for next 12 steps
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgat.data import StandardScaler, _load_npz  # noqa: E402
from stgat.graph import load_adjacency  # noqa: E402
from stgat.model import STGAT  # noqa: E402


class TrafficPredictor:
    """Load a trained STGAT model and make live forecasts."""

    def __init__(
        self,
        model: STGAT,
        scaler: StandardScaler,
        adjacency: torch.Tensor,
        device: torch.device,
    ):
        self.model = model
        self.scaler = scaler
        self.adjacency = adjacency.to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        data_dir: str | Path,
        adjacency: str | Path | None = None,
        adjacency_type: str = "raw",
        device: str | torch.device | None = None,
    ) -> TrafficPredictor:
        """Build a predictor from a training checkpoint.

        Parameters
        ----------
        checkpoint : str or Path
            Path to the ``.pt`` checkpoint saved by ``scripts/train.py``.
        data_dir : str or Path
            Directory containing ``train.npz`` (used to compute the scaler
            statistics).  Only the training split is read — the scaler mean/std
            must match what the model was trained with.
        adjacency : str or Path, optional
            Path to the DCRNN-style adjacency pickle.  If ``None``, the path is
            read from the config embedded in the checkpoint.
        adjacency_type : str
            Normalisation: ``"raw"``, ``"sym"``, ``"row"``, or ``"identity"``.
        device : str or torch.device, optional
            Device to run inference on.  Defaults to CUDA if available.
        """
        checkpoint_path = Path(checkpoint)
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config_dict = checkpoint_data.get("config", {})

        # --- Device ---
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        # --- Model ---
        model_cfg = config_dict.get("model", {})
        model = STGAT(
            num_nodes=model_cfg.get("num_nodes"),
            input_features=model_cfg.get("input_features", 2),
            input_steps=model_cfg.get("input_steps", 12),
            output_steps=model_cfg.get("output_steps", 12),
            hidden_channels=model_cfg.get("hidden_channels", 64),
            attention_heads=tuple(model_cfg.get("attention_heads", (4, 4, 4, 6))),
            blocks=model_cfg.get("blocks", 4),
            dropout=model_cfg.get("dropout", 0.6),
        ).to(device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model.eval()

        # --- Scaler (computed from training split, matching training) ---
        data_path = Path(data_dir) / "train.npz"
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        train_x, _ = _load_npz(data_path)
        speed = train_x[..., 0]
        scaler = StandardScaler(
            mean=np.float32(speed.mean()),
            std=np.float32(speed.std()),
        )

        # --- Adjacency ---
        if adjacency is None:
            adj_path = config_dict.get("data", {}).get("adjacency_path")
            if adj_path is None:
                raise ValueError("No adjacency path provided and none found in checkpoint config")
            adj_path = Path(adj_path)
            if not adj_path.is_absolute():
                adj_path = Path(data_dir).parent / adj_path
            adj_type = config_dict.get("data", {}).get("adjacency_type", adjacency_type)
        else:
            adj_path = Path(adjacency)
            adj_type = adjacency_type

        adj_matrix = load_adjacency(adj_path, adj_type)

        return cls(model=model, scaler=scaler, adjacency=adj_matrix, device=device)

    @torch.no_grad()
    def predict(
        self,
        speeds: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forecast the next 12 speed values for every sensor.

        Parameters
        ----------
        speeds : np.ndarray, shape ``(input_steps, num_nodes)``
            Raw (un-normalized) speed readings for the last ``input_steps``
            time intervals.  ``speeds[t, n]`` is the speed at sensor *n* at
            relative time step *t* (t=0 is the oldest, t=-1 is the newest).
        timestamps : np.ndarray, shape ``(input_steps,)``, optional
            Nanosecond Unix timestamps for each of the ``input_steps`` rows.
            Used to compute the time-of-day feature.  If ``None``, the
            time-of-day feature is set to 0.5 (midday) for all steps, which
            will reduce accuracy.

        Returns
        -------
        forecast : np.ndarray, shape ``(output_steps, num_nodes)``
            Predicted speeds for the next 12 time steps, in original
            (un-normalized) speed units.
        """
        input_steps = self.model.input_steps
        num_nodes = self.model.num_nodes

        if speeds.shape != (input_steps, num_nodes):
            raise ValueError(f"speeds must have shape ({input_steps}, {num_nodes}), got {speeds.shape}")

        # --- Build input tensor [1, input_steps, num_nodes, 2] ---
        x_speed = np.asarray(speeds, dtype=np.float32).copy()
        x_speed = self.scaler.transform(x_speed)  # normalise speed channel

        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=np.float64)
            if timestamps.shape != (input_steps,):
                raise ValueError(f"timestamps must have shape ({input_steps},), got {timestamps.shape}")
            # nanosecond → second → time-of-day in [0, 1)
            seconds_of_day = (timestamps / 1e9).astype(np.float64) % 86400
            tod = (seconds_of_day / 86400).astype(np.float32)
        else:
            tod = np.full((input_steps,), 0.5, dtype=np.float32)

        # Stack [speed, time_of_day] along feature axis
        x = np.stack(
            [x_speed, np.broadcast_to(tod[:, np.newaxis], (input_steps, num_nodes))],
            axis=-1,
        )  # shape (12, N, 2)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 12, N, 2)

        # --- Run model ---
        pred = self.model(x_tensor, self.adjacency)  # (1, 12, N, 1)
        pred_np = pred.squeeze(0).squeeze(-1).cpu().numpy()  # (12, N)

        # Inverse-transform back to original speed units
        forecast = pred_np * float(self.scaler.std) + float(self.scaler.mean)
        return forecast.astype(np.float32)

    def predict_single(
        self,
        speeds: Sequence[float],
        timestamps: Sequence[float] | None = None,
        node_index: int = 0,
    ) -> np.ndarray:
        """Convenience wrapper: forecast for a single sensor from a flat list.

        Parameters
        ----------
        speeds : sequence of float
            Last ``input_steps`` speed values for the sensor.
        timestamps : sequence of float, optional
            Nanosecond timestamps corresponding to each speed value.
        node_index : int
            Which sensor index this data belongs to (0-based).

        Returns
        -------
        forecast : np.ndarray, shape ``(output_steps,)``
        """
        input_steps = self.model.input_steps
        num_nodes = self.model.num_nodes

        speeds_arr = np.zeros((input_steps, num_nodes), dtype=np.float32)
        speeds_arr[:, node_index] = np.asarray(speeds, dtype=np.float32)

        ts_arr = np.asarray(timestamps, dtype=np.float64) if timestamps is not None else None

        full_forecast = self.predict(speeds_arr, ts_arr)
        return full_forecast[:, node_index]


# ---------------------------------------------------------------------------
# CLI: quick smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run STGAT inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing train.npz (for scaler statistics)",
    )
    parser.add_argument(
        "--adjacency",
        default=None,
        help="Path to adjacency .pkl (default: read from checkpoint config)",
    )
    args = parser.parse_args()

    predictor = TrafficPredictor.from_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        adjacency=args.adjacency,
    )

    # Quick smoke test: forecast using the mean speed for every sensor
    mean_speed = float(predictor.scaler.mean)
    dummy_speeds = np.full(
        (predictor.model.input_steps, predictor.model.num_nodes),
        mean_speed,
        dtype=np.float32,
    )
    forecast = predictor.predict(dummy_speeds)
    print(
        f"Model: {predictor.model.num_nodes} nodes, "
        f"{predictor.model.input_steps}→{predictor.model.output_steps} steps"
    )
    print(f"Device: {predictor.device}")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}] mph")
    print("✓ Inference pipeline ready")


if __name__ == "__main__":
    main()
