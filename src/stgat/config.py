from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

AdjacencyType = Literal["raw", "identity", "sym", "row"]
SchedulerType = Literal["plateau", "cosine"]


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_dir: str
    adjacency_path: str
    adjacency_type: AdjacencyType = "raw"

    def __post_init__(self) -> None:
        if not self.data_dir:
            raise ValueError("data_dir must not be empty")
        if not self.adjacency_path:
            raise ValueError("adjacency_path must not be empty")
        valid_types = {"raw", "identity", "sym", "row"}
        if self.adjacency_type not in valid_types:
            raise ValueError(
                f"adjacency_type must be one of {sorted(valid_types)}, got {self.adjacency_type!r}"
            )


@dataclass
class ModelConfig:
    """Configuration for the STGAT model architecture."""

    num_nodes: int
    input_features: int = 2
    input_steps: int = 12
    output_steps: int = 12
    hidden_channels: int = 64
    attention_heads: list[int] = field(default_factory=lambda: [4, 4, 4, 6])
    blocks: int = 4
    dropout: float = 0.6

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError(f"num_nodes must be positive, got {self.num_nodes}")
        if self.input_features <= 0:
            raise ValueError(f"input_features must be positive, got {self.input_features}")
        if self.input_steps <= 0:
            raise ValueError(f"input_steps must be positive, got {self.input_steps}")
        if self.output_steps <= 0:
            raise ValueError(f"output_steps must be positive, got {self.output_steps}")
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {self.hidden_channels}")
        if self.blocks <= 0:
            raise ValueError(f"blocks must be positive, got {self.blocks}")
        if len(self.attention_heads) != self.blocks:
            raise ValueError(
                f"len(attention_heads) ({len(self.attention_heads)}) must equal blocks ({self.blocks})"
            )
        if not all(h > 0 for h in self.attention_heads):
            raise ValueError(f"all attention_heads must be positive, got {self.attention_heads}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 25
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""

    type: SchedulerType = "plateau"
    factor: float = 0.5
    patience: int = 10
    min_lr: float = 1e-6

    def __post_init__(self) -> None:
        valid_types = {"plateau", "cosine"}
        if self.type not in valid_types:
            raise ValueError(f"scheduler type must be one of {sorted(valid_types)}, got {self.type!r}")
        if not 0.0 < self.factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {self.factor}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.min_lr <= 0:
            raise ValueError(f"min_lr must be positive, got {self.min_lr}")


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    batch_size: int
    epochs: int
    learning_rate: float
    checkpoint_path: str
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0
    null_value: float = 0.0
    cuda: bool = True
    multi_gpu: bool = False
    amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    accumulation_steps: int = 1
    num_workers: int = 0
    checkpoint_every: int = 0
    seed: int | None = None
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    scheduler: SchedulerConfig | None = None
    val_every_n_epochs: int = 1
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path must not be empty")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive or None, got {self.grad_clip}")
        if self.accumulation_steps <= 0:
            raise ValueError(f"accumulation_steps must be positive, got {self.accumulation_steps}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must be >= 0, got {self.checkpoint_every}")
        if self.val_every_n_epochs <= 0:
            raise ValueError(f"val_every_n_epochs must be positive, got {self.val_every_n_epochs}")
        valid_amp = {"float16", "bfloat16"}
        if self.amp_dtype not in valid_amp:
            raise ValueError(f"amp_dtype must be one of {sorted(valid_amp)}, got {self.amp_dtype!r}")


@dataclass
class STGATConfig:
    """Top-level configuration combining data, model, and training sections."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> STGATConfig:
        """Construct a validated STGATConfig from a raw dictionary (e.g., from YAML).

        Raises ValueError with a descriptive message if any section fails validation.
        """
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a mapping, got {type(raw).__name__}")

        expected_sections = {"data", "model", "training"}
        missing = expected_sections - set(raw.keys())
        if missing:
            raise ValueError(f"config missing required section(s): {sorted(missing)}")

        unknown = set(raw.keys()) - expected_sections
        if unknown:
            logger.warning("Unknown config section(s) ignored: %s", sorted(unknown))

        # Parse subsections, providing helpful context on failure
        try:
            data = DataConfig(**raw["data"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid [data] section: {exc}") from exc

        try:
            model = ModelConfig(**raw["model"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid [model] section: {exc}") from exc

        training_raw = dict(raw["training"])

        # Nested objects
        if "early_stopping" in training_raw and isinstance(training_raw["early_stopping"], dict):
            try:
                training_raw["early_stopping"] = EarlyStoppingConfig(**training_raw["early_stopping"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid [training.early_stopping] section: {exc}") from exc

        if "scheduler" in training_raw and isinstance(training_raw["scheduler"], dict):
            try:
                training_raw["scheduler"] = SchedulerConfig(**training_raw["scheduler"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid [training.scheduler] section: {exc}") from exc

        try:
            training = TrainingConfig(**training_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid [training] section: {exc}") from exc

        return cls(data=data, model=model, training=training)

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (for storing in checkpoints)."""
        return {
            "data": {
                "data_dir": self.data.data_dir,
                "adjacency_path": self.data.adjacency_path,
                "adjacency_type": self.data.adjacency_type,
            },
            "model": {
                "num_nodes": self.model.num_nodes,
                "input_features": self.model.input_features,
                "input_steps": self.model.input_steps,
                "output_steps": self.model.output_steps,
                "hidden_channels": self.model.hidden_channels,
                "attention_heads": list(self.model.attention_heads),
                "blocks": self.model.blocks,
                "dropout": self.model.dropout,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "grad_clip": self.training.grad_clip,
                "null_value": self.training.null_value,
                "cuda": self.training.cuda,
                "multi_gpu": self.training.multi_gpu,
                "amp": self.training.amp,
                "amp_dtype": self.training.amp_dtype,
                "accumulation_steps": self.training.accumulation_steps,
                "num_workers": self.training.num_workers,
                "checkpoint_path": self.training.checkpoint_path,
                "checkpoint_every": self.training.checkpoint_every,
                "seed": self.training.seed,
                "val_every_n_epochs": self.training.val_every_n_epochs,
                "compile_model": self.training.compile_model,
                "early_stopping": {
                    "patience": self.training.early_stopping.patience,
                    "min_delta": self.training.early_stopping.min_delta,
                },
                "scheduler": {
                    "type": self.training.scheduler.type,
                    "factor": self.training.scheduler.factor,
                    "patience": self.training.scheduler.patience,
                    "min_lr": self.training.scheduler.min_lr,
                }
                if self.training.scheduler is not None
                else None,
            },
        }


def load_config(path: str | Path) -> STGATConfig:
    """Load and validate a YAML configuration file.

    Returns a fully validated ``STGATConfig`` with typed, documented fields.
    Raises ``ValueError`` if any section is missing, misspelled, or contains
    invalid values.
    """
    raw_path = Path(path)
    with raw_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a YAML mapping: {raw_path}")
    return STGATConfig.from_dict(raw)
