# STGAT Paper Implementation

Clean implementation of Kong et al. 2020, "STGAT: Spatial-Temporal Graph Attention Networks for Traffic Flow Forecasting."

This project intentionally does not copy the rough original scripts. It implements the paper-level architecture as one integrated model:

- physical adjacency path
- learnable adaptive adjacency path
- stacked spatial-temporal blocks
- gated temporal convolutions with dilation pattern `1, 2, 1`
- multi-head graph attention
- learned gated fusion
- masked MAE, MAPE, and RMSE evaluation

## Paper Fidelity Notes

The core architecture matches the paper faithfully. However, there are a few intentional implementation choices that differ from the paper's description:

### Gated Temporal Convolution (GTCN) Differences

| Detail | Paper (Eq 1-2) | This Implementation |
|---|---|---|
| Filter branch activation | Linear: `X * V + c` | `tanh(Conv2d(X))` — the tanh nonlinearity follows the gated activation pattern used in WaveNet and Gated PixelCNN, and was found to improve training stability |
| Residual connection source | Same padded input used in gate/filter | Unpadded input via 1×1 conv — the residual bypasses the dilated temporal padding, preserving the original signal before padding |
| Residual structure | One block-level residual from input to last GTCN output | Two-level: per-GTCN residual (via gating equation) **plus** block-level residual around the attention layer — this double residual improves gradient flow through deep temporal stacks |

### Other Minor Differences

- **Output head**: A 512-neuron hidden layer with ReLU is used before the final linear projection (the paper does not specify the output layer structure)
- **Dropout placement**: Applied both at the graph attention coefficient level and at the ST-Block level (paper mentions GAT output dropout only)
- **Gradient clipping** (5.0), **gradient accumulation**, and **mixed-precision support** are practical additions not mentioned in the original paper

The paper also evaluates single-path variants (STGAT-Adj and STGAT-Adp) and cross-graph generalization — these ablation experiments are not implemented in this codebase.

## Data Layout

Place METR-LA or PEMS-BAY files under:

```text
GNN_Project/data/METR-LA/train.npz
GNN_Project/data/METR-LA/val.npz
GNN_Project/data/METR-LA/test.npz
GNN_Project/data/METR-LA/adj_mx_dijsk.pkl
```

Each `.npz` file must contain `x` and `y` arrays shaped `[samples, time, nodes, features]`.

## Config Reference

Configuration is in YAML format. All fields under `training` except the top-level ones are optional with sensible defaults.

### New Training Features (v0.2)

```yaml
training:
  # ... existing fields ...

  # Progress & Logging
  # (automatic — tqdm progress bars, Python logging to console + file,
  #  CSV metrics log written alongside checkpoint)

  # Reproducibility
  seed: 42                 # random seed (null to skip)

  # Periodic Checkpointing
  checkpoint_every: 20     # save periodic checkpoint every N epochs (0 = disabled)

  # Early Stopping
  early_stopping:
    patience: 25           # stop after N epochs without improvement (0 = disabled)
    min_delta: 0.0         # minimum change in val_mae to qualify as improvement

  # Learning Rate Scheduling
  scheduler:
    type: plateau          # "plateau" (ReduceLROnPlateau) or "cosine" (CosineAnnealingLR)
    factor: 0.5            # LR reduction factor (plateau only)
    patience: 10           # epochs before reducing LR (plateau only)
    min_lr: 1.0e-6         # minimum learning rate
```

## Commands

Use the existing Conda environment:

```bash
# Run tests
PYTHONPATH=GNN_Project/src conda run -n deeplearning python -m pytest GNN_Project/tests -v

# Generate data
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/generate_data.py --traffic-df-filename GNN_Project/metr-la.h5 --output-dir GNN_Project/data/METR-LA
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/generate_data.py --traffic-df-filename GNN_Project/pems-bay.h5 --output-dir GNN_Project/data/PEMS-BAY

# Train
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/train.py --config GNN_Project/configs/metr_la.yaml

# Train with custom seed and epochs
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/train.py --config GNN_Project/configs/metr_la.yaml --seed 123 --epochs 50

# Resume from checkpoint
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/train.py --config GNN_Project/configs/metr_la.yaml --resume GNN_Project/checkpoints/metr_la_best.pt

# Debug / smoke test (only 5 batches)
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/train.py --config GNN_Project/configs/metr_la.yaml --limit-batches 5 --epochs 2

# Evaluate
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/evaluate.py --config GNN_Project/configs/metr_la.yaml --checkpoint GNN_Project/checkpoints/metr_la_best.pt
```

## Training Output

Training uses tqdm progress bars at both epoch and batch level, plus structured logging:

```
11:23:45  INFO   Using device: cuda
11:23:45  INFO   Random seed set to 42 (deterministic cuDNN)
11:23:47  INFO   Starting training: 100 epochs, batch_size=16, accumulation_steps=2, lr=3.0e-04
11:23:47  INFO   Checkpoint dir: checkpoints  |  CSV log: checkpoints/metr_la.csv

Epochs:  45%|████████▍         | 45/100 [03:23<04:08, 4.52s/epoch, train_mae=28.34, val_mae=32.12, best=31.05, patience=3/25]
```

Each epoch produces a detailed log line:
```
11:26:12  INFO   epoch= 45  train_mae=28.3412  val_mae=32.1234  val_mape=12.45  val_rmse=8.23  lr=1.5e-04  time=4.5s  GPU mem: 2.1/4.5 GiB (alloc/reserved)
```

Metrics are also written to a CSV file (`checkpoints/<config_name>.csv`) for post-hoc analysis.

## Project Structure

```
GNN_Project/
  configs/           # YAML configuration files
  data/              # Dataset files (.npz, .pkl)
  scripts/           # Entry points (train.py, evaluate.py, generate_data.py)
  src/stgat/         # Library code
    model.py         # STGAT model architecture
    engine.py        # Training/evaluation loops, checkpointing
    data.py          # Data loading and scaling
    graph.py         # Adjacency matrix loading
    metrics.py       # Masked MAE/MAPE/RMSE
    config.py        # YAML config loader
    generate_data.py # HDF5 → NPZ preprocessing
  tests/             # Unit tests
```
