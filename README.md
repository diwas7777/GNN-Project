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

## Data Layout

Place METR-LA or PEMS-BAY files under:

```text
GNN_Project/data/METR-LA/train.npz
GNN_Project/data/METR-LA/val.npz
GNN_Project/data/METR-LA/test.npz
GNN_Project/data/METR-LA/adj_mx_dijsk.pkl
```

Each `.npz` file must contain `x` and `y` arrays shaped `[samples, time, nodes, features]`.

## Commands

Use the existing Conda environment:

```bash
PYTHONPATH=GNN_Project/src conda run -n deeplearning python -m unittest discover -s GNN_Project/tests -v
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/generate_data.py --traffic-df-filename GNN_Project/metr-la.h5 --output-dir GNN_Project/data/METR-LA
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/generate_data.py --traffic-df-filename GNN_Project/pems-bay.h5 --output-dir GNN_Project/data/PEMS-BAY
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/train.py --config GNN_Project/configs/metr_la.yaml
PYTHONPATH=GNN_Project/src conda run -n deeplearning python GNN_Project/scripts/evaluate.py --config GNN_Project/configs/metr_la.yaml --checkpoint GNN_Project/checkpoints/metr_la_best.pt
```
