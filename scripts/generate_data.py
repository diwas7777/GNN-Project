from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stgat.generate_data import generate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DCRNN-style train/val/test npz files from traffic h5 data")
    parser.add_argument("--traffic-df-filename", required=True, help="Path to metr-la.h5 or pems-bay.h5")
    parser.add_argument("--output-dir", required=True, help="Directory where train/val/test npz files will be written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shapes = generate_splits(args.traffic_df_filename, args.output_dir)
    for split, shape in shapes.items():
        print(f"{split}: x_shape={shape}")


if __name__ == "__main__":
    main()
