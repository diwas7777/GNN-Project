import unittest

import numpy as np
import torch

from stgat.data import load_npz_splits, make_dataloaders
from stgat.engine import evaluate, should_use_amp, train_one_epoch
from stgat.model import STGAT


class EngineSmokeTest(unittest.TestCase):
    def test_amp_is_disabled_for_dataparallel_even_when_requested(self):
        model = torch.nn.DataParallel(torch.nn.Linear(2, 2))

        self.assertFalse(should_use_amp(model, torch.device("cuda"), requested=True))

    def test_amp_requires_cuda_and_request_without_dataparallel(self):
        model = torch.nn.Linear(2, 2)

        self.assertTrue(should_use_amp(model, torch.device("cuda"), requested=True))
        self.assertFalse(should_use_amp(model, torch.device("cuda"), requested=False))
        self.assertFalse(should_use_amp(model, torch.device("cpu"), requested=True))

    def test_train_and_evaluate_on_synthetic_data(self):
        rng = np.random.default_rng(42)
        train_x = rng.normal(size=(4, 12, 5, 2)).astype(np.float32)
        val_x = rng.normal(size=(2, 12, 5, 2)).astype(np.float32)
        test_x = rng.normal(size=(2, 12, 5, 2)).astype(np.float32)
        train_y = rng.normal(size=(4, 12, 5, 1)).astype(np.float32)
        val_y = rng.normal(size=(2, 12, 5, 1)).astype(np.float32)
        test_y = rng.normal(size=(2, 12, 5, 1)).astype(np.float32)
        scaler, train_x, val_x, test_x = load_npz_splits.scale_speed_feature(train_x, val_x, test_x)
        arrays = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
            "scaler": scaler,
        }
        loaders = make_dataloaders(arrays, batch_size=2)
        model = STGAT(
            num_nodes=5,
            input_features=2,
            input_steps=12,
            output_steps=12,
            hidden_channels=8,
            attention_heads=(2, 2),
            blocks=2,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_mae = train_one_epoch(
            model, loaders["train"], torch.eye(5), optimizer, scaler, torch.device("cpu")
        )
        metrics = evaluate(model, loaders["test"], torch.eye(5), scaler, torch.device("cpu"))

        self.assertGreaterEqual(train_mae, 0.0)
        self.assertEqual(len(metrics["horizons"]), 12)
        self.assertIn("mae", metrics["average"])


if __name__ == "__main__":
    unittest.main()
