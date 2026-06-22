import pickle
import unittest

import numpy as np
import torch

from stgat.data import StandardScaler, TrafficDataset, load_npz_splits
from stgat.graph import load_adjacency
from stgat.metrics import masked_mae, masked_mape, masked_rmse


class DataMetricsGraphTest(unittest.TestCase):
    def test_standard_scaler_uses_train_statistics_only(self):
        train = np.array([[[[1.0], [3.0]]], [[[5.0], [7.0]]]], dtype=np.float32)
        val = np.array([[[[100.0], [200.0]]]], dtype=np.float32)
        test = np.array([[[[300.0], [400.0]]]], dtype=np.float32)

        scaler, train_scaled, val_scaled, test_scaled = load_npz_splits.scale_speed_feature(train, val, test)

        self.assertEqual(scaler.mean, np.float32(4.0))
        self.assertEqual(scaler.std, np.float32(np.std(train[..., 0])))
        np.testing.assert_allclose(train_scaled[..., 0], (train[..., 0] - 4.0) / scaler.std)
        np.testing.assert_allclose(val_scaled[..., 0], (val[..., 0] - 4.0) / scaler.std)
        np.testing.assert_allclose(test_scaled[..., 0], (test[..., 0] - 4.0) / scaler.std)

    def test_traffic_dataset_returns_batch_ready_tensors(self):
        x = np.zeros((2, 12, 4, 2), dtype=np.float32)
        y = np.ones((2, 12, 4, 1), dtype=np.float32)

        dataset = TrafficDataset(x, y)

        sample_x, sample_y = dataset[0]
        self.assertEqual(sample_x.shape, (12, 4, 2))
        self.assertEqual(sample_y.shape, (12, 4, 1))
        self.assertEqual(sample_x.dtype, torch.float32)
        self.assertEqual(sample_y.dtype, torch.float32)

    def test_traffic_dataset_normalizes_speed_on_access(self):
        x = np.array([[[[3.0, 0.25], [5.0, 0.5]]]], dtype=np.float32)
        y = np.array([[[[7.0, 0.75], [9.0, 1.0]]]], dtype=np.float32)
        scaler = StandardScaler(mean=np.float32(5.0), std=np.float32(2.0))

        dataset = TrafficDataset(x, y, scaler=scaler)
        sample_x, sample_y = dataset[0]

        np.testing.assert_allclose(sample_x[..., 0].numpy(), np.array([[-1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(sample_y[..., 0].numpy(), np.array([[1.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(sample_x[..., 1].numpy(), np.array([[0.25, 0.5]], dtype=np.float32))

    def test_masked_metrics_ignore_zero_labels(self):
        pred = torch.tensor([[2.0, 5.0, 10.0]])
        true = torch.tensor([[1.0, 0.0, 8.0]])

        self.assertTrue(torch.isclose(masked_mae(pred, true, null_value=0.0), torch.tensor(1.5)))
        self.assertTrue(torch.isclose(masked_rmse(pred, true, null_value=0.0), torch.sqrt(torch.tensor(2.5))))
        self.assertTrue(torch.isclose(masked_mape(pred, true, null_value=0.0), torch.tensor(0.625)))

    def test_load_adjacency_from_dcrnn_pickle(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adj.pkl"
            matrix = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float32)
            with path.open("wb") as f:
                pickle.dump((["a", "b"], {"a": 0, "b": 1}, matrix), f)

            adjacency = load_adjacency(path, adjacency_type="raw")

        self.assertEqual(adjacency.shape, (2, 2))
        self.assertEqual(adjacency.dtype, torch.float32)
        self.assertTrue(torch.equal(adjacency, torch.from_numpy(matrix)))

    def test_load_adjacency_accepts_list_payload(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adj.pkl"
            matrix = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            with path.open("wb") as f:
                pickle.dump([["a", "b"], {"a": 0, "b": 1}, matrix], f)

            adjacency = load_adjacency(path, adjacency_type="raw")

        self.assertTrue(torch.equal(adjacency, torch.from_numpy(matrix)))


if __name__ == "__main__":
    unittest.main()
