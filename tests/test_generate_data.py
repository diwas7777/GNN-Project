import unittest

import numpy as np

from stgat.generate_data import generate_graph_seq2seq_io, split_train_val_test


class GenerateDataTest(unittest.TestCase):
    def test_generate_graph_seq2seq_io_uses_dcrnn_offsets(self):
        values = np.arange(30 * 2, dtype=np.float32).reshape(30, 2)
        time_of_day = (np.arange(30, dtype=np.float32) % 288) / 288.0

        x, y, x_offsets, y_offsets = generate_graph_seq2seq_io(values, time_of_day)

        self.assertEqual(x.shape, (7, 12, 2, 2))
        self.assertEqual(y.shape, (7, 12, 2, 2))
        np.testing.assert_array_equal(x_offsets.reshape(-1), np.arange(-11, 1))
        np.testing.assert_array_equal(y_offsets.reshape(-1), np.arange(1, 13))
        np.testing.assert_array_equal(x[0, :, :, 0], values[0:12])
        np.testing.assert_array_equal(y[0, :, :, 0], values[12:24])
        np.testing.assert_array_equal(x[0, :, 0, 1], time_of_day[0:12])
        np.testing.assert_array_equal(y[0, :, 0, 1], time_of_day[12:24])

    def test_split_train_val_test_uses_dcrnn_ratios(self):
        x = np.zeros((100, 12, 2, 2), dtype=np.float32)
        y = np.ones((100, 12, 2, 2), dtype=np.float32)

        splits = split_train_val_test(x, y)

        self.assertEqual(splits["train"][0].shape[0], 70)
        self.assertEqual(splits["val"][0].shape[0], 10)
        self.assertEqual(splits["test"][0].shape[0], 20)


if __name__ == "__main__":
    unittest.main()
