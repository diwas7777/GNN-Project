import unittest

import torch

from stgat.engine import save_checkpoint
from stgat.model import GatedFusion, STGAT, STGATWithAdjacency


class ModelTest(unittest.TestCase):
    def test_stgat_forward_shape_and_adaptive_adjacency_registration(self):
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
        x = torch.randn(3, 12, 5, 2)
        physical_adjacency = torch.eye(5)

        output = model(x, physical_adjacency)

        self.assertEqual(output.shape, (3, 12, 5, 1))
        named_parameters = dict(model.named_parameters())
        self.assertIn("adaptive_adjacency", named_parameters)
        self.assertTrue(named_parameters["adaptive_adjacency"].requires_grad)
        self.assertTrue(any("attention_heads" in name for name in named_parameters))

    def test_adaptive_adjacency_receives_gradient_after_training_step(self):
        torch.manual_seed(7)
        model = STGAT(
            num_nodes=4,
            input_features=2,
            input_steps=12,
            output_steps=12,
            hidden_channels=6,
            attention_heads=(2, 2),
            blocks=2,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(2, 12, 4, 2)
        y = torch.randn(2, 12, 4, 1)

        optimizer.zero_grad()
        loss = torch.nn.functional.l1_loss(model(x, torch.eye(4)), y)
        loss.backward()
        before = model.adaptive_adjacency.detach().clone()
        optimizer.step()

        self.assertIsNotNone(model.adaptive_adjacency.grad)
        self.assertFalse(torch.equal(before, model.adaptive_adjacency.detach()))

    def test_gated_fusion_is_learned_and_shape_preserving(self):
        fusion = GatedFusion(channels=8)
        physical = torch.randn(2, 12, 5, 8)
        adaptive = torch.randn(2, 12, 5, 8)

        fused = fusion(physical, adaptive)

        self.assertEqual(fused.shape, physical.shape)
        self.assertTrue(any(parameter.requires_grad for parameter in fusion.parameters()))

    def test_adjacency_bound_model_saves_base_model_state(self):
        import tempfile
        from pathlib import Path

        model = STGAT(
            num_nodes=4,
            input_features=2,
            input_steps=12,
            output_steps=12,
            hidden_channels=6,
            attention_heads=(2, 2),
            blocks=2,
            dropout=0.0,
        )
        wrapped = STGATWithAdjacency(model, torch.eye(4))
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(path, wrapped, optimizer, epoch=1, config={}, best_val_mae=1.0)
            checkpoint = torch.load(path, map_location="cpu")

        self.assertIn("adaptive_adjacency", checkpoint["model_state_dict"])
        self.assertFalse(any(key.startswith("model.") for key in checkpoint["model_state_dict"]))


if __name__ == "__main__":
    unittest.main()
