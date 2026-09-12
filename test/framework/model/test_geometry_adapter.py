"""Optional geometry context must preserve both stable decoder paths."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from engine.backbone.hgnetv2 import HGNetv2
from engine.core import YAMLConfig, GLOBAL_CONFIG
from test.framework.model.test_model_pipeline import build_tiny_model, build_criterion


class ZeroAdapter(nn.Module):
    def forward(self, queries, references, context, layer):
        context.append((layer, references.detach().clone()))
        return torch.zeros_like(queries)


class GeometryAdapterTest(unittest.TestCase):
    def test_config_and_registry_are_isolated_across_datasets(self):
        baseline_path = "configs/experiments/uav_rod/dfine_obb_o2.yml"
        expected = YAMLConfig(baseline_path).yaml_cfg
        moda = YAMLConfig("configs/experiments/moda/dfine_obb_o2.yml")
        registry = moda.global_cfg
        registry["HGNetv2"]["_kwargs"]["in_channels"] = 777
        self.assertEqual(GLOBAL_CONFIG["HGNetv2"]["_kwargs"]["in_channels"], 3)
        self.assertEqual(YAMLConfig(baseline_path).yaml_cfg, expected)

    def test_zero_adapter_preserves_outputs_losses_and_gradients(self):
        for mode in ("o2_adr", "direct_angle"):
            torch.manual_seed(17)
            baseline = build_tiny_model(refinement_mode=mode)
            extended = copy.deepcopy(baseline)
            extended.geometry_adapter = ZeroAdapter()
            self.assertEqual(set(baseline.state_dict()), set(extended.state_dict()))
            extended.load_state_dict(baseline.state_dict(), strict=True)
            features = [torch.randn(2, 32, n, n) for n in (8, 4, 2)]
            targets = [dict(labels=torch.tensor([1]), boxes=torch.tensor([[.4,.6,.2,.1,.95]])),
                       dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0,5))]
            for training in (True, False):
                baseline.train(training)
                extended.train(training)
                torch.manual_seed(31)
                left = baseline(features, targets)
                torch.manual_seed(31)
                context = []
                right = extended(features, targets, context=context)
                self.assertEqual(len(context), 2)
                for key in ("pred_boxes", "pred_logits"):
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                if training:
                    losses_left = build_criterion(mode)(left, targets)
                    losses_right = build_criterion(mode)(right, targets)
                    self.assertEqual(set(losses_left), set(losses_right))
                    for key in losses_left:
                        torch.testing.assert_close(losses_left[key], losses_right[key], rtol=0, atol=0)
                    sum(losses_left.values()).backward()
                    sum(losses_right.values()).backward()
                    for a, b in zip(baseline.parameters(), extended.parameters()):
                        if a.grad is not None:
                            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "requires image context"):
                extended(features)

    def test_rgb_pretrained_migration_preserves_identical_band_response(self):
        torch.manual_seed(2)
        original = HGNetv2("B0", pretrained=False, freeze_norm=False)
        with tempfile.TemporaryDirectory() as folder:
            torch.save(original.state_dict(), Path(folder) / "PPHGNetV2_B0_stage1.pth")
            rgb = HGNetv2("B0", local_model_dir=folder+"/", freeze_norm=False)
            multispectral = HGNetv2("B0", in_channels=8, local_model_dir=folder+"/", freeze_norm=False)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, rgb.state_dict()[name], rtol=0, atol=0)
            mono = torch.randn(1, 1, 32, 32)
            left = rgb.stem.stem1.conv(mono.repeat(1, 3, 1, 1))
            right = multispectral.stem.stem1.conv(mono.repeat(1, 8, 1, 1))
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
