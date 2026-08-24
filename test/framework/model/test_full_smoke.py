import unittest

import torch

from engine.core import YAMLConfig


class FullDFINESmokeTest(unittest.TestCase):
    def _run_model_step(self, config_path, **overrides):
        config = YAMLConfig(config_path, **overrides)
        images, targets = next(
            (batch for batch in config.train_dataloader
             if any(len(target["boxes"]) for target in batch[1])),
            (None, None),
        )
        self.assertIsNotNone(images, "smoke dataset contains no non-empty training batch")
        original_targets = targets
        device = torch.device("cuda")
        model = config.model.to(device).train()
        criterion = config.criterion.to(device)
        images = images.to(device)
        targets = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        with torch.autocast(
                device_type="cuda", enabled=bool(config.use_amp)):
            outputs = model(images, targets)
        # Match the real engine: model forward is autocast, geometry/loss is
        # evaluated outside autocast. The historical O² failure had a finite
        # loss but non-finite backward, so loss-only assertions are insufficient.
        with torch.autocast(device_type="cuda", enabled=False):
            loss = sum(
                criterion(outputs, targets, collect_diagnostics=True).values())
        loss.backward()
        self.assertEqual(outputs["pred_boxes"].shape[-1], 5)
        self.assertTrue(torch.isfinite(loss))
        nonfinite_gradients = sum(
            int((~torch.isfinite(parameter.grad)).sum())
            for parameter in model.parameters() if parameter.grad is not None
        )
        self.assertEqual(nonfinite_gradients, 0)
        self.assertGreater(criterion.last_diagnostics["matched_count"], 0)
        model.eval()
        with torch.inference_mode():
            predictions, diagnostics = config.postprocessor.to(device)(
                model(images), targets, return_diagnostics=True)
        self.assertEqual(len(predictions), len(images))
        self.assertEqual(predictions[0]["boxes"].shape[-1], 5)
        self.assertEqual(len(diagnostics), len(images))
        self.assertEqual(len(diagnostics[0]["status"]), len(diagnostics[0]["pre_nms_boxes"]))
        return config, original_targets

    @unittest.skipUnless(torch.cuda.is_available(), "full 1024 smoke test requires CUDA")
    def test_real_image_o2_forward_backward_inference(self):
        self._run_model_step(
            "configs/dfine/dfine_obb_o2.yml",
            HGNetv2={"pretrained": False},
            train_dataloader={
                "dataset": {"root": "/home/liuxiaolong/futurama/data/CODrone/train_t"},
                "total_batch_size": 1,
                "num_workers": 0,
                "drop_last": False,
            },
        )

    @unittest.skipUnless(torch.cuda.is_available(), "O² tile smoke test requires CUDA")
    def test_real_tile_o2_forward_backward_inference_and_metadata(self):
        config, targets = self._run_model_step(
            "configs/dfine/dfine_obb_o2_tile.yml",
            HGNetv2={"pretrained": False},
            train_dataloader={
                "dataset": {
                    "root": "/home/liuxiaolong/futurama/data/CODrone/standard_patches_t/train_t"
                },
                "total_batch_size": 1,
                "num_workers": 0,
                "drop_last": False,
            },
        )
        self.assertEqual(config.yaml_cfg["HGNetv2"]["name"], "B2")
        self.assertEqual(config.yaml_cfg["RotatedDFINETransformer"]["num_layers"], 4)
        self.assertEqual(
            config.yaml_cfg["RotatedDFINETransformer"]["refinement_mode"], "o2_adr")
        self.assertEqual(targets[0]["partition_id"], "codrone_dota_w1180_g200_iof0p7")
        self.assertEqual(targets[0]["tile_size"].tolist(), [1180, 1180])

    @unittest.skipUnless(torch.cuda.is_available(), "UAV-ROD 1024 smoke test requires CUDA")
    def test_real_uav_rod_o2_forward_backward_inference(self):
        config, targets = self._run_model_step(
            "configs/experiments/uav_rod/dfine_obb_o2.yml",
            HGNetv2={"pretrained": False},
            train_dataloader={
                "total_batch_size": 1,
                "num_workers": 0,
                "drop_last": False,
            },
        )
        self.assertEqual(config.yaml_cfg["num_classes"], 1)
        self.assertEqual(config.train_dataloader.dataset.classes, ("car",))
        self.assertGreater(len(targets[0]["boxes"]), 0)


if __name__ == "__main__":
    unittest.main()
