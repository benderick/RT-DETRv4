import unittest

import torch

from engine.core import YAMLConfig


class FullDFINESmokeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "full 1024 smoke test requires CUDA")
    def test_real_image_full_model_forward_backward_inference(self):
        config = YAMLConfig("configs/dfine/dfine_hgnetv2_s_codrone_obb_smoke.yml")
        images, targets = next(iter(config.train_dataloader))
        device = torch.device("cuda")
        model = config.model.to(device).train()
        criterion = config.criterion.to(device)
        images = images.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(images, targets)
        loss = sum(criterion(outputs, targets).values())
        loss.backward()
        self.assertEqual(outputs["pred_boxes"].shape[-1], 5)
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.inference_mode():
            predictions = config.postprocessor.to(device)(model(images), targets)
        self.assertEqual(len(predictions), len(images))
        self.assertEqual(predictions[0]["boxes"].shape[-1], 5)


if __name__ == "__main__":
    unittest.main()
