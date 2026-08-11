import tempfile
import unittest
from pathlib import Path

import torch

from engine.data.dataset import CODroneDetection, CODroneEvaluator
from engine.rtv4.obb_visualization import save_obb_visualization


DATA_ROOT = Path("/home/liuxiaolong/futurama/data/CODrone/val_t")


@unittest.skipUnless(DATA_ROOT.is_dir(), "local CODrone dataset is unavailable")
class EvaluatorVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.dataset = CODroneDetection(DATA_ROOT)

    def perfect_predictions(self):
        result = {}
        for image_id in range(len(self.dataset)):
            ground_truth = self.dataset.get_ground_truth(image_id)
            result[image_id] = {
                "boxes": ground_truth["boxes"].clone(),
                "labels": ground_truth["labels"].clone(),
                "scores": torch.ones(len(ground_truth["boxes"])),
            }
        return result

    def test_perfect_prediction_metrics_and_dota_export(self):
        evaluator = CODroneEvaluator(self.dataset)
        evaluator.update(self.perfect_predictions())
        evaluator.accumulate()
        for value in evaluator.metrics.values():
            self.assertAlmostEqual(value, 1.0, places=7)
        with tempfile.TemporaryDirectory() as directory:
            evaluator.export_dota_results(directory)
            files = list(Path(directory).glob("Task1_*.txt"))
            self.assertEqual(len(files), 12)
            self.assertTrue(any(path.stat().st_size > 0 for path in files))

    def test_visualization_writes_valid_image(self):
        image, _ = self.dataset.load_item(0)
        gt = self.dataset.get_ground_truth(0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "obb.jpg"
            save_obb_visualization(output, image, gt["boxes"][:20], gt["labels"][:20],
                                   class_names=self.dataset.classes)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
