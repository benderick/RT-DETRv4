import unittest
from pathlib import Path

import torch

from engine.data.dataset import CODroneDetection


DATA_ROOT = Path("/home/liuxiaolong/futurama/data/CODrone")


@unittest.skipUnless(DATA_ROOT.is_dir(), "local CODrone dataset is unavailable")
class FullDatasetAuditTest(unittest.TestCase):
    def test_every_full_split_annotation_parses(self):
        expected = {
            "train": (5002, 219527),
            "val": (2000, 89468),
            "test": (3002, 134597),
        }
        ignored_objects = 0
        for split, (image_count, object_count) in expected.items():
            dataset = CODroneDetection(DATA_ROOT / split)
            self.assertEqual(len(dataset), image_count)
            parsed_objects = 0
            for index in range(len(dataset)):
                ground_truth = dataset.get_ground_truth(index)
                boxes, labels = ground_truth["boxes"], ground_truth["labels"]
                self.assertTrue(torch.isfinite(boxes).all())
                self.assertTrue((boxes[:, 2] >= boxes[:, 3]).all())
                self.assertTrue(((labels >= 0) & (labels < 12)).all())
                parsed_objects += len(boxes)
                ignored_objects += len(ground_truth["ignore_boxes"])
            self.assertEqual(parsed_objects, object_count)
        self.assertEqual(ignored_objects, 0)


if __name__ == "__main__":
    unittest.main()
