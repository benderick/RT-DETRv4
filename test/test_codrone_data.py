import math
import random
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from engine.data.dataset import CODroneDetection
from engine.data.transforms import (
    Compose,
    RotatedConvertToTensor,
    RotatedRandomFlip,
    RotatedRandomRotate,
    RotatedResizePad,
    RotatedSanitizeBoxes,
)
from engine.rtv4.rotated_box_ops import rbox_to_corners


DATA_ROOT = Path("/home/liuxiaolong/futurama/data/CODrone")


@unittest.skipUnless(DATA_ROOT.is_dir(), "local CODrone dataset is unavailable")
class CODroneDataTest(unittest.TestCase):
    def test_tiny_split_loading_and_empty_annotations(self):
        train = CODroneDetection(DATA_ROOT / "train_t")
        val = CODroneDetection(DATA_ROOT / "val_t")
        test = CODroneDetection(DATA_ROOT / "test_t")
        self.assertEqual((len(train), len(val), len(test)), (15, 17, 17))
        self.assertEqual(len(train.classes), 12)
        empty_count = sum(len(val.get_ground_truth(index)["boxes"]) == 0 for index in range(len(val)))
        self.assertGreaterEqual(empty_count, 2)
        self.assertEqual(sum(len(val.get_ground_truth(index)["ignore_boxes"])
                             for index in range(len(val))), 6)
        image, target = train.load_item(0)
        self.assertEqual(image.size, (3840, 2160))
        self.assertEqual(target["boxes"].shape[1], 5)
        self.assertTrue((target["boxes"][:, 2] >= target["boxes"][:, 3]).all())

    def test_resize_pad_metadata_and_normalization(self):
        dataset = CODroneDetection(DATA_ROOT / "train_t")
        image, target = dataset.load_item(0)
        image, target, _ = RotatedResizePad((1024, 1024))((image, target, dataset))
        self.assertEqual(image.size, (1024, 1024))
        torch.testing.assert_close(target["scale_factor"], torch.tensor([1024 / 3840, 1024 / 3840]))
        self.assertEqual(target["padding"].tolist(), [0, 0, 0, 448])
        image, target, _ = RotatedConvertToTensor()((image, target, dataset))
        self.assertEqual(tuple(image.shape), (3, 1024, 1024))
        self.assertTrue((target["boxes"][:, :4] >= 0).all())
        self.assertTrue((target["boxes"][:, 4] < 1).all())

    def test_forced_flip_and_rotation_geometry(self):
        image = Image.new("RGB", (100, 100))
        target = {"boxes": torch.tensor([[25.0, 30.0, 20.0, 10.0, 0.2]])}
        with mock.patch("random.random", return_value=0.0), mock.patch("random.choice", return_value="horizontal"):
            _, flipped, _ = RotatedRandomFlip(p=1)((image, target, None))
        self.assertAlmostEqual(float(flipped["boxes"][0, 0]), 75.0)
        self.assertAlmostEqual(float(flipped["boxes"][0, 4]), math.pi - 0.2, places=5)
        target = {"boxes": torch.tensor([[25.0, 30.0, 20.0, 10.0, 0.2]])}
        with mock.patch("random.random", return_value=0.0), mock.patch("random.uniform", return_value=90.0):
            _, rotated, _ = RotatedRandomRotate(p=1, angle_range=180)((image, target, None))
        self.assertAlmostEqual(float(rotated["boxes"][0, 0]), 30.0, places=4)
        self.assertAlmostEqual(float(rotated["boxes"][0, 1]), 75.0, places=4)

    def test_random_augmentation_invariants(self):
        transforms = Compose(ops=[
            {"type": "RotatedResizePad", "size": [1024, 1024]},
            {"type": "RotatedRandomFlip", "p": 0.75,
             "directions": ["horizontal", "vertical", "diagonal"]},
            {"type": "RotatedRandomRotate", "p": 0.5, "angle_range": 180},
            {"type": "RotatedSanitizeBoxes", "min_size": 1, "min_visible": 0.2},
            {"type": "RotatedConvertToTensor"},
        ])
        dataset = CODroneDetection(DATA_ROOT / "train_t", transforms)
        for seed in range(10):
            random.seed(seed)
            torch.manual_seed(seed)
            image, target = dataset[seed % len(dataset)]
            boxes = target["boxes"]
            self.assertEqual(tuple(image.shape), (3, 1024, 1024))
            self.assertTrue(torch.isfinite(boxes).all())
            self.assertTrue((boxes[:, 2] >= boxes[:, 3]).all())
            self.assertTrue(((boxes[:, 4] >= 0) & (boxes[:, 4] < 1)).all())
            self.assertTrue(torch.isfinite(rbox_to_corners(boxes, normalized_angle=True)).all())


if __name__ == "__main__":
    unittest.main()
