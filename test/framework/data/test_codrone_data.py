import math
import random
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from engine.data.dataset import CODroneDetection
from engine.core import YAMLConfig
from engine.data.transforms import (
    Compose,
    RotatedConvertToTensor,
    RotatedPad,
    RotatedRandomFlip,
    RotatedRandomRotate,
    RotatedResize,
    RotatedResizePad,
    RotatedSanitizeBoxes,
)
from engine.rtv4.rotated_box_ops import rbox_to_corners


DATA_ROOT = Path("/home/liuxiaolong/futurama/data/CODrone")
STANDARD_TINY_ROOT = DATA_ROOT / "standard_patches_t"


class CODroneTileConfigTest(unittest.TestCase):
    def test_primary_o2_tile_protocol_is_exact(self):
        config = YAMLConfig(
            "configs/dfine/dfine_obb_o2_tile.yml")
        yaml = config.yaml_cfg
        self.assertEqual(yaml["HGNetv2"]["name"], "B2")
        self.assertEqual(yaml["HybridEncoder"]["in_channels"], [384, 768, 1536])
        self.assertEqual(yaml["RotatedDFINETransformer"]["num_layers"], 4)
        self.assertEqual(
            yaml["RotatedDFINETransformer"]["refinement_mode"], "o2_adr")
        self.assertEqual(yaml["RotatedDFINETransformer"]["ocd_mode"], "box")
        self.assertEqual(yaml["eval_spatial_size"], [1024, 1024])
        self.assertEqual(yaml["epoches"], 72)
        self.assertEqual(yaml["evaluator"]["type"], "MergedDotaOBBEvaluator")
        self.assertEqual(yaml["evaluator"]["merge_iou_threshold"], 0.1)
        self.assertEqual(yaml["train_dataloader"]["total_batch_size"], 8)
        self.assertIsNone(yaml["train_dataloader"]["collate_fn"]["base_size_repeat"])
        self.assertEqual(yaml["train_dataloader"]["collate_fn"]["mixup_prob"], 0.0)
        transform_names = {
            operation["type"]
            for operation in yaml["train_dataloader"]["dataset"]["transforms"]["ops"]
        }
        self.assertNotIn("Mosaic", transform_names)
        self.assertNotIn("MixUp", transform_names)


@unittest.skipUnless(DATA_ROOT.is_dir(), "local CODrone dataset is unavailable")
class CODroneDataTest(unittest.TestCase):
    @unittest.skipUnless(
        (STANDARD_TINY_ROOT / "train_t" / "_SUCCESS").is_file(),
        "official tiny CODrone patches have not been materialized",
    )
    def test_official_patch_metadata_reaches_the_model_target(self):
        dataset = CODroneDetection(STANDARD_TINY_ROOT / "train_t")
        self.assertEqual(len(dataset), 120)
        assignment_count = 0
        for index in range(len(dataset)):
            image, target = dataset.load_item(index)
            self.assertEqual(image.size, (1180, 1180))
            self.assertTrue(target["partition_id"].startswith("codrone_dota_w1180_g200"))
            self.assertEqual(target["tile_size"].tolist(), [1180, 1180])
            self.assertEqual(target["tile_overlap"].tolist(), [200, 200])
            self.assertEqual(len(target["boxes"]), len(target["source_object_index"]))
            self.assertTrue(torch.isfinite(target["boundary_distance_px"]).all())
            self.assertTrue((target["visible_ratio"] >= 0.7).all())
            self.assertTrue((target["source_tile_count"] >= 1).all())
            assignment_count += len(target["boxes"])
        self.assertEqual(assignment_count, 1279)

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
        self.assertEqual(int(flipped["aug_flip_code"]), 1)
        target = {"boxes": torch.tensor([[25.0, 30.0, 20.0, 10.0, 0.2]])}
        with mock.patch("random.random", return_value=0.0), mock.patch("random.uniform", return_value=90.0):
            _, rotated, _ = RotatedRandomRotate(p=1, angle_range=180)((image, target, None))
        self.assertAlmostEqual(float(rotated["boxes"][0, 0]), 30.0, places=4)
        self.assertAlmostEqual(float(rotated["boxes"][0, 1]), 75.0, places=4)
        self.assertAlmostEqual(float(rotated["aug_rotation_degrees"]), 90.0)
        self.assertTrue(bool(rotated["aug_rotation_applied"]))

    def test_resize_rotate_then_pad_uses_the_resized_image_center(self):
        image = Image.new("RGB", (200, 100))
        target = {
            "boxes": torch.tensor([[150.0, 50.0, 20.0, 10.0, 0.2]])
        }
        image, target, _ = RotatedResize((100, 100))((image, target, None))
        self.assertEqual(image.size, (100, 50))
        torch.testing.assert_close(
            target["boxes"][0, :2], torch.tensor([75.0, 25.0]))
        with mock.patch("random.random", return_value=0.0), \
                mock.patch("random.uniform", return_value=90.0):
            image, target, _ = RotatedRandomRotate(p=1)(
                (image, target, None))
        # Rotation is around (50, 25), before right/bottom padding.  Padding
        # first would incorrectly rotate around (50, 50) and produce y=25.
        torch.testing.assert_close(
            target["boxes"][0, :2], torch.tensor([50.0, 0.0]),
            atol=1e-5, rtol=0)
        image, target, _ = RotatedPad((100, 100))((image, target, None))
        self.assertEqual(image.size, (100, 100))
        torch.testing.assert_close(
            target["boxes"][0, :2], torch.tensor([50.0, 0.0]),
            atol=1e-5, rtol=0)
        self.assertEqual(target["padding"].tolist(), [0, 0, 0, 50])

    def test_random_augmentation_invariants(self):
        transforms = Compose(ops=[
            {"type": "RotatedResize", "size": [1024, 1024]},
            {"type": "RotatedRandomFlip", "p": 0.75,
             "directions": ["horizontal", "vertical", "diagonal"]},
            {"type": "RotatedRandomRotate", "p": 0.5, "angle_range": 180},
            {"type": "RotatedSanitizeBoxes", "min_size": 1, "min_visible": 0.2},
            {"type": "RotatedPad", "size": [1024, 1024]},
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
