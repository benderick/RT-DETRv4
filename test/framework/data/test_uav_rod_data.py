import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from engine.core import YAMLConfig
from engine.data.dataset import UAVRODDetection
from engine.evaluation.obb import DotaOBBEvaluator
from engine.rtv4.rotated_box_ops import rbox_to_corners
from tools.dataset.convert_uav_rod_to_dota import (
    annotation_to_dota_lines,
    convert_split,
    parse_rotated_voc,
    rotated_box_to_dota_corners,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/home/liuxiaolong/futurama/data/UAV-ROD")


def _write_example_split(
    root: Path, *, class_name: str = "car", difficult: int = 0
) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(
        root / "images" / "DJI_0001_000030.jpg")
    xml = f"""<annotation>
  <filename>DJI_0001_000030.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>{class_name}</name><truncated>1</truncated><difficult>{difficult}</difficult>
    <robndbox><cx>50</cx><cy>40</cy><w>20</w><h>40</h><angle>0</angle></robndbox>
  </object>
</annotation>
"""
    (root / "annotations" / "DJI_0001_000030.xml").write_text(
        xml, encoding="utf-8")
    return root


class UAVRODConversionUnitTest(unittest.TestCase):
    def test_clockwise_image_axis_geometry(self):
        corners = rotated_box_to_dota_corners(50, 40, 20, 40, 0)
        self.assertEqual(corners, (40, 20, 60, 20, 60, 60, 40, 60))
        quarter_turn = rotated_box_to_dota_corners(
            50, 40, 20, 40, math.pi / 2)
        expected = (70, 30, 70, 50, 30, 50, 30, 30)
        for actual, wanted in zip(quarter_turn, expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_parse_and_format_preserve_difficulty(self):
        with tempfile.TemporaryDirectory() as directory:
            split = _write_example_split(Path(directory), difficult=1)
            annotation = parse_rotated_voc(
                split / "annotations" / "DJI_0001_000030.xml")
            self.assertEqual(annotation.objects[0].difficult, 1)
            self.assertEqual(annotation.objects[0].truncated, 1)
            self.assertEqual(
                annotation_to_dota_lines(annotation),
                ["40 20 60 20 60 60 40 60 car 1"],
            )

    def test_conversion_adapter_and_perfect_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            split = _write_example_split(Path(directory) / "train")
            summary = convert_split(split)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["objects"], 1)
            self.assertEqual(summary["difficult_objects"], 0)
            self.assertEqual(summary["truncated_objects"], 1)
            self.assertTrue((split / "conversion_manifest.json").is_file())
            validate = convert_split(split, validate_only=True)
            self.assertEqual(
                validate["output_inventory_sha256"],
                summary["output_inventory_sha256"],
            )
            with self.assertRaises(FileExistsError):
                convert_split(split)

            dataset = UAVRODDetection(split)
            image, target = dataset.load_item(0)
            self.assertEqual(image.size, (100, 80))
            self.assertEqual(target["boxes"].shape, (1, 5))
            torch.testing.assert_close(
                target["boxes"][0],
                torch.tensor([50.0, 40.0, 40.0, 20.0, math.pi / 2]),
            )
            self.assertEqual(target["difficulty"].tolist(), [0])
            self.assertEqual(
                dataset.get_image_metadata(0),
                {"video_id": "DJI_0001", "frame_index": 30},
            )
            provenance = dataset.get_dataset_provenance()
            self.assertEqual(
                provenance["conversion"]["source_format"],
                "VOC XML robndbox(cx,cy,w,h,angle_radians_clockwise)",
            )

            evaluator = DotaOBBEvaluator(dataset, iou_thresholds=[0.5, 0.75])
            evaluator.update({
                0: {
                    "boxes": target["boxes"].clone(),
                    "scores": torch.ones(1),
                    "labels": torch.zeros(1, dtype=torch.int64),
                }
            })
            evaluator.accumulate()
            self.assertEqual(evaluator.metrics["AP50_DOTA07"], 1.0)
            self.assertEqual(evaluator.metrics["AP75_DOTA07"], 1.0)

    def test_unknown_class_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            split = _write_example_split(
                Path(directory) / "train", class_name="truck")
            with self.assertRaisesRegex(ValueError, "Unknown UAV-ROD class"):
                convert_split(split)


class UAVRODConfigTest(unittest.TestCase):
    def test_angle_and_o2_bindings_resolve_uav_rod(self):
        expected = {
            "dfine_obb_angle.yml": "direct_angle",
            "dfine_obb_o2.yml": "o2_adr",
        }
        for config_name, mode in expected.items():
            config = YAMLConfig(str(
                ROOT / "configs" / "experiments" / "uav_rod" / config_name))
            yaml = config.yaml_cfg
            self.assertEqual(yaml["num_classes"], 1)
            self.assertEqual(
                yaml["train_dataloader"]["dataset"]["type"],
                "UAVRODDetection",
            )
            self.assertEqual(
                yaml["val_dataloader"]["dataset"]["root"],
                "../data/UAV-ROD/test",
            )
            self.assertEqual(
                yaml["RotatedDFINETransformer"]["refinement_mode"], mode)
            self.assertEqual(
                yaml["RotatedDFINETransformer"]["num_queries"], 300)
            self.assertEqual(yaml["evaluator"]["type"], "DotaOBBEvaluator")
            self.assertEqual(
                yaml["train_dataloader"]["collate_fn"]["mixup_prob"], 0.0)


@unittest.skipUnless(
    (DATA_ROOT / "train" / "annfile").is_dir()
    and (DATA_ROOT / "test" / "annfile").is_dir(),
    "converted local UAV-ROD dataset is unavailable",
)
class UAVRODFullDatasetTest(unittest.TestCase):
    def test_official_inventory_and_geometry(self):
        train = UAVRODDetection(DATA_ROOT / "train")
        test = UAVRODDetection(DATA_ROOT / "test")
        self.assertEqual((len(train), len(test)), (1150, 427))
        self.assertEqual(train.classes, ("car",))
        object_count = sum(
            len(dataset.get_ground_truth(index)["boxes"])
            for dataset in (train, test)
            for index in range(len(dataset))
        )
        self.assertEqual(object_count, 30090)
        for dataset in (train, test):
            for index in (0, len(dataset) // 2, len(dataset) - 1):
                image, target = dataset.load_item(index)
                boxes = target["boxes"]
                self.assertEqual(image.size, tuple(target["orig_size"].tolist()))
                self.assertTrue(torch.isfinite(boxes).all())
                self.assertTrue((boxes[:, 2] >= boxes[:, 3]).all())
                self.assertTrue(((boxes[:, 4] >= 0) & (boxes[:, 4] < math.pi)).all())
                self.assertTrue(torch.isfinite(rbox_to_corners(boxes)).all())

    def test_configured_training_and_validation_loading(self):
        config_path = (
            ROOT / "configs" / "experiments" / "uav_rod" / "dfine_obb_o2.yml"
        )
        config = YAMLConfig(
            str(config_path),
            train_dataloader={"total_batch_size": 2, "num_workers": 0},
            val_dataloader={"total_batch_size": 2, "num_workers": 0},
        )
        train_images, train_targets = next(iter(config.train_dataloader))
        val_images, val_targets = next(iter(config.val_dataloader))
        for images, targets in (
            (train_images, train_targets), (val_images, val_targets)
        ):
            self.assertEqual(tuple(images.shape), (2, 3, 1024, 1024))
            self.assertEqual(len(targets), 2)
            for target in targets:
                boxes = target["boxes"]
                self.assertTrue(torch.isfinite(boxes).all())
                self.assertTrue((boxes[:, 2] >= boxes[:, 3]).all())
                self.assertTrue(((boxes[:, 4] >= 0) & (boxes[:, 4] < 1)).all())


if __name__ == "__main__":
    unittest.main()
