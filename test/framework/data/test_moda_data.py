import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine.core import YAMLConfig
from engine.data.dataset import MODADetection
from engine.data.dataset.moda_dataset import annotation_inventory
from engine.data.transforms import (RotatedResize, RotatedRandomFlip, RotatedRandomRotate,
    RotatedPad, RotatedSanitizeBoxes, RotatedConvertToTensor)
from engine.evaluation.obb import DotaOBBEvaluator
from engine.solver.det_solver import checkpoint_selection_score
from engine.rtv4.obb_visualization import draw_obbs
from tools.dataset.prepare_moda_splits import prepare_splits


ROOT = Path(__file__).resolve().parents[3]


def make_data(root, count=1):
    root = Path(root) / "train"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for i in range(count):
        name = f"2023010{i+1}_frame"
        image = np.zeros((8, 80, 60), dtype=np.uint8)
        # A non-square rectangle whose pixels and GT share known coordinates.
        image[:, 20:40, 15:25] = np.arange(1, 9)[:, None, None] * 20
        np.save(root / "images" / f"{name}.npy", image)
        (root / "labels" / f"{name}.txt").write_text(f"20 15 40 15 40 25 20 25 car {i%3}\n")
    return root


class MODADataTest(unittest.TestCase):
    def test_cwh_pixels_difficulty_and_perfect_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_data(tmp, 3)
            dataset = MODADetection(root)
            evaluator = DotaOBBEvaluator(dataset, selection_metric="mAP50_95")
            for i in range(3):
                image, target = dataset.load_item(i)
                self.assertEqual(tuple(image.shape), (8, 60, 80))
                self.assertEqual(target["orig_size"].tolist(), [80, 60])
                self.assertEqual(image[:,20,30].tolist(), list(range(20,161,20)))
                self.assertEqual(image[:,30,20].sum(), 0)
                self.assertEqual(target["difficulty"].tolist(), [0])
                self.assertEqual(target["source_difficulty"].tolist(), [i])
                evaluator.update({i: dict(boxes=target["boxes"], labels=target["labels"], scores=torch.ones(1))})
            evaluator.accumulate(verbose=False)
            self.assertAlmostEqual(evaluator.metrics["mAP50_95"], 1)
            self.assertEqual(checkpoint_selection_score([0.1,0.2,0.3,0.7,0.5,0.6], evaluator), 0.7)
            preview = draw_obbs(root / "images" / "20230101_frame.npy", target["boxes"], target["labels"])
            self.assertEqual(preview.size, (80,60))

    def test_missing_images_require_explicit_hashed_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_data(tmp, 2)
            (root / "images" / "20230102_frame.npy").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "1 missing images"):
                MODADetection(root)
            output = Path(tmp)/"splits"
            prepare_splits(root, output, debug_images=1)
            dataset = MODADetection(root, split_file=output/"debug.json")
            self.assertEqual(len(dataset),1)
            (root/"labels"/"20230101_frame.txt").write_text("")
            with self.assertRaisesRegex(ValueError,"inventory differs"):
                MODADetection(root, split_file=output/"debug.json")

    def test_sync_flip_rotation_padding_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            image, target = MODADetection(make_data(tmp)).load_item(0)
            flipped, t, _ = RotatedRandomFlip(p=1, directions=["horizontal"])((image,target,None))
            torch.testing.assert_close(t["boxes"][0,:2], torch.tensor([50.,20.]))
            self.assertEqual(flipped[0,20,49], 20)
            back, t, _ = RotatedRandomFlip(p=1, directions=["horizontal"])((flipped,t,None))
            torch.testing.assert_close(back,image)
            with patch("engine.data.transforms.rotated_transforms.random.uniform",return_value=90.):
                rotated, t, _ = RotatedRandomRotate(p=1,fill=0)((image,t,None))
            torch.testing.assert_close(t["boxes"][0,:2], torch.tensor([30.,40.]), atol=1e-4,rtol=0)
            self.assertEqual(rotated[0,40,30],20)
            self.assertLess(int(t["valid_mask"].sum()),80*60)
            padded,t,_=RotatedPad((96,96),fill=0)((rotated,t,None))
            self.assertEqual(tuple(padded.shape),(8,96,96))
            self.assertFalse(t["valid_mask"][60:,:].any())
            tensor,t,_=RotatedConvertToTensor()((padded,t,None))
            self.assertEqual(tensor.dtype,torch.float32)
            self.assertAlmostEqual(float(tensor[0,40,30]),20/255,places=6)
            self.assertEqual(t["padding"].tolist(),[0,0,16,36])

    def test_resize_and_empty_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_data(tmp)
            dataset=MODADetection(root)
            image,t=dataset.load_item(0)
            image,t,_=RotatedResize((160,160))((image,t,None))
            self.assertEqual(tuple(image.shape),(8,120,160))
            torch.testing.assert_close(t["boxes"][0,:4],torch.tensor([60.,40.,40.,20.]))
            (root/"labels"/"20230101_frame.txt").write_text("")
            image,t=MODADetection(root).load_item(0)
            _,t,_=RotatedSanitizeBoxes()((image,t,None))
            self.assertEqual(t["boxes"].shape,(0,5))
            self.assertEqual(t["source_difficulty"].shape,(0,))

    def test_group_manifests_disjoint_and_configs_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=make_data(tmp,6)
            mapping={p.stem:f"scene_{i//2}" for i,p in enumerate(sorted((root/"labels").glob("*.txt")))}
            path=Path(tmp)/"groups.json"; path.write_text(json.dumps(mapping))
            out=Path(tmp)/"split"
            prepare_splits(root,out,group_map=path,dev_fraction=0.33)
            train=json.loads((out/"train.json").read_text()); dev=json.loads((out/"dev.json").read_text())
            self.assertFalse(set(train["group_ids"]) & set(dev["group_ids"]))
            self.assertEqual(len(MODADetection(root,split_file=out/"train.json"))+len(MODADetection(root,split_file=out/"dev.json")),6)
            with self.assertRaises(FileExistsError):
                prepare_splits(root,out,group_map=path)
        for method,mode in (("o2","o2_adr"),("angle","direct_angle")):
            config=YAMLConfig(str(ROOT/f"configs/experiments/moda/dfine_obb_{method}_fressdet.yml"))
            self.assertEqual(config.yaml_cfg["HGNetv2"]["in_channels"],8)
            self.assertEqual(config.yaml_cfg["RotatedDFINETransformer"]["refinement_mode"],mode)
            self.assertEqual(config.yaml_cfg["val_dataloader"]["dataset"]["root"],"./data/MODA/test")
            self.assertEqual(config.yaml_cfg["evaluator"]["selection_metric"],"AP50")
            self.assertEqual(config.epoches,20)


if __name__ == "__main__":
    unittest.main()
