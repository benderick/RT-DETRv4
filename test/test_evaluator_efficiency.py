import unittest

import torch

import engine.data.dataset.codrone_eval as codrone_eval_module
from engine.data.dataset import CODroneEvaluator


class SyntheticOBBDataset:
    classes = ("car", "bus")

    def __init__(self):
        self.image_ids = [f"image_{index}" for index in range(3)]

    def __len__(self):
        return len(self.image_ids)

    def get_ground_truth(self, image_id):
        offset = float(image_id) * 100.0
        boxes = torch.tensor([
            [offset + 10.0, 10.0, 8.0, 4.0, 0.0],
            [offset + 30.0, 10.0, 8.0, 4.0, 0.0],
        ])
        return {
            "boxes": boxes,
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "difficulty": torch.zeros(2, dtype=torch.long),
            "ignore_boxes": boxes.new_empty((0, 5)),
        }


class EvaluatorEfficiencyTest(unittest.TestCase):
    def test_iou_is_batched_by_image_and_class(self):
        dataset = SyntheticOBBDataset()
        evaluator = CODroneEvaluator(dataset, iou_thresholds=[0.5, 0.75, 0.9])
        predictions = {}
        for image_id in range(len(dataset)):
            gt = dataset.get_ground_truth(image_id)
            boxes = torch.cat((gt["boxes"], gt["boxes"]), dim=0)
            predictions[image_id] = {
                "boxes": boxes,
                "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
                "scores": torch.tensor([0.9, 0.8, 0.1, 0.1]),
            }
        evaluator.update(predictions)

        calls = []
        original_rotated_iou = codrone_eval_module.rotated_iou

        def fake_rotated_iou(boxes1, boxes2, aligned=False, normalized_angle=True):
            calls.append((tuple(boxes1.shape), tuple(boxes2.shape), aligned, normalized_angle))
            return (boxes1[:, None, 0] == boxes2[None, :, 0]).to(torch.float32)

        codrone_eval_module.rotated_iou = fake_rotated_iou
        try:
            evaluator.accumulate()
        finally:
            codrone_eval_module.rotated_iou = original_rotated_iou

        self.assertEqual(len(calls), len(dataset) * len(dataset.classes))
        self.assertTrue(torch.isfinite(torch.as_tensor(evaluator.stats)).all())


if __name__ == "__main__":
    unittest.main()
