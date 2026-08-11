import math
import unittest

import torch

from engine.rtv4.rotated_box_ops import (
    aligned_kld_loss,
    angle_distance,
    class_aware_rotated_nms,
    corners_to_rboxes,
    pairwise_chamfer_cost,
    regularize_rboxes,
    rbox_to_corners,
    rotated_iou,
)


class RotatedBoxOpsTest(unittest.TestCase):
    def test_canonicalization_preserves_geometry(self):
        boxes = torch.tensor([[10.0, 20.0, 3.0, 8.0, 0.2]])
        canonical = regularize_rboxes(boxes)
        self.assertGreaterEqual(float(canonical[0, 2]), float(canonical[0, 3]))
        self.assertAlmostEqual(float(rotated_iou(boxes, canonical, normalized_angle=False)[0, 0]), 1.0, places=5)

    def test_corner_round_trip(self):
        boxes = torch.tensor([
            [100.0, 80.0, 30.0, 12.0, 0.0],
            [30.0, 20.0, 9.0, 4.0, math.pi - 0.01],
            [50.0, 70.0, 10.0, 10.0, 0.7],
        ])
        recovered = corners_to_rboxes(rbox_to_corners(boxes))
        overlaps = rotated_iou(boxes, recovered, aligned=True, normalized_angle=False)
        torch.testing.assert_close(overlaps, torch.ones_like(overlaps), atol=1e-5, rtol=1e-5)

    def test_periodic_angle_distance(self):
        distance = angle_distance(torch.tensor([0.99]), torch.tensor([0.01]))
        self.assertAlmostEqual(float(distance), 0.02, places=6)

    def test_iou_and_class_aware_nms(self):
        boxes = torch.tensor([
            [10.0, 10.0, 8.0, 4.0, 0.2],
            [10.1, 10.1, 8.0, 4.0, 0.2],
            [10.1, 10.1, 8.0, 4.0, 0.2],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])
        keep = class_aware_rotated_nms(boxes, scores, labels, 0.5)
        self.assertEqual(keep.tolist(), [0, 2])
        self.assertGreater(float(rotated_iou(boxes[:1], boxes[1:2], normalized_angle=False)), 0.9)

    def test_geometry_losses_are_finite_and_differentiable(self):
        prediction = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.99]], requires_grad=True)
        target = torch.tensor([[0.51, 0.49, 0.19, 0.11, 0.01]])
        loss = aligned_kld_loss(prediction, target).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        cost = pairwise_chamfer_cost(prediction.detach(), target)
        self.assertEqual(tuple(cost.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
