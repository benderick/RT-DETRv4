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
        self.assertAlmostEqual(float(rotated_iou(boxes, canonical, model_space=False)[0, 0]), 1.0, places=5)

    def test_corner_round_trip(self):
        boxes = torch.tensor([
            [100.0, 80.0, 30.0, 12.0, 0.0],
            [30.0, 20.0, 9.0, 4.0, math.pi - 0.01],
            [50.0, 70.0, 10.0, 10.0, 0.7],
        ])
        recovered = corners_to_rboxes(rbox_to_corners(boxes))
        overlaps = rotated_iou(boxes, recovered, aligned=True, model_space=False)
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
        self.assertGreater(float(rotated_iou(boxes[:1], boxes[1:2], model_space=False)), 0.9)

    def test_model_space_iou_is_stable_at_the_half_turn_seam(self):
        # This real UAV-ROD pair must describe the same overlap before and
        # after the model/pixel coordinate conversion.
        pixel_prediction = torch.tensor([[
            36.52423095703125, 744.5892944335938,
            68.25360107421875, 58.81178283691406,
            0.020618179813027382,
        ]])
        pixel_target = torch.tensor([[
            34.46719741821289, 744.7645874023438,
            69.01750183105469, 60.52589416503906,
            3.140796661376953,
        ]])
        expected = rotated_iou(
            pixel_prediction, pixel_target, aligned=True, model_space=False)

        canvas = 1920.0
        model_prediction = pixel_prediction.clone()
        model_target = pixel_target.clone()
        model_prediction[..., :4] /= canvas
        model_target[..., :4] /= canvas
        model_prediction[..., 4] /= math.pi
        model_target[..., 4] /= math.pi
        observed = rotated_iou(
            model_prediction, model_target, aligned=True, model_space=True)
        pairwise = rotated_iou(
            model_prediction, model_target, aligned=False, model_space=True)

        self.assertGreater(float(expected), .9)
        torch.testing.assert_close(observed, expected, atol=2e-5, rtol=0)
        torch.testing.assert_close(pairwise[0, 0], expected[0], atol=2e-5, rtol=0)

    def test_nearly_coincident_thin_boxes_have_unit_iou_and_are_suppressed(self):
        # Near-coincident geometry must not split into duplicate detections.
        boxes = torch.tensor([
            [0.11650142818689346, 0.35978102684020996,
             0.2270190268754959, 0.015340147539973259,
             0.8291192054748535],
            [0.11650142073631287, 0.35978102684020996,
             0.2270190566778183, 0.015340177342295647,
             0.8291192054748535],
        ])
        overlap = rotated_iou(
            boxes[:1], boxes[1:], aligned=True, model_space=True)
        self.assertGreater(float(overlap), 0.99999)

        pixel_boxes = boxes.clone()
        pixel_boxes[..., :4] *= 1024.0
        pixel_boxes[..., 4] *= math.pi
        keep = class_aware_rotated_nms(
            pixel_boxes, torch.tensor([0.9, 0.8]), torch.tensor([0, 0]), 0.5)
        self.assertEqual(keep.tolist(), [0])

    def test_pixel_iou_preserves_small_sides_at_large_coordinates(self):
        boxes = torch.tensor([
            [1.0e8, 1.0e8, 2.0, 2.0, 0.0],
            [1.0e8, 1.0e8, 1.0, 1.0, 0.0],
        ])
        overlap = rotated_iou(
            boxes[:1], boxes[1:], aligned=True, model_space=False)
        self.assertAlmostEqual(float(overlap), 0.25, places=6)

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
