import unittest

import torch

from engine.rtv4 import (
    RotatedDFINETransformer,
    RotatedHungarianMatcher,
    RotatedPostProcessor,
    RotatedRTv4Criterion,
)
from engine.rtv4.rotated_denoising import get_rotated_contrastive_denoising_training_group


def build_tiny_model(
    aux_loss=True, num_denoising=10, refinement_mode="o2_adr",
):
    is_o2 = refinement_mode == "o2_adr"
    return RotatedDFINETransformer(
        num_classes=3, hidden_dim=32, num_queries=20,
        feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
        num_points=[2, 2, 2], nhead=4, num_layers=2, dim_feedforward=64,
        num_denoising=num_denoising, reg_max=8, aux_loss=aux_loss,
        refinement_mode=refinement_mode,
        ocd_mode="box" if is_o2 else "standard")


def build_criterion(refinement_mode="o2_adr"):
    is_o2 = refinement_mode == "o2_adr"
    matcher = RotatedHungarianMatcher(
        {"cost_class": 2,
         "cost_bbox": 0 if is_o2 else 5,
         "cost_angle": 0 if is_o2 else 2,
         "cost_kld": 2,
         "cost_chamfer": 5 if is_o2 else 0.5},
        chamfer_distance="paper_squared" if is_o2 else "released_l2",
    )
    return RotatedRTv4Criterion(
        matcher, {"loss_focal": 1, "loss_bbox": 5,
                  "loss_angle": 5 if is_o2 else 2,
                  "loss_kld": 2, "loss_fgl": 0.15},
        num_classes=3, reg_max=8)


class ModelPipelineTest(unittest.TestCase):
    def test_forward_loss_backward_and_empty_target(self):
        model = build_tiny_model()
        features = [torch.randn(2, 32, 8, 8, requires_grad=True),
                    torch.randn(2, 32, 4, 4, requires_grad=True),
                    torch.randn(2, 32, 2, 2, requires_grad=True)]
        targets = [
            {"labels": torch.tensor([0, 2]),
             "boxes": torch.tensor([[.5, .5, .2, .1, .99], [.2, .3, .1, .05, .2]])},
            {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty((0, 5))},
        ]
        outputs = model(features, targets)
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 20, 5))
        self.assertEqual(tuple(outputs["distribution_names"]), (
            "external_left", "external_top", "external_right", "external_bottom",
            "vertex_epsilon", "vertex_eta"))
        self.assertTrue((outputs["pred_boxes"][..., 2] >= outputs["pred_boxes"][..., 3]).all())
        losses = build_criterion()(outputs, targets)
        total = sum(losses.values())
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(features[0].grad).all())

    def test_denoising_is_strictly_capped_for_crowded_images(self):
        embedding = torch.nn.Embedding(4, 32, padding_idx=3)
        count = 468
        targets = [{"labels": torch.zeros(count, dtype=torch.long),
                    "boxes": torch.rand(count, 5).clamp(.01, .99)}]
        _, boxes, mask, meta = get_rotated_contrastive_denoising_training_group(
            targets, 3, 600, embedding, num_denoising=100)
        self.assertLessEqual(boxes.shape[1], 100)
        self.assertEqual(mask.shape[0], boxes.shape[1] + 600)
        self.assertEqual(len(meta["dn_target_idx"][0]), 50)

    def test_postprocessor_restores_original_coordinates(self):
        original = torch.tensor([100.0, 200.0, 40.0, 20.0, 0.3])
        scale = 0.25
        canvas_box = original.clone()
        canvas_box[:4] *= scale
        canvas_box[:4] /= 1024
        canvas_box[4] /= torch.pi
        outputs = {"pred_logits": torch.tensor([[[10.0]]]),
                   "pred_boxes": canvas_box.reshape(1, 1, 5)}
        metadata = [{"size": torch.tensor([1024, 1024]),
                     "scale_factor": torch.tensor([scale, scale]),
                     "padding": torch.tensor([0, 0, 0, 484])}]
        result = RotatedPostProcessor(
            num_classes=1, num_top_queries=1, score_threshold=0,
            nms_iou_threshold=.5)(outputs, metadata)[0]
        torch.testing.assert_close(result["boxes"][0], original, atol=1e-4, rtol=1e-4)

    def test_postprocessor_has_an_exact_nms_free_control(self):
        outputs = {
            "pred_logits": torch.tensor([[[8.0], [7.0]]]),
            "pred_boxes": torch.tensor([[[0.50, 0.50, 0.4, 0.2, 0.0],
                                           [0.51, 0.50, 0.4, 0.2, 0.0]]]),
        }
        metadata = [{"size": torch.tensor([100, 100]),
                     "scale_factor": torch.ones(2),
                     "padding": torch.zeros(4)}]
        processor = RotatedPostProcessor(
            num_classes=1, num_top_queries=2, score_threshold=0,
            nms_iou_threshold=0.1, max_detections=10)
        with_nms = processor(outputs, metadata)[0]
        without_nms, diagnostics = processor(
            outputs, metadata, return_diagnostics=True, apply_nms=False)
        self.assertEqual(len(with_nms["boxes"]), 1)
        self.assertEqual(len(without_nms[0]["boxes"]), 2)
        self.assertFalse(diagnostics[0]["apply_nms"])
        self.assertEqual(diagnostics[0]["status"].tolist(), [0, 0])
        processor.apply_nms = False
        self.assertEqual(len(processor(outputs, metadata)[0]["boxes"]), 2)
        self.assertEqual(
            len(processor(outputs, metadata, apply_nms=True)[0]["boxes"]), 1)


if __name__ == "__main__":
    unittest.main()
