import math
import unittest
from pathlib import Path

import torch

from engine.core import YAMLConfig
from engine.rtv4 import (
    RotatedDFINETransformer,
    RotatedHungarianMatcher,
    RotatedRTv4Criterion,
)
from engine.rtv4.dfine_decoder import rotate_sampling_offsets
from engine.rtv4.obb.methods.o2.adr import (
    adr_target_residual,
    adr_to_rbox,
    distribution_integral,
    o2_weighting_function,
    translate_with_project,
)
from engine.rtv4.rotated_box_ops import pairwise_chamfer_cost, rotated_iou
from engine.rtv4.rotated_denoising import (
    apply_ocd_angle_noise,
    apply_ocd_box_noise,
    apply_ocd_probability_noise,
    get_rotated_contrastive_denoising_training_group,
)


def _tiny_o2_model(num_denoising=10):
    return RotatedDFINETransformer(
        num_classes=3, hidden_dim=32, num_queries=20,
        feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
        num_points=[2, 2, 2], nhead=4, num_layers=2, dim_feedforward=64,
        num_denoising=num_denoising, reg_max=8, aux_loss=True,
        refinement_mode="o2_adr", ocd_mode="box",
    )


class O2ADRTest(unittest.TestCase):
    def test_paper_weighting_function_and_unbiased_bin_translation(self):
        project = o2_weighting_function(32, a=0.5, c=0.25)
        self.assertEqual(len(project), 33)
        self.assertAlmostEqual(float(project[0]), -1.0, places=7)
        self.assertAlmostEqual(float(project[16]), 0.0, places=7)
        self.assertAlmostEqual(float(project[-1]), 1.0, places=7)
        self.assertTrue((project[1:] > project[:-1]).all())

        targets = torch.tensor([-1.2, -0.13, 0.0, 0.37, 1.2])
        left, weight_right, weight_left = translate_with_project(targets, project)
        recovered = project[left] * weight_left + project[left + 1] * weight_right
        torch.testing.assert_close(
            recovered, targets.clamp(project[0], project[-1]), atol=1e-6, rtol=1e-6)

    def test_distribution_integral_has_no_hidden_dtype_promotion(self):
        logits = torch.zeros((2, 6 * 9), dtype=torch.float16)
        project = o2_weighting_function(8)
        residual = distribution_integral(logits, project)
        self.assertEqual(residual.dtype, logits.dtype)
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-4, rtol=0)

    def test_adr_target_residual_reconstructs_nearby_target(self):
        reference = torch.tensor([
            [0.50, 0.50, 0.30, 0.12, 0.13],
            [0.30, 0.40, 0.18, 0.08, 0.92],
        ])
        target = torch.tensor([
            [0.51, 0.49, 0.29, 0.11, 0.15],
            [0.29, 0.41, 0.17, 0.07, 0.90],
        ])
        residual = adr_target_residual(reference, target)
        recovered = adr_to_rbox(reference, residual)
        overlap = rotated_iou(recovered, target, aligned=True)
        torch.testing.assert_close(overlap, torch.ones_like(overlap), atol=1e-4, rtol=1e-4)

    def test_adr_model_forward_loss_backward_and_instability_log(self):
        model = _tiny_o2_model()
        features = [
            torch.randn(2, 32, 8, 8, requires_grad=True),
            torch.randn(2, 32, 4, 4, requires_grad=True),
            torch.randn(2, 32, 2, 2, requires_grad=True),
        ]
        targets = [
            {"labels": torch.tensor([0, 2]),
             "boxes": torch.tensor([
                 [.5, .5, .2, .1, .99], [.2, .3, .1, .05, .2]])},
            {"labels": torch.empty(0, dtype=torch.long),
             "boxes": torch.empty((0, 5))},
        ]
        outputs = model(features, targets)
        self.assertEqual(tuple(outputs["pred_corners"].shape), (2, 20, 6 * 9))
        self.assertEqual(tuple(outputs["distribution_names"]), (
            "external_left", "external_top", "external_right", "external_bottom",
            "vertex_epsilon", "vertex_eta"))
        self.assertFalse(any(parameter.numel() for head in model.dec_angle_head
                             for parameter in head.parameters()))

        matcher = RotatedHungarianMatcher({
            "cost_class": 2, "cost_bbox": 0, "cost_angle": 0,
            "cost_kld": 2, "cost_chamfer": 5,
        }, chamfer_distance="paper_squared")
        criterion = RotatedRTv4Criterion(
            matcher,
            {"loss_focal": 1, "loss_bbox": 5, "loss_angle": 5,
             "loss_kld": 2, "loss_fgl": 0.15},
            num_classes=3, reg_max=8,
        )
        losses = criterion(outputs, targets, collect_diagnostics=True)
        total = sum(losses.values())
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(features[0].grad).all())
        instability = criterion.last_diagnostics["assignment_instability"]
        self.assertEqual(instability["decoder_layer_count"], 2)
        self.assertEqual(instability["ground_truth_count"], 2)

    def test_exact_instability_fraction(self):
        targets = [{"labels": torch.tensor([0, 0, 0])}]
        layers = [
            [(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))],
            [(torch.tensor([0, 3, 2]), torch.tensor([0, 1, 2]))],
        ]
        value = RotatedRTv4Criterion._assignment_instability(layers, targets)
        self.assertEqual(value["changed_count"], 1)
        self.assertAlmostEqual(value["instability"], 1 / 3)


class O2MatchingAndDenoisingTest(unittest.TestCase):
    def test_squared_chamfer_matches_paper_definition(self):
        first = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        second = torch.tensor([[0.51, 0.5, 0.2, 0.1, 0.0]])
        squared = pairwise_chamfer_cost(
            first, second, distance_mode="paper_squared")
        self.assertAlmostEqual(float(squared), 2 * 0.01 ** 2, places=7)
        ordinary = pairwise_chamfer_cost(
            first, second, distance_mode="released_l2")
        self.assertAlmostEqual(float(ordinary), 2 * 0.01, places=6)

    def test_ocd_box_positive_and_negative_ranges(self):
        torch.manual_seed(7)
        box = torch.tensor([[[0.5, 0.5, 0.4, 0.2, 0.2],
                             [0.5, 0.5, 0.4, 0.2, 0.2]]])
        negative = torch.tensor([[[0.0], [1.0]]])
        noised = apply_ocd_box_noise(box, negative, lambda1=0.1, lambda2=0.2)
        original_xyxy = torch.cat((box[..., :2] - box[..., 2:4] / 2,
                                   box[..., :2] + box[..., 2:4] / 2), dim=-1)
        noised_xyxy = torch.cat((noised[..., :2] - noised[..., 2:4] / 2,
                                 noised[..., :2] + noised[..., 2:4] / 2), dim=-1)
        scale = torch.cat((box[..., 2:4], box[..., 2:4]), dim=-1) / 2
        normalized_delta = (noised_xyxy - original_xyxy).abs() / scale
        self.assertTrue((normalized_delta[:, 0] <= 0.1 + 1e-6).all())
        self.assertTrue((normalized_delta[:, 1] >= 0.1 - 1e-6).all())
        self.assertTrue((normalized_delta[:, 1] <= 0.2 + 1e-6).all())
        torch.testing.assert_close(noised[..., 4], box[..., 4])

    def test_all_ocd_modes_keep_their_declared_invariants(self):
        box = torch.tensor([[[0.5, 0.5, 0.4, 0.2, 0.0],
                             [0.5, 0.5, 0.4, 0.2, 0.25]]])
        negative = torch.tensor([[[0.0], [1.0]]])
        angle_noised = apply_ocd_angle_noise(box, negative)
        # This records the paper's known zero-angle degeneracy exactly.
        self.assertEqual(float(angle_noised[0, 0, 4]), 0.0)
        torch.testing.assert_close(angle_noised[..., :4], box[..., :4])
        probability_noised = apply_ocd_probability_noise(box, negative)
        torch.testing.assert_close(probability_noised[..., :2], box[..., :2])

        embedding = torch.nn.Embedding(4, 16, padding_idx=3)
        target = [{"labels": torch.tensor([0, 1]), "boxes": box[0]}]
        _, query_boxes, _, meta = get_rotated_contrastive_denoising_training_group(
            target, 3, 20, embedding, num_denoising=10, mode="geometric")
        self.assertLessEqual(query_boxes.shape[1], 10)
        self.assertEqual(meta["dn_noise_mode"], "geometric")


class O2RotatedAttentionTest(unittest.TestCase):
    def test_sampling_offset_rotates_with_reference_angle(self):
        offset = torch.tensor([[[[[1.0, 0.0]]]]])
        quarter_turn = torch.tensor([[[[[0.5]]]]])
        rotated = rotate_sampling_offsets(offset, quarter_turn)
        expected = torch.tensor([[[[[0.0, 1.0]]]]])
        torch.testing.assert_close(rotated, expected, atol=1e-6, rtol=1e-6)

    def test_diagnostic_mode_exposes_sampling_evidence(self):
        model = _tiny_o2_model(num_denoising=0).eval().set_diagnostic_mode(True)
        features = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4),
                    torch.randn(1, 32, 2, 2)]
        outputs = model(features)
        self.assertEqual(outputs["diagnostic_pre_boxes"].shape, (1, 20, 5))
        self.assertEqual(outputs["diagnostic_pre_logits"].shape, (1, 20, 3))
        self.assertEqual(outputs["diagnostic_sampling_locations"].shape[:3], (2, 1, 20))
        self.assertEqual(outputs["diagnostic_sampling_rotated_offsets"].shape[-1], 2)
        self.assertEqual(tuple(outputs["diagnostic_sampling_points_per_level"]), (2, 2, 2))


class O2ConfigurationTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[3]

    def test_four_established_public_variants_have_exact_semantics(self):
        expected = (
            ("dfine_obb_angle.yml", "direct_angle", "standard", 100, 0.5,
             "released_l2", True,
             "DotaOBBEvaluator"),
            ("dfine_obb_o2.yml", "o2_adr", "box", 200, 5.0,
             "paper_squared", False,
             "DotaOBBEvaluator"),
            ("dfine_obb_angle_tile.yml", "direct_angle", "standard", 100, 0.5,
             "released_l2", True,
             "MergedDotaOBBEvaluator"),
            ("dfine_obb_o2_tile.yml", "o2_adr", "box", 200, 5.0,
             "paper_squared", True,
             "MergedDotaOBBEvaluator"),
        )
        config_dir = self.ROOT / "configs" / "dfine"
        self.assertEqual(
            {path.name for path in config_dir.glob("*obb*.yml")
             if "stage" not in path.name},
            {item[0] for item in expected},
        )
        for (name, refinement_mode, mode, count, chamfer, distance,
             apply_nms, evaluator) in expected:
            self.assertNotIn("codrone", name)
            self.assertNotIn("hgnet", name)
            self.assertNotIn("b0", name.lower())
            config = YAMLConfig(str(self.ROOT / "configs" / "dfine" / name))
            decoder = config.yaml_cfg["RotatedDFINETransformer"]
            matcher = config.yaml_cfg["RotatedRTv4Criterion"]["matcher"]
            self.assertEqual(decoder["refinement_mode"], refinement_mode)
            self.assertNotIn("use_adr", decoder)
            self.assertEqual(decoder["ocd_mode"], mode)
            self.assertEqual(decoder["num_denoising"], count)
            self.assertEqual(matcher["weight_dict"]["cost_chamfer"], chamfer)
            self.assertEqual(matcher["chamfer_distance"], distance)
            self.assertEqual(
                config.yaml_cfg["RotatedPostProcessor"].get("apply_nms", True),
                apply_nms)
            self.assertEqual(config.postprocessor.apply_nms, apply_nms)
            self.assertEqual(config.yaml_cfg["evaluator"]["type"], evaluator)
            criterion = config.yaml_cfg["RotatedRTv4Criterion"]
            if refinement_mode == "o2_adr":
                self.assertEqual(criterion["angle_loss_mode"], "periodic_pi")
                self.assertEqual(criterion["square_anisotropy_threshold"], 0.0)
                self.assertNotIn("amp_abort_min_scale", config.yaml_cfg)

    def test_direct_angle_variant_has_a_real_scalar_angle_head(self):
        model = RotatedDFINETransformer(
            num_classes=3, hidden_dim=32, num_queries=20,
            feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
            num_points=[2, 2, 2], nhead=4, num_layers=2, dim_feedforward=64,
            num_denoising=0, reg_max=8, refinement_mode="direct_angle",
            ocd_mode="standard",
        )
        self.assertFalse(model.use_adr)
        self.assertTrue(all(head.layers[-1].out_features == 1
                            for head in model.dec_angle_head))
        self.assertEqual(model.dec_bbox_head[0].layers[-1].out_features, 4 * 9)

if __name__ == "__main__":
    unittest.main()
