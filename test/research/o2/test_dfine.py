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
from engine.rtv4.dfine_decoder import MSDeformableAttention, rotate_sampling_offsets
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
from tools.research.o2.run_acceptance import (
    O2_DFINE_M_PARAMETER_RANGE,
    _decoder_contract_differences,
    _source_aligned_config_contract,
)


def _tiny_o2_model(num_denoising=10):
    return RotatedDFINETransformer(
        num_classes=3, hidden_dim=32, num_queries=20,
        feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
        num_points=[2, 2, 2], nhead=4, num_layers=2, dim_feedforward=64,
        num_denoising=num_denoising, reg_max=8, aux_loss=True,
        refinement_mode="o2_adr", ocd_mode="box",
    )


def _assert_nested_exact(test, first, second):
    test.assertEqual(type(first), type(second))
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            _assert_nested_exact(test, first[key], second[key])
    elif isinstance(first, (tuple, list)):
        test.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            _assert_nested_exact(test, left, right)
    else:
        test.assertEqual(first, second)


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

    def test_zero_adr_residual_reconstructs_random_reference_geometry(self):
        torch.manual_seed(23)
        references = torch.cat((
            torch.rand(256, 2) * .8 + .1,
            torch.rand(256, 2) * .35 + .005,
            torch.rand(256, 1),
        ), dim=-1)
        residual = adr_target_residual(references, references)
        recovered = adr_to_rbox(references, residual)
        overlaps = rotated_iou(
            recovered, references, aligned=True, model_space=True)
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-7, rtol=0)
        torch.testing.assert_close(
            overlaps, torch.ones_like(overlaps), atol=1e-5, rtol=0)

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
        })
        criterion = RotatedRTv4Criterion(
            matcher,
            {"loss_vfl": 1, "loss_bbox": 5, "loss_angle": 5,
             "loss_kld": 2, "loss_fgl": 0.15},
            losses=("vfl", "boxes", "local"),
            alpha=.75, num_classes=3, reg_max=8,
        )
        losses = criterion(outputs, targets, collect_diagnostics=True)
        total = sum(losses.values())
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(features[0].grad).all())
        instability = criterion.last_diagnostics["assignment_instability"]
        self.assertEqual(instability["decoder_layer_count"], 2)
        self.assertEqual(instability["ground_truth_count"], 2)

    def test_decoder_layers_accumulate_six_distribution_logits(self):
        model = _tiny_o2_model(num_denoising=0).eval().set_diagnostic_mode(True)
        first = torch.linspace(
            -0.3, 0.4, model.dec_bbox_head[0].layers[-1].out_features)
        second = torch.linspace(
            0.2, -0.1, model.dec_bbox_head[1].layers[-1].out_features)
        with torch.no_grad():
            model.dec_bbox_head[0].layers[-1].bias.copy_(first)
            model.dec_bbox_head[1].layers[-1].bias.copy_(second)
        features = [
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
        ]
        outputs = model(features)
        logits = outputs["diagnostic_layer_distributions"]
        torch.testing.assert_close(logits[0], first.expand_as(logits[0]))
        torch.testing.assert_close(
            logits[1], (first + second).expand_as(logits[1]))
        self.assertEqual(
            outputs["diagnostic_adr_geometry_contract"],
            "dfine4_plus_vertex2_equal_diagonal",
        )
        self.assertEqual(
            outputs["diagnostic_layer_adr_residuals"].shape[-1], 6)
        self.assertEqual(
            outputs["diagnostic_layer_adr_values"].shape[-1], 6)
        torch.testing.assert_close(
            outputs["diagnostic_layer_input_refs"][1:],
            outputs["diagnostic_layer_boxes"][:-1], atol=0, rtol=0)
        torch.testing.assert_close(
            outputs["diagnostic_layer_anchors"],
            outputs["diagnostic_pre_boxes"].unsqueeze(0).expand_as(
                outputs["diagnostic_layer_anchors"]),
            atol=0, rtol=0,
        )
        torch.testing.assert_close(
            outputs["diagnostic_layer_logits"] -
            outputs["diagnostic_layer_class_logits_before_lqe"],
            outputs["diagnostic_layer_lqe_logit_delta"],
        )

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
    def test_default_denoising_strategy_seam_is_bit_exact_o2(self):
        model = _tiny_o2_model()
        targets = [
            {"labels": torch.tensor([0, 2]), "boxes": torch.tensor([
                [.50, .50, .20, .06, .01],
                [.23, .31, .12, .05, .67],
            ])},
            {"labels": torch.tensor([1]), "boxes": torch.tensor([
                [.71, .62, .18, .07, .24],
            ])},
        ]
        arguments = dict(
            targets=targets, num_classes=model.num_classes,
            num_queries=model.num_queries,
            class_embed=model.denoising_class_embed,
            num_denoising=model.num_denoising,
            label_noise_ratio=model.label_noise_ratio,
            box_noise_scale=model.box_noise_scale, mode=model.ocd_mode,
            lambda1=model.ocd_lambdas[0], lambda2=model.ocd_lambdas[1],
            lambda3=model.ocd_lambdas[2], lambda4=model.ocd_lambdas[3],
            lambda5=model.ocd_lambdas[4], lambda6=model.ocd_lambdas[5],
            crowded_policy=model.ocd_crowded_policy,
        )
        torch.manual_seed(101)
        expected = get_rotated_contrastive_denoising_training_group(**arguments)
        torch.manual_seed(101)
        observed = model._build_denoising_group(targets)
        _assert_nested_exact(self, observed, expected)

    def test_chamfer_matches_the_released_o2_definition(self):
        first = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        second = torch.tensor([[0.51, 0.5, 0.2, 0.1, 0.0]])
        distance = pairwise_chamfer_cost(first, second)
        self.assertAlmostEqual(float(distance), 2 * 0.01, places=6)

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
        self.assertEqual(query_boxes.shape[1], 20)
        self.assertEqual(meta["dn_noise_mode"], "geometric")
        self.assertEqual(meta["dn_group_base_count"], 10)

    def test_ocd_box_noise_keeps_numeric_angle_through_query_generation(self):
        torch.manual_seed(11)
        boxes = torch.tensor([
            [.5, .5, .4, .2, .0],
            [.5, .5, .4, .2, .25],
        ])
        embedding = torch.nn.Embedding(4, 16, padding_idx=3)
        _, query_boxes, _, meta = get_rotated_contrastive_denoising_training_group(
            [{"labels": torch.tensor([0, 1]), "boxes": boxes}],
            3, 20, embedding, num_denoising=10, mode="box")
        observed = query_boxes.sigmoid()[0, :, 4]
        expected_group = boxes[:, 4].repeat(2)
        torch.testing.assert_close(observed, expected_group.repeat(meta["dn_num_group"]))


class O2RotatedAttentionTest(unittest.TestCase):
    def test_sampling_offset_rotates_with_reference_angle(self):
        offset = torch.tensor([[[[[1.0, 0.0]]]]])
        quarter_turn = torch.tensor([[[[[0.5]]]]])
        rotated = rotate_sampling_offsets(offset, quarter_turn)
        expected = torch.tensor([[[[[0.0, 1.0]]]]])
        torch.testing.assert_close(rotated, expected, atol=1e-6, rtol=1e-6)

        arbitrary = torch.tensor([[[[[2.0, -3.0]]]]])
        rotated_arbitrary = rotate_sampling_offsets(arbitrary, quarter_turn)
        torch.testing.assert_close(
            rotated_arbitrary, torch.tensor([[[[[3.0, 2.0]]]]]),
            atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            torch.linalg.vector_norm(rotated_arbitrary, dim=-1),
            torch.linalg.vector_norm(arbitrary, dim=-1),
        )

    def test_dfine_scales_in_local_box_axes_then_rotates_the_offset_vector(self):
        attention = MSDeformableAttention(
            embed_dim=8, num_heads=1, num_levels=1, num_points=1)
        attention.diagnostic_mode = True
        with torch.no_grad():
            attention.sampling_offsets.weight.zero_()
            attention.sampling_offsets.bias.copy_(torch.tensor([1.0, 0.0]))
        query = torch.zeros(1, 1, 8)
        reference = torch.tensor([[[[.5, .5, .4, .2, .5]]]])
        value = (torch.zeros(1, 1, 8, 1),)
        attention(query, reference, value, [[1, 1]])
        # Default offset_scale=.5: local (1,0) first becomes (.2,0)
        # from width=.4, then theta=pi/2 rotates it to (0,.2).
        torch.testing.assert_close(
            attention.last_unrotated_offsets,
            torch.tensor([[[[[.2, 0.]]]]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            attention.last_rotated_offsets,
            torch.tensor([[[[[0., .2]]]]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            attention.last_sampling_locations,
            torch.tensor([[[[[.5, .7]]]]]), atol=1e-6, rtol=0)

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

    def test_layer_outputs_do_not_require_expensive_attention_traces(self):
        model = _tiny_o2_model(num_denoising=0).eval().set_diagnostic_mode(
            True, capture_attention=False)
        features = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4),
                    torch.randn(1, 32, 2, 2)]
        outputs = model(features)
        self.assertIn("diagnostic_layer_boxes", outputs)
        self.assertIn("diagnostic_layer_input_refs", outputs)
        self.assertNotIn("diagnostic_sampling_locations", outputs)


class O2ConfigurationTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[3]

    def test_four_established_public_variants_have_exact_semantics(self):
        expected = (
            ("dfine_obb_angle.yml", "direct_angle", "standard", 100, 0.5,
             False, "DotaOBBEvaluator"),
            ("dfine_obb_o2.yml", "o2_adr", "box", 100, 5.0,
             False, "DotaOBBEvaluator"),
            ("dfine_obb_angle_tile.yml", "direct_angle", "standard", 100, 0.5,
             True, "MergedDotaOBBEvaluator"),
            ("dfine_obb_o2_tile.yml", "o2_adr", "box", 100, 5.0,
             True, "MergedDotaOBBEvaluator"),
        )
        config_dir = self.ROOT / "configs" / "dfine"
        common_base = self.ROOT / "configs" / "base" / "dfine_obb.yml"
        self.assertTrue(common_base.is_file())
        self.assertNotIn(
            "dfine_obb_angle.yml",
            (config_dir / "dfine_obb_o2.yml").read_text(encoding="utf-8"),
        )
        self.assertEqual(
            {path.name for path in config_dir.glob("*obb*.yml")
             if "stage" not in path.name},
            {item[0] for item in expected},
        )
        for (name, refinement_mode, mode, count, chamfer,
             apply_nms, evaluator) in expected:
            self.assertNotIn("codrone", name)
            self.assertNotIn("hgnet", name)
            self.assertNotIn("b0", name.lower())
            config = YAMLConfig(str(self.ROOT / "configs" / "dfine" / name))
            decoder = config.yaml_cfg["RotatedDFINETransformer"]
            matcher = config.yaml_cfg["RotatedRTv4Criterion"]["matcher"]
            optimizer = config.yaml_cfg["optimizer"]
            self.assertEqual(decoder["refinement_mode"], refinement_mode)
            self.assertNotIn("use_adr", decoder)
            self.assertEqual(decoder["num_queries"], 300)
            self.assertEqual(decoder["num_layers"], 4)
            self.assertTrue(decoder.get("aux_loss", True))
            self.assertEqual(decoder["ocd_mode"], mode)
            self.assertEqual(decoder["num_denoising"], count)
            self.assertEqual(config.yaml_cfg["HGNetv2"]["name"], "B2")
            self.assertEqual(config.yaml_cfg["HybridEncoder"]["expansion"], 1.0)
            self.assertEqual(config.yaml_cfg["HybridEncoder"]["depth_mult"], .67)
            self.assertEqual(config.yaml_cfg["num_top_queries"], 300)
            self.assertEqual(config.yaml_cfg["epoches"], 72)
            self.assertEqual(
                config.yaml_cfg["train_dataloader"]["total_batch_size"], 8)
            self.assertEqual(optimizer["lr"], 5e-5)
            self.assertEqual(optimizer["params"][0]["lr"], 5e-6)
            self.assertEqual(optimizer["weight_decay"], 1e-4)
            self.assertEqual(
                config.yaml_cfg["lr_scheduler"]["milestones"], [500])
            self.assertEqual(matcher["weight_dict"]["cost_chamfer"], chamfer)
            self.assertNotIn("chamfer_distance", matcher)
            self.assertEqual(
                config.yaml_cfg["RotatedPostProcessor"].get("apply_nms", True),
                apply_nms)
            self.assertEqual(config.postprocessor.apply_nms, apply_nms)
            self.assertEqual(config.yaml_cfg["evaluator"]["type"], evaluator)
            transform_types = tuple(
                operation["type"]
                for operation in config.yaml_cfg["train_dataloader"]
                ["dataset"]["transforms"]["ops"])
            self.assertEqual(transform_types, (
                "RotatedResize",
                "RotatedRandomFlip",
                "RotatedRandomRotate",
                "RotatedSanitizeBoxes",
                "RotatedPad",
                "RotatedConvertToTensor",
            ))
            criterion = config.yaml_cfg["RotatedRTv4Criterion"]
            self.assertEqual(criterion["losses"], ["vfl", "boxes", "local"])
            self.assertEqual(criterion["alpha"], .75)
            self.assertEqual(criterion["weight_dict"]["loss_vfl"], 1.0)
            if refinement_mode == "o2_adr":
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

    def test_uav_recipe_satisfies_the_runtime_training_contract(self):
        config = YAMLConfig(str(
            self.ROOT / "configs" / "experiments" / "uav_rod" /
            "dfine_obb_o2.yml"))
        contract = _source_aligned_config_contract(config.yaml_cfg)
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["differences"], {})
        self.assertTrue(contract["actual"]["auxiliary_training"])

        parameter_count = sum(
            parameter.numel() for parameter in config.model.parameters())
        minimum_parameters, maximum_parameters = O2_DFINE_M_PARAMETER_RANGE
        self.assertGreaterEqual(parameter_count, minimum_parameters)
        self.assertLess(parameter_count, maximum_parameters)

    def test_real_forward_tensors_satisfy_decoder_acceptance_contract(self):
        model = _tiny_o2_model(num_denoising=0).eval().set_diagnostic_mode(
            True, capture_attention=False)
        features = [
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
        ]
        with torch.inference_mode():
            outputs = model(features)
        differences = _decoder_contract_differences(outputs)
        self.assertEqual(differences, {
            "max_fixed_anchor_difference": 0.0,
            "max_reference_chain_difference": 0.0,
            "max_adr_reconstruction_corner_difference": 0.0,
            "max_lqe_identity_difference": 0.0,
        })

if __name__ == "__main__":
    unittest.main()
