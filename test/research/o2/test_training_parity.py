"""Training-semantics regression tests for the paper-derived O^2 baseline."""

import math
import unittest
from pathlib import Path

import torch

from engine.rtv4.rotated_box_ops import (
    aligned_kld_loss,
    pairwise_chamfer_cost,
    pairwise_kld_cost,
)
from engine.rtv4.rotated_criterion import RotatedRTv4Criterion
from engine.rtv4.rotated_denoising import (
    get_rotated_contrastive_denoising_training_group,
)
from engine.rtv4.rotated_matcher import RotatedHungarianMatcher


def _released_ai4rs_kld(
    pred,
    target,
    *,
    sqrt=False,
    fun="log1p",
    tau=1.0,
):
    """Literal tensor translation of ai4rs/MMRotate gaussian_dist_loss.py."""

    def gaussian(boxes):
        angle = boxes[..., 4] * math.pi
        cosine, sine = angle.cos(), angle.sin()
        rotation = torch.stack(
            (cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
        scale = 0.5 * torch.diag_embed(
            boxes[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2))
        covariance = rotation.bmm(scale.square()).bmm(rotation.transpose(1, 2))
        return boxes[..., :2].reshape(-1, 2), covariance

    xy_p, sigma_p = gaussian(pred)
    xy_t, sigma_t = gaussian(target)
    inverse = torch.stack(
        (sigma_p[..., 1, 1], -sigma_p[..., 0, 1],
         -sigma_p[..., 1, 0], sigma_p[..., 0, 0]), dim=-1,
    ).reshape(-1, 2, 2)
    inverse = inverse / sigma_p.det()[:, None, None]
    delta = (xy_p - xy_t).unsqueeze(-1)
    center = 0.5 * delta.transpose(1, 2).bmm(inverse).bmm(delta).reshape(-1)
    shape = 0.5 * inverse.bmm(sigma_t).diagonal(dim1=-2, dim2=-1).sum(-1)
    shape += 0.5 * (sigma_p.det().log() - sigma_t.det().log()) - 1
    distance = center + shape
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()
    if fun == "log1p":
        distance = torch.log1p(distance)
    elif fun == "sqrt":
        distance = distance.clamp(1e-7).sqrt()
    elif fun != "none":
        raise ValueError(fun)
    if tau >= 1:
        distance = 1 - 1 / (tau + distance)
    return distance.reshape(pred.shape[:-1])


class KLDAndChamferParityTest(unittest.TestCase):
    def test_o2_training_components_default_to_released_kld_semantics(self):
        matcher = RotatedHungarianMatcher({"cost_kld": 2})
        criterion = RotatedRTv4Criterion(
            matcher, {}, losses=("boxes",))
        self.assertFalse(matcher.kld_sqrt)
        self.assertEqual((matcher.kld_fun, matcher.kld_tau), ("log1p", 1.0))
        self.assertFalse(criterion.kld_sqrt)
        self.assertEqual((criterion.kld_fun, criterion.kld_tau), ("log1p", 1.0))

    def test_kld_matches_released_ai4rs_for_all_public_knobs(self):
        pred = torch.tensor([
            [0.52, 0.47, 0.31, 0.09, 0.13],
            [0.23, 0.71, 0.08, 0.04, 0.88],
        ], dtype=torch.float64)
        target = torch.tensor([
            [0.50, 0.50, 0.28, 0.11, 0.17],
            [0.20, 0.68, 0.10, 0.03, 0.94],
        ], dtype=torch.float64)
        for sqrt, fun, tau in (
            (False, "log1p", 1.0),
            (True, "log1p", 1.0),
            (False, "none", 0.0),
            (False, "sqrt", 2.0),
        ):
            expected = _released_ai4rs_kld(
                pred, target, sqrt=sqrt, fun=fun, tau=tau)
            actual = aligned_kld_loss(
                pred, target, sqrt=sqrt, fun=fun, tau=tau)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_pairwise_kld_uses_same_configured_semantics(self):
        pred = torch.tensor([
            [.5, .5, .3, .1, .1], [.2, .7, .15, .04, .8]],
            dtype=torch.float64)
        target = torch.tensor([
            [.48, .52, .28, .12, .2], [.3, .6, .1, .05, .9]],
            dtype=torch.float64)
        actual = pairwise_kld_cost(
            pred, target, sqrt=False, fun="log1p", tau=1)
        expected = torch.stack([
            _released_ai4rs_kld(
                box.expand_as(target), target, sqrt=False, fun="log1p", tau=1)
            for box in pred
        ])
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_closed_form_kld_is_finite_for_thin_rotated_boxes(self):
        prediction = torch.tensor(
            [[0.52, 0.47, 0.20, 1e-7, 0.137]], requires_grad=True)
        target = torch.tensor([[0.50, 0.50, 0.18, 0.01, 0.181]])
        loss = aligned_kld_loss(
            prediction, target, sqrt=False, fun="log1p", tau=1.0).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_rotated_criterion_refuses_to_hide_nonfinite_raw_losses(self):
        criterion = RotatedRTv4Criterion(
            RotatedHungarianMatcher({"cost_kld": 2}), {}, losses=("boxes",))
        outputs = {
            "pred_logits": torch.zeros((1, 1, 1)),
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, float("nan"), 0.1, 0.0]]]),
        }
        targets = [{
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]]),
        }]
        with self.assertRaisesRegex(
                FloatingPointError, "non-finite raw losses"):
            criterion(outputs, targets)
        self.assertIn("loss_bbox", criterion.last_nonfinite_losses)

    def test_chamfer_modes_name_the_paper_source_disagreement(self):
        first = torch.tensor([[.5, .5, .2, .1, 0.]])
        second = torch.tensor([[.51, .5, .2, .1, 0.]])
        paper = pairwise_chamfer_cost(
            first, second, distance_mode="paper_squared")
        released = pairwise_chamfer_cost(
            first, second, distance_mode="released_l2")
        self.assertAlmostEqual(float(paper), 2 * .01 ** 2, places=7)
        self.assertAlmostEqual(float(released), 2 * .01, places=6)
        with self.assertRaisesRegex(ValueError, "Unknown Chamfer distance mode"):
            pairwise_chamfer_cost(
                first, second, distance_mode="ambiguous")

        matcher = RotatedHungarianMatcher(
            {"cost_class": 2, "cost_kld": 2, "cost_chamfer": 5},
            chamfer_distance="released_l2",
            kld_sqrt=False,
            kld_fun="log1p",
            kld_tau=1,
        )
        report = matcher(
            {"pred_logits": torch.zeros(1, 1, 2),
             "pred_boxes": first.unsqueeze(0)},
            [{"labels": torch.tensor([0]), "boxes": second}],
            return_costs=True,
        )
        self.assertEqual(report["chamfer_distance"], "released_l2")
        self.assertEqual(report["chamfer_source_alignment"], "released O2-RTDETR source")
        self.assertEqual(report["kld"], {
            "sqrt": False,
            "fun": "log1p",
            "tau": 1.0,
            "source_alignment": "released O2-RTDETR configuration",
        })


class SquareAwareAngleLossTest(unittest.TestCase):
    @staticmethod
    def _criterion(mode="square_aware_soft", threshold=.05):
        return RotatedRTv4Criterion(
            matcher=None,
            weight_dict={},
            losses=("boxes",),
            angle_loss_mode=mode,
            square_anisotropy_threshold=threshold,
        )

    def test_exact_square_quarter_turn_has_zero_angle_loss(self):
        target = torch.tensor([[.5, .5, .2, .2, .1]])
        prediction = target.clone()
        prediction[:, 4] += .5
        adjusted, ordinary, blend = self._criterion()._angle_loss_terms(
            prediction, target)
        torch.testing.assert_close(ordinary, torch.tensor([.5]))
        torch.testing.assert_close(adjusted, torch.zeros(1))
        torch.testing.assert_close(blend, torch.zeros(1))

    def test_near_square_transitions_softly_and_rectangle_is_unchanged(self):
        target = torch.tensor([
            [.5, .5, .21, .20, .1],
            [.5, .5, .40, .20, .1],
        ])
        prediction = target.clone()
        prediction[:, 4] += .5
        adjusted, ordinary, blend = self._criterion()._angle_loss_terms(
            prediction, target)
        expected_blend = (.01 / .41) / .05
        self.assertAlmostEqual(float(blend[0]), expected_blend, places=6)
        self.assertAlmostEqual(float(adjusted[0]), .5 * expected_blend, places=6)
        self.assertAlmostEqual(float(ordinary[0]), .5, places=6)
        self.assertEqual(float(blend[1]), 1.0)
        self.assertEqual(float(adjusted[1]), float(ordinary[1]))

    def test_paper_baseline_mode_remains_explicitly_available(self):
        target = torch.tensor([[.5, .5, .2, .2, .1]])
        prediction = target.clone()
        prediction[:, 4] += .5
        adjusted, ordinary, blend = self._criterion(
            "periodic_pi")._angle_loss_terms(prediction, target)
        torch.testing.assert_close(adjusted, ordinary)
        torch.testing.assert_close(blend, torch.ones(1))

    def test_diagnostics_expose_the_remaining_adr_axis_chart_seam(self):
        criterion = self._criterion()
        target_boxes = torch.tensor([
            [.5, .5, .30, .10, 1e-6 / math.pi],
            [.5, .5, .30, .10, .23],
        ])
        outputs = {
            "pred_logits": torch.tensor([[[8., -8.], [8., -8.]]]),
            "pred_boxes": target_boxes.unsqueeze(0),
            "ref_points": target_boxes.unsqueeze(0),
            "adr_project": torch.linspace(-1., 1., 33),
        }
        targets = [{"labels": torch.tensor([0, 0]), "boxes": target_boxes}]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        diagnostics = criterion._main_match_diagnostics(outputs, targets, indices)
        seam = diagnostics["adr_chart_seam"]
        self.assertGreaterEqual(seam["near_endpoint_count_0p001"], 1)
        self.assertIn("identity can switch", seam["known_limitation"])
        self.assertEqual(seam["matched_count"], 2)


class DfineUnionAndCrowdedOCDTest(unittest.TestCase):
    def test_union_keeps_distinct_pairs_and_one_target_per_query(self):
        main = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        aux = [(torch.tensor([2, 1]), torch.tensor([0, 1]))]
        conflicting = [(torch.tensor([2]), torch.tensor([1]))]
        source, target = RotatedRTv4Criterion._get_union_indices(
            main, [aux, conflicting]) [0]
        pairs = set(zip(source.tolist(), target.tolist()))
        self.assertEqual(pairs, {(0, 0), (1, 1), (2, 0)})

    def test_forward_uses_union_only_for_geometry(self):
        class LayerMatcher:
            def __call__(self, outputs, targets, **kwargs):
                if outputs["match_layer"] == "main":
                    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
                else:
                    indices = [(torch.tensor([2, 1]), torch.tensor([0, 1]))]
                return {"indices": indices}

        target_boxes = torch.tensor([
            [.2, .3, .2, .1, .1],
            [.7, .6, .3, .1, .2],
        ])
        main_boxes = torch.stack((
            target_boxes[0],
            target_boxes[1],
            target_boxes[0] + torch.tensor([.09, 0, 0, 0, 0]),
        )).unsqueeze(0)
        auxiliary = {
            "match_layer": "aux",
            "pred_boxes": main_boxes.clone(),
            "pred_logits": torch.zeros(1, 3, 2),
        }
        outputs = {
            "match_layer": "main",
            "pred_boxes": main_boxes,
            "pred_logits": torch.zeros(1, 3, 2),
            "aux_outputs": [auxiliary],
            "up": torch.tensor(1.),
            "reg_scale": torch.tensor(1.),
        }
        targets = [{"labels": torch.tensor([0, 1]), "boxes": target_boxes}]
        union = RotatedRTv4Criterion(
            LayerMatcher(), {}, losses=("boxes",), use_uni_set=True)
        layer_only = RotatedRTv4Criterion(
            LayerMatcher(), {}, losses=("boxes",), use_uni_set=False)
        union_losses = union(outputs, targets)
        layer_losses = layer_only(outputs, targets)
        self.assertAlmostEqual(float(union_losses["loss_bbox"]), .03, places=6)
        self.assertEqual(float(layer_losses["loss_bbox"]), 0.)

    def test_released_dynamic_exceeds_budget_instead_of_dropping_gt(self):
        boxes = torch.tensor([
            [.1 + .1 * index, .5, .05, .03, .1]
            for index in range(7)
        ])
        targets = [{"labels": torch.arange(7) % 2, "boxes": boxes}]
        embedding = torch.nn.Embedding(3, 8, padding_idx=2)
        _, _, _, meta = get_rotated_contrastive_denoising_training_group(
            targets,
            num_classes=2,
            num_queries=20,
            class_embed=embedding,
            num_denoising=10,
            box_noise_scale=0,
            mode="none",
            crowded_policy="released_dynamic",
        )
        self.assertEqual(meta["dn_actual_query_count"], 14)
        self.assertTrue(meta["dn_budget_exceeded"])
        self.assertEqual(meta["dn_selected_gt_counts"], [7])
        self.assertEqual(meta["dn_dropped_gt_counts"], [0])
        self.assertEqual(len(meta["dn_dropped_target_idx"][0]), 0)
        torch.testing.assert_close(meta["dn_target_idx"][0], torch.arange(7))

    def test_strict_budget_drop_is_explicit_and_auditable(self):
        torch.manual_seed(3)
        boxes = torch.tensor([
            [.1 + .1 * index, .5, .05, .03, .1]
            for index in range(7)
        ])
        targets = [{"labels": torch.arange(7) % 2, "boxes": boxes}]
        embedding = torch.nn.Embedding(3, 8, padding_idx=2)
        _, _, _, meta = get_rotated_contrastive_denoising_training_group(
            targets,
            num_classes=2,
            num_queries=20,
            class_embed=embedding,
            num_denoising=10,
            box_noise_scale=0,
            mode="none",
            crowded_policy="strict_budget_random",
        )
        self.assertEqual(meta["dn_actual_query_count"], 10)
        self.assertFalse(meta["dn_budget_exceeded"])
        self.assertEqual(meta["dn_selected_gt_counts"], [5])
        self.assertEqual(meta["dn_dropped_gt_counts"], [2])
        self.assertEqual(len(meta["dn_target_idx"][0]), 5)
        self.assertEqual(len(meta["dn_selected_target_idx"][0]), 5)
        self.assertEqual(len(meta["dn_dropped_target_idx"][0]), 2)
        self.assertEqual(
            set(meta["dn_selected_target_idx"][0].tolist()) |
            set(meta["dn_dropped_target_idx"][0].tolist()),
            set(range(7)),
        )

    def test_primary_config_records_every_non_equivalent_choice(self):
        text = (Path(__file__).resolve().parents[3] /
                "configs/dfine/dfine_obb_o2.yml").read_text()
        for declaration in (
            "ocd_crowded_policy: released_dynamic",
            "kld_sqrt: False",
            "kld_fun: log1p",
            "kld_tau: 1.0",
            "chamfer_distance: paper_squared",
            "angle_loss_mode: periodic_pi",
            "square_anisotropy_threshold: 0.0",
        ):
            self.assertIn(declaration, text)
        self.assertNotIn("amp_abort_min_scale", text)
        self.assertNotIn("angle_loss_mode: square_aware_soft", text)


if __name__ == "__main__":
    unittest.main()
