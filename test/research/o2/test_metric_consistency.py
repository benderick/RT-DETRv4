import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.rtv4.obb.methods.o2.adr import (
    ADR_COMPONENT_NAMES,
    o2_weighting_function,
)
from tools.research.o2.metric_consistency import (
    PROTOCOL,
    render_case,
    render_empirical_mismatch,
    render_theoretical_transition,
    select_cases,
    summarize_records,
    theoretical_transition_data,
)
from tools.research.o2.run_metric_consistency_audit import _collect_batch
from engine.rtv4.rotated_postprocessor import RotatedPostProcessor


STAGE_NAMES = ["prebox", "layer0", "layer1", "layer2", "layer3"]


def _distribution(component, violation):
    vertex = component.startswith("vertex")
    return {
        "expected_residual": 0.0,
        "entropy": 1.5 if vertex else 1.2,
        "variance": 0.50 if vertex else 0.02,
        "peak_probability": 0.45 if vertex else 0.70,
        "endpoint_mass": 0.60 if vertex else 0.02,
    }


def _record(index, image_path, kind):
    violation = kind == "violation"
    control = kind == "control"
    pre_iou = 0.90 if (violation or control) else 0.70
    final_iou = 0.84 if violation else 0.95 if control else 0.74
    layer0_iou = pre_iou - 0.08 if violation else pre_iou + 0.01
    values = [pre_iou, layer0_iou, (layer0_iou + final_iou) / 2,
              final_iou - 0.01, final_iou]
    target = [0.01, -0.01, 0.02, -0.02,
              -0.95 if violation else 0.10,
              -0.90 if violation else 0.08]
    gt = [128.0, 128.0, 72.0, 28.0, 0.01]
    stages = []
    for stage_index, (name, iou) in enumerate(zip(STAGE_NAMES, values)):
        stage = {
            "name": name,
            "box": [128.0 + stage_index, 128.0, 72.0, 28.0,
                    0.012 + stage_index * .001],
            "rotated_iou": iou,
            "center_error_gt_diagonal": .01,
            "angle_error_deg": 1.0,
            "width_relative_error": .02,
            "height_relative_error": .02,
            "target_class_score": .8,
        }
        if stage_index:
            stage.update({
                "initial_anchor_box": [127.0, 128.0, 71.0, 28.0, .012],
                "predicted_residual": [0.0] * 6,
                "raw_orthogonality_error": .02,
                "target_score_before_lqe": .75,
                "target_class_lqe_logit_delta": .1,
                "distributions": {
                    component: _distribution(component, violation)
                    for component in ADR_COMPONENT_NAMES
                },
            })
        stages.append(stage)
    return {
        "record_id": f"synthetic:{index:04d}",
        "image_id": 0,
        "image_name": "synthetic",
        "image_path": str(image_path),
        "query_index": index,
        "gt_index": index,
        "gt_label": 0,
        "difficulty": 0,
        "gt_box": gt,
        "gt_angle_deg": gt[4] * 180 / np.pi,
        "gt_offset_fractions": [.001, .002] if violation else [.3, .4],
        "gt_chart_seam_distance": .001 if violation else .3,
        "target_residual": target,
        "max_vertex_target_abs": max(abs(target[4]), abs(target[5])),
        "target_outside_codebook": False,
        "stages": stages,
    }


def _case_payload(record):
    project = np.asarray(o2_weighting_function(32), dtype=float)
    probabilities = np.zeros((4, 6, 33), dtype=float)
    probabilities[:, :4, 16] = 1.0
    probabilities[:, 4:, 0] = .48
    probabilities[:, 4:, -1] = .48
    probabilities[:, 4:, 16] = .04
    return {
        **record,
        "distribution_codebook": project.tolist(),
        "full_distribution_probabilities": probabilities.tolist(),
    }


class O2MetricConsistencyTest(unittest.TestCase):
    def test_runtime_collector_writes_one_compact_full_validation_record(self):
        class Dataset:
            classes = ("car",)

            @staticmethod
            def get_ground_truth(image_id):
                return {
                    "image_name": "synthetic",
                    "image_path": "/tmp/synthetic.jpg",
                }

        layers, batch, queries, classes, bins = 4, 1, 2, 1, 33
        target_box = torch.tensor([.5, .5, .3, .1, .25])
        pre_boxes = target_box.reshape(1, 1, 5).expand(batch, queries, 5).clone()
        layer_boxes = pre_boxes.unsqueeze(0).expand(layers, -1, -1, -1).clone()
        project = o2_weighting_function(32)
        outputs = {
            "diagnostic_pre_boxes": pre_boxes,
            "diagnostic_pre_logits": torch.full((batch, queries, classes), 2.0),
            "diagnostic_layer_boxes": layer_boxes,
            "diagnostic_layer_logits": torch.full(
                (layers, batch, queries, classes), 2.0),
            "diagnostic_layer_anchors": layer_boxes.clone(),
            "diagnostic_layer_distributions": torch.zeros(
                layers, batch, queries, 6 * bins),
            "diagnostic_layer_adr_residuals": torch.zeros(
                layers, batch, queries, 6),
            "diagnostic_layer_adr_raw_orthogonality_error": torch.zeros(
                layers, batch, queries),
            "diagnostic_layer_class_logits_before_lqe": torch.full(
                (layers, batch, queries, classes), 1.5),
            "diagnostic_layer_lqe_logit_delta": torch.full(
                (layers, batch, queries, classes), .5),
            "diagnostic_distribution_project": project,
            "diagnostic_distribution_names": ADR_COMPONENT_NAMES,
        }
        target = {
            "boxes": target_box.reshape(1, 5),
            "labels": torch.tensor([0]),
            "difficulty": torch.tensor([0]),
            "image_id": torch.tensor([0]),
            "size": torch.tensor([256.0, 256.0]),
            "orig_size": torch.tensor([256.0, 256.0]),
            "scale_factor": torch.tensor([1.0, 1.0]),
            "padding": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        }
        matching = [(torch.tensor([0]), torch.tensor([0]))]
        records, cases = [], {}
        invariants = {
            "ground_truth_count": 0,
            "matched_count": 0,
            "max_fixed_anchor_difference": 0.0,
            "max_model_pixel_riou_difference": 0.0,
        }
        _collect_batch(
            outputs, [target], matching, Dataset(),
            RotatedPostProcessor(
                num_classes=1, num_top_queries=2, apply_nms=False),
            STAGE_NAMES, records, cases, invariants,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(invariants["ground_truth_count"], 1)
        self.assertEqual(invariants["matched_count"], 1)
        self.assertEqual(invariants["max_fixed_anchor_difference"], 0.0)
        self.assertLessEqual(
            invariants["max_model_pixel_riou_difference"], 1e-6)
        self.assertEqual([stage["name"] for stage in records[0]["stages"]], STAGE_NAMES)
        self.assertEqual(
            set(records[0]["stages"][-1]["distributions"]),
            set(ADR_COMPONENT_NAMES),
        )
        self.assertIn("local_control_success", cases)

    def test_pre_registered_summary_supports_synthetic_metric_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            Image.new("RGB", (256, 256), (80, 90, 100)).save(image)
            records = [
                *(_record(index, image, "violation") for index in range(40)),
                *(_record(index + 40, image, "control") for index in range(40)),
                *(_record(index + 80, image, "other") for index in range(10)),
            ]
            summary = summarize_records(records, STAGE_NAMES)
            self.assertEqual(summary["status"], "SUPPORTED")
            self.assertTrue(all(summary["pre_registered_gates"].values()))
            self.assertGreaterEqual(
                summary["geometric_locality_violation"]["count"],
                PROTOCOL["minimum_locality_violation_count"],
            )
            self.assertGreater(
                summary["effect_sizes"]["control_minus_violation_mean_gain"], .05)

            cases = select_cases(records)
            self.assertEqual(set(cases), {
                "locality_failure", "local_control_success", "layer0_failure"})
            self.assertTrue(cases["locality_failure"]["record_id"].startswith("synthetic:"))

    def test_controlled_transition_exposes_coordinate_jump_not_geometry_jump(self):
        data = theoretical_transition_data(samples=4001)
        angles = data["target_physical_angle_deg"]
        below = np.flatnonzero(angles < 0)[-1]
        above = np.flatnonzero(angles > 0)[0]
        epsilon_jump = abs(
            data["epsilon_target_residual"][above]
            - data["epsilon_target_residual"][below])
        iou_jump = abs(data["rotated_iou"][above] - data["rotated_iou"][below])
        self.assertGreater(epsilon_jump, .9)
        self.assertLess(iou_jump, .001)

    def test_three_mechanism_figures_render_from_declared_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = directory / "image.jpg"
            Image.new("RGB", (256, 256), (80, 90, 100)).save(image)
            records = [
                *(_record(index, image, "violation") for index in range(40)),
                *(_record(index + 40, image, "control") for index in range(40)),
            ]
            summary = summarize_records(records, STAGE_NAMES)
            theoretical = directory / "theoretical.png"
            empirical = directory / "empirical.png"
            case = directory / "case.png"
            self.assertTrue(render_theoretical_transition(theoretical))
            self.assertTrue(render_empirical_mismatch(summary, empirical))
            self.assertTrue(render_case(
                _case_payload(records[0]), case, "locality_failure"))
            for path in (theoretical, empirical, case):
                self.assertGreater(path.stat().st_size, 1000)
                self.assertGreater(path.with_suffix(".pdf").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
