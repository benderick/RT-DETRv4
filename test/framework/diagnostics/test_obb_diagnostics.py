import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from engine.diagnostics import OBBDiagnostics
from engine.rtv4 import RotatedHungarianMatcher, RotatedPostProcessor
from test.framework.model.test_model_pipeline import build_tiny_model


class _Dataset:
    classes = ("car",)
    root = Path("/synthetic/codrone")

    def __len__(self):
        return 1

    def get_ground_truth(self, image_id):
        return {"image_name": "park_day_60m_30c_frame_10", "image_path": None}

    def get_image_metadata(self, image_id):
        return {
            "illumination": "day", "altitude_m": 60.0,
            "view_angle_deg": 30.0, "frame_index": 10,
        }


def _target():
    return {
        "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.0]]),
        "labels": torch.tensor([0]),
        "difficulty": torch.tensor([0]),
        "image_id": torch.tensor([0]),
        "orig_size": torch.tensor([100, 100]),
        "size": torch.tensor([100, 100]),
        "scale_factor": torch.tensor([1.0, 1.0]),
        "padding": torch.tensor([0, 0, 0, 0]),
        "partition_id": "codrone_dota_w100_g20_iof0p7",
        "tile_id": "park__100__80___0",
        "tile_origin": torch.tensor([80, 0]),
        "tile_size": torch.tensor([100, 100]),
        "tile_overlap": torch.tensor([20, 20]),
        "tile_step": torch.tensor([80, 80]),
        "source_image_id": "park_day_60m_30c_frame_10",
        "source_image_size": torch.tensor([180, 100]),
        "source_object_index": torch.tensor([17]),
        "visible_ratio": torch.tensor([0.75]),
        "source_tile_count": torch.tensor([2]),
        "boundary_distance_px": torch.tensor([-10.0]),
        "boundary_distance_object_scale": torch.tensor([-0.2]),
    }


class OBBDiagnosticsTest(unittest.TestCase):
    def test_detailed_epoch_interval_keeps_first_periodic_and_final_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                diagnostics_enabled=True,
                diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=1,
                diagnostics_detailed_epoch_interval=10,
                diagnostics_query_topk=1,
                epoches=20,
                yaml_cfg={"model": "synthetic"},
                resume=None,
            )
            diagnostics = OBBDiagnostics(
                cfg, directory, model=torch.nn.Linear(3, 2)
            )
            dataset = _Dataset()
            diagnostics.start_evaluation(0, dataset)
            self.assertTrue(diagnostics.needs_detailed_eval_layers())
            diagnostics.start_evaluation(1, dataset)
            self.assertFalse(diagnostics.needs_detailed_eval_layers())
            diagnostics.start_evaluation(9, dataset)
            self.assertTrue(diagnostics.needs_detailed_eval_layers())
            diagnostics.start_evaluation(19, dataset)
            self.assertTrue(diagnostics.needs_detailed_eval_layers())
            diagnostics.close()

    def test_failed_atomic_append_preserves_last_complete_training_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                diagnostics_enabled=True, diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=0, diagnostics_query_topk=0,
                yaml_cfg={"model": "synthetic"}, resume=None,
            )
            diagnostics = OBBDiagnostics(cfg, directory)
            diagnostics.record_train_step({"epoch": 0, "global_step": 0})
            path = next(
                (Path(directory) / "diagnostics" / "train").glob(
                    "steps.rank*.jsonl.gz"))
            last_complete_stream = path.read_bytes()

            with mock.patch(
                    "engine.diagnostics.obb_diagnostics.os.fsync",
                    side_effect=OSError("synthetic storage failure")):
                with self.assertRaisesRegex(OSError, "synthetic storage failure"):
                    diagnostics.record_train_step({"epoch": 0, "global_step": 1})

            self.assertEqual(path.read_bytes(), last_complete_stream)
            with gzip.open(path, "rt") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual([record["global_step"] for record in records], [0])
            self.assertFalse(list(path.parent.glob(".*.rankwriter-*.tmp")))
            diagnostics.close()

    def test_postprocessor_traces_the_actual_nms_parent(self):
        outputs = {
            "pred_logits": torch.tensor([[[8.0], [7.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.2, 0.0],
                                           [0.51, 0.5, 0.4, 0.2, 0.0]]]),
        }
        processor = RotatedPostProcessor(
            num_classes=1, num_top_queries=2, score_threshold=0.0,
            nms_iou_threshold=0.1, max_detections=10)
        results, debug = processor(outputs, [_target()], return_diagnostics=True)
        self.assertEqual(len(results[0]["boxes"]), 1)
        self.assertEqual(debug[0]["status"].tolist(), [0, 1])
        self.assertEqual(debug[0]["suppressed_by"].tolist(), [-1, 0])
        self.assertGreater(float(debug[0]["suppression_iou"][1]), 0.1)

    def test_nms_parent_is_first_causal_box_not_largest_overlap(self):
        processor = RotatedPostProcessor(
            num_classes=1, num_top_queries=3, score_threshold=0.0,
            nms_iou_threshold=0.1, max_detections=10)
        boxes = torch.tensor([
            [20.0, 20.0, 20.0, 10.0, 0.0],
            [37.0, 20.0, 20.0, 10.0, 0.0],
            [29.0, 20.0, 20.0, 10.0, 0.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.zeros(3, dtype=torch.long)
        _, _, status, parent, parent_iou = processor._nms_trace(
            boxes, scores, labels)
        self.assertEqual(status.tolist(), [0, 0, 1])
        self.assertEqual(parent.tolist(), [-1, -1, 0])
        # Candidate 2 overlaps box 1 more, but box 0 is processed first and is
        # therefore the actual NMS edge that must appear in mechanism figures.
        self.assertLess(float(parent_iou[2]), 0.4)

    def test_matcher_exposes_selected_cost_components(self):
        outputs = {
            "pred_logits": torch.tensor([[[8.0], [-4.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.2, 0.0],
                                           [0.1, 0.1, 0.2, 0.1, 0.4]]]),
        }
        matcher = RotatedHungarianMatcher({
            "cost_class": 2, "cost_bbox": 5, "cost_angle": 2,
            "cost_kld": 2, "cost_chamfer": 0.5,
        })
        matching = matcher(outputs, [_target()], return_costs=True)
        self.assertEqual(matching["indices"][0][0].tolist(), [0])
        self.assertEqual(set(matching["matched_costs"][0]),
                         {"class", "bbox", "angle", "kld", "chamfer", "total"})
        self.assertEqual(len(matching["matched_costs"][0]["total"]), 1)

    def test_decoder_diagnostic_mode_preserves_all_eval_layers(self):
        model = build_tiny_model(num_denoising=0)
        features = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4),
                    torch.randn(1, 32, 2, 2)]
        model.eval().set_diagnostic_mode(True)
        outputs = model(features)
        self.assertEqual(outputs["diagnostic_layer_boxes"].shape[0], 2)
        torch.testing.assert_close(outputs["pred_boxes"], outputs["diagnostic_layer_boxes"][-1])

    def test_o2_diagnostics_preserve_the_exact_adr_decode_reference(self):
        """Bin-level research logs must reconstruct the actual O2 output."""

        from engine.rtv4.obb.methods.o2.adr import (
            adr_to_rbox,
            distribution_integral,
        )
        from engine.rtv4.rotated_box_ops import rotated_iou

        model = build_tiny_model(num_denoising=0, refinement_mode="o2_adr")
        features = [
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
        ]
        model.eval().set_diagnostic_mode(True)
        with torch.no_grad():
            outputs = model(features)
        references = outputs["diagnostic_layer_refs"]
        distributions = outputs["diagnostic_layer_distributions"]
        project = outputs["diagnostic_distribution_project"]
        residuals = distribution_integral(
            distributions, project, components=6
        )
        reconstructed = adr_to_rbox(
            references, residuals, normalized_angle=True
        )
        expected = outputs["diagnostic_layer_boxes"]
        iou = rotated_iou(
            reconstructed.reshape(-1, 5), expected.reshape(-1, 5),
            aligned=True, normalized_angle=True,
        )
        torch.testing.assert_close(iou, torch.ones_like(iou))

        target = _target()
        processor = RotatedPostProcessor(
            num_classes=3, num_top_queries=20, score_threshold=0.0,
            nms_iou_threshold=0.1, max_detections=20,
        )
        results, post = processor(outputs, [target], return_diagnostics=True)
        matcher = RotatedHungarianMatcher({
            "cost_class": 2, "cost_bbox": 0, "cost_angle": 0,
            "cost_kld": 2, "cost_chamfer": 5,
        })
        matching = matcher(outputs, [target], return_costs=True)
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                diagnostics_enabled=True,
                diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=1,
                diagnostics_query_topk=1,
                yaml_cfg={
                    "RotatedDFINETransformer": {
                        "refinement_mode": "o2_adr"
                    }
                },
            )
            diagnostics = OBBDiagnostics(cfg, directory)
            diagnostics.start_evaluation(0, _Dataset())
            diagnostics.record_evaluation_batch(
                outputs, [target], results, post, matching, _Dataset(),
                device=torch.device("cpu"),
            )
            diagnostics.finish_evaluation()
            query_file = next(
                (Path(directory) / "diagnostics" / "eval" / "epoch_0000").glob(
                    "queries.rank*.jsonl.gz"
                )
            )
            with gzip.open(query_file, "rt") as handle:
                records = [json.loads(line) for line in handle]
            self.assertTrue(records)
            for record in records:
                for layer in record["layers"]:
                    self.assertIn("reference_box", layer)
            matched_record = next(
                record for record in records
                if record["matched_gt_index"] is not None
            )
            for layer in matched_record["layers"]:
                self.assertIn("reference_box", layer)
                self.assertIn("reference_geometry", layer)
                self.assertEqual(
                    set(layer["fine_grained_distributions"]),
                    {
                        "external_left", "external_top", "external_right",
                        "external_bottom", "vertex_epsilon", "vertex_eta",
                    },
                )
            diagnostics.close()

    def test_complete_evaluation_record_is_self_contained(self):
        outputs = {
            "pred_logits": torch.tensor([[[8.0], [7.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.2, 0.0],
                                           [0.51, 0.5, 0.4, 0.2, 0.0]]]),
        }
        target = _target()
        processor = RotatedPostProcessor(
            num_classes=1, num_top_queries=2, score_threshold=0.0,
            nms_iou_threshold=0.1, max_detections=10)
        results, post = processor(outputs, [target], return_diagnostics=True)
        matcher = RotatedHungarianMatcher({
            "cost_class": 2, "cost_bbox": 5, "cost_angle": 2,
            "cost_kld": 2, "cost_chamfer": 0.5,
        })
        matching = matcher(outputs, [target], return_costs=True)
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                diagnostics_enabled=True, diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=1, diagnostics_query_topk=1,
                yaml_cfg={"model": "synthetic"},
            )
            diagnostics = OBBDiagnostics(
                cfg, directory, model=torch.nn.Linear(3, 2)
            )
            dataset = _Dataset()
            diagnostics.start_evaluation(3, dataset)
            diagnostics.record_evaluation_batch(
                outputs, [target], results, post, matching, dataset,
                timings={"forward_ms": 1.0}, device=torch.device("cpu"))
            evaluator = SimpleNamespace(
                metrics={"AP50": 1.0}, per_class={"car": 1.0}, stats=[1.0])
            diagnostics.finish_evaluation(evaluator)
            diagnostics.record_train_step({
                "epoch": 3, "global_step": 9,
                "gpu_memory": diagnostics.memory_snapshot(torch.device("cpu")),
            })
            diagnostics.record_train_step({"epoch": 3, "global_step": 10})
            diagnostics.record_train_epoch(3, {"loss": 1.25})

            root = Path(directory) / "diagnostics"
            self.assertTrue((root / "manifest.json").is_file())
            with (root / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(len(manifest["config_sha256"]), 64)
            self.assertEqual(manifest["run_id"], Path(directory).name)
            self.assertIsNone(manifest["refinement_mode"])
            self.assertEqual(
                manifest["files"]["model_structure"], "model_structure.json"
            )
            with (root / "model_structure.json").open(
                    encoding="utf-8") as handle:
                model_structure = json.load(handle)
            self.assertEqual(model_structure["parameter_count"], 8)
            self.assertEqual(model_structure["trainable_parameter_count"], 8)
            eval_dir = root / "eval" / "epoch_0003"
            expected = {"images", "ground_truth", "detections", "matches", "queries", "nms"}
            self.assertEqual({path.name.split(".")[0] for path in eval_dir.glob("*.jsonl.gz")}, expected)
            with gzip.open(next(eval_dir.glob("images.rank*.jsonl.gz")), "rt") as handle:
                image_record = json.loads(handle.readline())
            self.assertEqual(image_record["evaluation_batch_index"], 0)
            self.assertEqual(image_record["batch_item_index"], 0)
            with gzip.open(next(eval_dir.glob("matches.rank*.jsonl.gz")), "rt") as handle:
                match = json.loads(handle.readline())
            for key in ("center_error_px", "angle_error_deg", "corner_chamfer_px",
                        "rotated_iou", "hungarian_cost", "failure_stage"):
                self.assertIn(key, match)
            self.assertEqual(match["failure_stage"], "success")
            self.assertTrue(match["final_tp50"])
            self.assertTrue(match["final_tp75"])
            self.assertEqual(match["evaluation_outcome"], "tp75")
            self.assertEqual(match["altitude_m"], 60.0)
            self.assertEqual(match["illumination"], "day")
            self.assertEqual(match["source_image_id"], "park_day_60m_30c_frame_10")
            self.assertEqual(match["source_object_index"], 17)
            self.assertEqual(match["source_object_uid"], "park_day_60m_30c_frame_10:17")
            self.assertEqual(match["tile_origin"], [80, 0])
            self.assertEqual(match["visible_ratio"], 0.75)
            self.assertEqual(match["source_tile_count"], 2)
            self.assertEqual(match["boundary_distance_px"], -10.0)
            with gzip.open(next((root / "train").glob("steps.rank*.jsonl.gz")), "rt") as handle:
                train_records = [json.loads(line) for line in handle]
            self.assertEqual([record["global_step"] for record in train_records], [9, 10])
            self.assertEqual(train_records[0]["gpu_memory"]["peak_allocated_mb"], 0.0)
            with gzip.open(next((root / "train").glob("epochs.rank*.jsonl.gz")), "rt") as handle:
                epoch_records = [json.loads(line) for line in handle]
            self.assertEqual(epoch_records[0]["statistics"]["loss"], 1.25)
            diagnostics.close()


if __name__ == "__main__":
    unittest.main()
