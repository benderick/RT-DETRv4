import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn

from engine.data.dataset import CODroneDetection
from engine.evaluation.obb import MergedDotaOBBEvaluator
from engine.data.dataset.codrone_tiling import CODroneTilingProtocol, split_codrone_split
from engine.data.transforms import Compose
from engine.diagnostics import OBBDiagnostics
from engine.rtv4 import RotatedHungarianMatcher, RotatedPostProcessor
from engine.solver.det_engine import evaluate


REAL_TINY_VAL = Path(
    "/home/liuxiaolong/futurama/data/CODrone/standard_patches_t/val_t")


class _FixedOutputModel(nn.Module):
    def __init__(self, logits, boxes):
        super().__init__()
        self.register_buffer("fixed_logits", logits)
        self.register_buffer("fixed_boxes", boxes)

    def forward(self, images):
        if len(images) != len(self.fixed_logits):
            raise ValueError("Synthetic model expects the complete ordered tile batch")
        return {"pred_logits": self.fixed_logits, "pred_boxes": self.fixed_boxes}


class _MatcherCriterion(nn.Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher


class MergedDotaOBBEvaluatorTest(unittest.TestCase):
    @unittest.skipUnless(
        (REAL_TINY_VAL / "_SUCCESS").is_file(),
        "official tiny CODrone patches have not been materialized",
    )
    def test_real_tiny_perfect_tile_ceiling_exposes_nms_collisions(self):
        tile_dataset = CODroneDetection(REAL_TINY_VAL)
        evaluator = MergedDotaOBBEvaluator(
            tile_dataset, iou_thresholds=[0.5, 0.75], merge_iou_threshold=0.1)
        evaluator.update(self._perfect_tile_predictions(tile_dataset))
        evaluator.accumulate()
        self.assertEqual(evaluator.merge_summary["candidate_count_after_local_nms"], 709)
        self.assertEqual(evaluator.merge_summary["kept_after_global_nms"], 482)
        self.assertEqual(evaluator.merge_summary["global_nms_suppressed"], 227)
        self.assertEqual(evaluator.merge_summary["same_best_gt_suppressions"], 196)
        self.assertEqual(evaluator.merge_summary["different_best_gt_suppressions"], 31)
        self.assertAlmostEqual(
            evaluator.metrics["mAP50_75_DOTA07"], 0.9772727272727273)

    @staticmethod
    def _materialize(root):
        source = root / "val"
        (source / "images").mkdir(parents=True)
        (source / "annfile").mkdir()
        image = np.full((180, 180, 3), 80, dtype=np.uint8)
        cv2.imwrite(str(source / "images" / "scene.jpg"), image)
        (source / "annfile" / "scene.txt").write_text("\n".join([
            "10 10 30 10 30 30 10 30 car 0",
            "70 20 110 20 110 60 70 60 truck 0",
            "40 40 50 40 50 50 40 50 ignored 0",
        ]) + "\n", encoding="utf-8")
        output = root / "patches" / "val"
        split_codrone_split(
            source, output,
            CODroneTilingProtocol(
                window_size=100, gap=20, image_extension=".png"),
            preview_samples=0,
        )
        return source, output

    @staticmethod
    def _perfect_tile_predictions(dataset):
        predictions = {}
        for image_id in range(len(dataset)):
            ground_truth = dataset.get_ground_truth(image_id)
            predictions[image_id] = {
                "boxes": ground_truth["boxes"].clone(),
                "labels": ground_truth["labels"].clone(),
                "scores": torch.full((len(ground_truth["boxes"]),), 0.9),
            }
        return predictions

    def test_translate_global_nms_and_original_image_ap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output = self._materialize(root)
            tile_dataset = CODroneDetection(output)
            evaluator = MergedDotaOBBEvaluator(
                tile_dataset, iou_thresholds=[0.5, 0.75], merge_iou_threshold=0.1)
            evaluator.update(self._perfect_tile_predictions(tile_dataset))
            evaluator.accumulate()

            self.assertEqual(len(evaluator.dataset), 1)
            self.assertEqual(len(evaluator.tile_predictions), 4)
            self.assertEqual(len(evaluator.predictions[0]["boxes"]), 2)
            self.assertEqual(evaluator.merge_summary["candidate_count_after_local_nms"], 3)
            self.assertEqual(evaluator.merge_summary["kept_after_global_nms"], 2)
            self.assertEqual(evaluator.merge_summary["global_nms_suppressed"], 1)
            self.assertEqual(evaluator.merge_summary["cross_tile_suppressions"], 1)
            for value in evaluator.metrics.values():
                self.assertAlmostEqual(value, 1.0, places=7)

            truck_records = [
                record for record in evaluator.merge_candidate_records
                if record["class_name"] == "truck"]
            self.assertEqual(len(truck_records), 2)
            suppressed = next(
                record for record in truck_records
                if record["status"] == "global_nms_overlap")
            self.assertTrue(suppressed["cross_tile_suppression"])
            self.assertEqual(suppressed["source_object_uid"], "scene:1")
            self.assertAlmostEqual(float(suppressed["best_source_gt_iou"]), 1.0)
            self.assertIsNotNone(suppressed["suppressor_tile_id"])

    def test_missing_tile_fails_instead_of_reporting_partial_ap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output = self._materialize(root)
            tile_dataset = CODroneDetection(output)
            evaluator = MergedDotaOBBEvaluator(tile_dataset)
            predictions = self._perfect_tile_predictions(tile_dataset)
            predictions.pop(max(predictions))
            evaluator.update(predictions)
            with self.assertRaisesRegex(RuntimeError, "missing 1 tile predictions"):
                evaluator.accumulate()

    def test_compact_merge_keeps_final_provenance_without_candidate_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output = self._materialize(root)
            tile_dataset = CODroneDetection(output)
            evaluator = MergedDotaOBBEvaluator(
                tile_dataset, iou_thresholds=[0.5, 0.75],
                record_merge_candidates=False)
            evaluator.update(self._perfect_tile_predictions(tile_dataset))
            evaluator.accumulate()

            self.assertEqual(evaluator.merge_candidate_records, [])
            self.assertFalse(evaluator.merge_summary["merge_candidate_trace_recorded"])
            self.assertIsNone(evaluator.merge_summary["same_best_gt_suppressions"])
            provenance = evaluator.final_prediction_metadata[0]
            count = len(evaluator.predictions[0]["boxes"])
            self.assertEqual(len(provenance["tile_ids"]), count)
            self.assertEqual(provenance["tile_origins"].shape, (count, 2))
            self.assertEqual(provenance["local_boxes"].shape, (count, 5))

    def test_merge_evidence_is_written_to_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output = self._materialize(root)
            tile_dataset = CODroneDetection(output)
            evaluator = MergedDotaOBBEvaluator(
                tile_dataset, iou_thresholds=[0.5, 0.75])
            evaluator.update(self._perfect_tile_predictions(tile_dataset))
            evaluator.accumulate()

            diagnostics_root = root / "run"
            cfg = SimpleNamespace(
                diagnostics_enabled=True,
                diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=0,
                diagnostics_query_topk=0,
                yaml_cfg={"model": "synthetic"},
                resume=None,
            )
            diagnostics = OBBDiagnostics(cfg, diagnostics_root)
            diagnostics.start_evaluation(2, tile_dataset)
            diagnostics.record_global_merge(evaluator)
            diagnostics.finish_evaluation(evaluator)
            diagnostics.close()

            eval_dir = diagnostics_root / "diagnostics" / "eval" / "epoch_0002"
            self.assertTrue((eval_dir / "merge_summary.json").is_file())
            with gzip.open(
                    next(eval_dir.glob("merge_candidates.rank*.jsonl.gz")),
                    "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 3)
            self.assertEqual(
                sum(record["status"] == "global_nms_overlap" for record in records), 1)
            with (eval_dir / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertEqual(metrics["merge_summary"]["kept_after_global_nms"], 2)

            visualization_dir = root / "merge_visuals"
            subprocess.run([
                sys.executable,
                "tools/analysis/visualize_tile_merge_mechanisms.py",
                str(diagnostics_root), "--epoch", "2", "--count", "1",
                "--mode", "duplicate",
                "--output", str(visualization_dir),
            ], check=True, cwd=Path(__file__).resolve().parents[3],
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(len(list(visualization_dir.glob("merge_*.png"))), 1)

            analysis_dir = root / "merge_analysis"
            subprocess.run([
                sys.executable,
                "tools/analysis/summarize_obb_diagnostics.py",
                str(diagnostics_root), "--epoch", "2",
                "--output", str(analysis_dir),
            ], check=True, cwd=Path(__file__).resolve().parents[3],
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertTrue((analysis_dir / "tile_merge_mechanisms.csv").is_file())
            self.assertTrue((analysis_dir / "tile_merge_nms_mechanism.png").is_file())

    def test_evaluate_engine_keeps_local_diagnostics_but_reports_source_ap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, output = self._materialize(root)
            transforms = Compose(ops=[
                {"type": "RotatedResizePad", "size": [100, 100]},
                {"type": "RotatedSanitizeBoxes", "min_size": 1, "min_visible": 0.0},
                {"type": "RotatedConvertToTensor", "normalize_boxes": True},
            ])
            tile_dataset = CODroneDetection(output, transforms=transforms)
            samples = [tile_dataset[index] for index in range(len(tile_dataset))]
            images = torch.stack([sample[0] for sample in samples])
            targets = [sample[1] for sample in samples]
            query_count, class_count = 2, len(tile_dataset.classes)
            logits = torch.full((len(samples), query_count, class_count), -10.0)
            boxes = torch.full((len(samples), query_count, 5), 0.1)
            for batch_index, target in enumerate(targets):
                for gt_index, (box, label) in enumerate(
                        zip(target["boxes"], target["labels"])):
                    boxes[batch_index, gt_index] = box
                    logits[batch_index, gt_index, label] = 10.0
            model = _FixedOutputModel(logits, boxes)
            matcher = RotatedHungarianMatcher({
                "cost_class": 2, "cost_bbox": 5, "cost_angle": 2,
                "cost_kld": 2, "cost_chamfer": 0.5,
            })
            criterion = _MatcherCriterion(matcher)
            postprocessor = RotatedPostProcessor(
                num_classes=class_count, num_top_queries=query_count * class_count,
                score_threshold=0.5, nms_iou_threshold=0.1,
                max_detections=10)
            evaluator = MergedDotaOBBEvaluator(
                tile_dataset, iou_thresholds=[0.5, 0.75])
            cfg = SimpleNamespace(
                diagnostics_enabled=True,
                diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=4,
                diagnostics_query_topk=2,
                yaml_cfg={"model": "fixed"},
                resume=None,
            )
            diagnostics = OBBDiagnostics(cfg, root / "engine_run")
            stats, _ = evaluate(
                model, criterion, postprocessor, [(images, targets)], evaluator,
                torch.device("cpu"), diagnostics=diagnostics, epoch=0)
            diagnostics.close()

            self.assertAlmostEqual(stats["dota_eval_rbox"][0], 1.0)
            eval_dir = root / "engine_run" / "diagnostics" / "eval" / "epoch_0000"
            with gzip.open(
                    next(eval_dir.glob("matches.rank*.jsonl.gz")),
                    "rt", encoding="utf-8") as handle:
                local_matches = [json.loads(line) for line in handle]
            self.assertEqual(len(local_matches), 3)
            self.assertTrue(all(record["tile_id"] for record in local_matches))
            self.assertTrue((eval_dir / "merge_summary.json").is_file())
            with gzip.open(
                    next(eval_dir.glob("merge_candidates.rank*.jsonl.gz")),
                    "rt", encoding="utf-8") as handle:
                merge_records = [json.loads(line) for line in handle]
            self.assertEqual(len(merge_records), 3)
            self.assertEqual(
                sum(record["status"] == "global_nms_overlap" for record in merge_records), 1)


if __name__ == "__main__":
    unittest.main()
