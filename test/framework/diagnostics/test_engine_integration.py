import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from PIL import Image

from engine.evaluation.obb import DotaOBBEvaluator
from engine.diagnostics import OBBDiagnostics
from engine.rtv4 import RotatedHungarianMatcher, RotatedPostProcessor, RotatedRTv4Criterion
from engine.solver.det_engine import evaluate, train_one_epoch
from engine.solver.det_solver import DetSolver


class _ToyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[4.0], [-2.0]]))
        self.box_parameters = nn.Parameter(torch.tensor([
            [0.0, 0.0, -0.4, -1.4, -4.0],
            [-2.0, -2.0, -1.0, -2.0, 0.0],
        ]))

    def forward(self, samples, targets=None, teacher_encoder_output=None):
        batch = len(samples)
        return {
            "pred_logits": self.logits.sigmoid().logit().unsqueeze(0).expand(batch, -1, -1),
            "pred_boxes": self.box_parameters.sigmoid().unsqueeze(0).expand(batch, -1, -1),
        }


class _Dataset:
    classes = ("car",)
    image_ids = ("synthetic_day_60m_30c_frame_1",)
    root = Path("/synthetic")

    def __init__(self, image_path=None):
        self.image_path = image_path

    def __len__(self):
        return 1

    def get_ground_truth(self, image_id):
        return {
            "boxes": torch.tensor([[50.0, 50.0, 40.0, 20.0, 0.0]]),
            "labels": torch.tensor([0]), "difficulty": torch.tensor([0]),
            "ignore_boxes": torch.empty((0, 5)),
            "image_name": self.image_ids[0], "image_path": self.image_path,
        }


def _target():
    return {
        "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.0]]),
        "labels": torch.tensor([0]), "difficulty": torch.tensor([0]),
        "image_id": torch.tensor([0]), "orig_size": torch.tensor([100, 100]),
        "size": torch.tensor([100, 100]),
        "scale_factor": torch.tensor([1.0, 1.0]),
        "padding": torch.tensor([0, 0, 0, 0]),
    }


class DiagnosticEngineIntegrationTest(unittest.TestCase):
    def test_solver_closes_diagnostics_when_training_raises(self):
        class _ClosingDiagnostics(OBBDiagnostics):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.was_closed = False

            def close(self):
                self.was_closed = True
                super().close()

        with tempfile.TemporaryDirectory() as directory:
            solver = object.__new__(DetSolver)
            solver.train = lambda: None
            solver.cfg = SimpleNamespace(
                diagnostics_enabled=True, diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=0, diagnostics_query_topk=0,
                yaml_cfg={"model": "failure-test"}, resume=None,
            )
            solver.output_dir = Path(directory)

            def fail(_args):
                raise RuntimeError("synthetic training failure")

            solver._fit_with_diagnostics = fail
            with mock.patch(
                    "engine.solver.det_solver.OBBDiagnostics", _ClosingDiagnostics):
                with self.assertRaisesRegex(RuntimeError, "synthetic training failure"):
                    solver.fit()
                self.assertTrue(solver.diagnostics.was_closed)

    def test_training_and_evaluation_engines_write_structured_logs(self):
        device = torch.device("cpu")
        model = _ToyDetector()
        matcher = RotatedHungarianMatcher({
            "cost_class": 2, "cost_bbox": 5, "cost_angle": 2,
            "cost_kld": 2, "cost_chamfer": 0.5,
        })
        criterion = RotatedRTv4Criterion(
            matcher, {"loss_focal": 1, "loss_bbox": 5,
                      "loss_angle": 2, "loss_kld": 2},
            losses=("focal", "boxes"), num_classes=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        images = torch.rand(1, 3, 32, 32)
        loader = [(images, [_target()])]

        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(
                diagnostics_enabled=True, diagnostics_train_interval=1,
                diagnostics_detailed_image_limit=1, diagnostics_query_topk=1,
                yaml_cfg={"model": "toy"}, resume=None,
            )
            diagnostics = OBBDiagnostics(cfg, directory)
            train_stats, _ = train_one_epoch(
                False, None, model, criterion, loader, optimizer, device, epoch=0,
                max_norm=0.1, print_freq=10, diagnostics=diagnostics)
            self.assertIn("loss", train_stats)

            image_path = Path(directory) / "synthetic.jpg"
            Image.new("RGB", (100, 100), (80, 90, 100)).save(image_path)
            dataset = _Dataset(str(image_path))
            evaluator = DotaOBBEvaluator(dataset, iou_thresholds=[0.5, 0.75])
            postprocessor = RotatedPostProcessor(
                num_classes=1, num_top_queries=2, score_threshold=0.0,
                nms_iou_threshold=0.1, max_detections=10)
            stats, _ = evaluate(
                model, criterion, postprocessor, loader, evaluator, device,
                diagnostics=diagnostics, epoch=0)
            self.assertIn("dota_eval_rbox", stats)

            root = Path(directory) / "diagnostics"
            with gzip.open(next((root / "train").glob("steps.rank*.jsonl.gz")), "rt") as handle:
                train_record = json.loads(handle.readline())
            self.assertIn("gradients_before_clip", train_record)
            self.assertIn("amp", train_record)
            self.assertFalse(train_record["amp"]["optimizer_step_skipped"])
            self.assertIn("forward", train_record["timing_ms"])
            self.assertIn("main_hungarian_matches", train_record)
            diagnostics.close()
            eval_dir = root / "eval" / "epoch_0000"
            with gzip.open(next(eval_dir.glob("matches.rank*.jsonl.gz")), "rt") as handle:
                match_record = json.loads(handle.readline())
            self.assertIn(match_record["failure_stage"], {
                "success", "raw_localization", "raw_classification",
                "score_threshold", "topk_truncation", "rotated_nms",
                "max_detections", "final_assignment",
            })
            with (eval_dir / "metrics.json").open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertIn("AP50_DOTA07", metrics["metrics"])

            analysis_dir = Path(directory) / "analysis"
            subprocess.run([
                sys.executable, "tools/analysis/summarize_obb_diagnostics.py",
                directory, "--output", str(analysis_dir),
            ], check=True, cwd=Path(__file__).resolve().parents[3],
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertTrue((analysis_dir / "summary.json").is_file())
            self.assertTrue((analysis_dir / "object_error_components.png").is_file())

            visualization_dir = Path(directory) / "mechanisms"
            subprocess.run([
                sys.executable, "tools/analysis/visualize_obb_mechanisms.py",
                directory, "--output", str(visualization_dir), "--count", "1",
            ], check=True, cwd=Path(__file__).resolve().parents[3],
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(len(list(visualization_dir.glob("case_*.png"))), 1)

            comparison_dir = Path(directory) / "paired"
            subprocess.run([
                sys.executable, "tools/analysis/compare_obb_mechanisms.py",
                directory, directory, "--output", str(comparison_dir), "--count", "1",
            ], check=True, cwd=Path(__file__).resolve().parents[3],
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(len(list(comparison_dir.glob("paired_*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
