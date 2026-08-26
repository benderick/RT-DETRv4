import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from engine.rtv4.obb.methods.o2.adr import o2_weighting_function
from tools.research.o2.visualize_mechanisms import (
    _adr_distribution_figure,
    _adr_geometry_figure,
    _decoder_progression_figure,
    _epoch_progression_figure,
    _select_tracks,
    _stage_metric_history,
)


def _record(image_path, epoch, final_offset):
    project = o2_weighting_function(32).tolist()
    probability = np.linspace(1.0, 2.0, 33)
    probability /= probability.sum()
    components = {
        name: {
            "probabilities": probability.tolist(),
            "weighted_values": (probability * np.asarray(project)).tolist(),
            "expected_residual": float(np.dot(probability, project)),
            "entropy": 3.0,
        }
        for name in (
            "external_left", "external_top", "external_right",
            "external_bottom", "vertex_epsilon", "vertex_eta",
        )
    }
    gt = [128.0, 128.0, 72.0, 28.0, 0.25]
    pre = [118.0, 126.0, 68.0, 26.0, 0.18]
    boxes = [
        pre,
        [121.0, 126.5, 69.0, 26.0, 0.19],
        [124.0, 127.0, 70.0, 27.0, 0.21],
        [126.0, 127.5, 71.0, 27.5, 0.23],
        [128.0 + final_offset, 128.0, 72.0, 28.0, 0.25],
    ]
    stages = [{
        "stage": "pre_box", "stage_index": -1, "display_name": "Pre-box",
        "box": boxes[0], "input_reference_box": [115., 125., 65., 25., .15],
        "rotated_iou": .55 + .05 * epoch, "target_class_score": .4 + .05 * epoch,
    }]
    for index, box in enumerate(boxes[1:]):
        iou = .60 + .07 * index + .05 * epoch - abs(final_offset) * .01
        stages.append({
            "stage": "decoder_layer", "stage_index": index,
            "display_name": f"Decoder {index}", "box": box,
            "input_reference_box": boxes[index], "initial_anchor_box": pre,
            "rotated_iou": iou, "angle_error_deg": 4.0 - .7 * index,
            "target_class_score": .5 + .08 * index,
            "transition_from_previous": {"rotated_iou_delta": .05},
            "fine_grained_distributions": components,
            "location_quality_estimator": {"target_class_logit_delta": .1 * index},
        })
    return {
        "epoch": epoch, "image_id": 0, "image_name": "synthetic",
        "image_path": str(image_path), "matched_gt_index": 0,
        "query_index": epoch + 3, "gt_box": gt,
        "distribution_codebook": project, "stages": stages,
        "stage_sequence": [stage["display_name"] for stage in stages],
    }


class O2VisualizationTest(unittest.TestCase):
    def test_paper_figures_and_selection_are_generated_from_declared_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.jpg"
            Image.new("RGB", (256, 256), (70, 80, 90)).save(image_path)
            records = {
                0: _record(image_path, 0, 5.0),
                4: _record(image_path, 4, 0.5),
            }
            tracks = {(0, 0): records}
            selected = _select_tracks(tracks, 1, [])
            self.assertEqual(selected[0][1], (0, 0))
            self.assertEqual(selected[0][0], "largest_cross_epoch_iou_gain")

            renderers = (
                (_adr_distribution_figure, records[4], "distribution.png"),
                (_adr_geometry_figure, records[4], "geometry.png"),
                (_decoder_progression_figure, records[4], "decoder.png"),
            )
            for renderer, record, name in renderers:
                path = directory / name
                self.assertTrue(renderer(record, path))
                self.assertGreater(path.stat().st_size, 1000)
            epoch_path = directory / "epochs.png"
            self.assertTrue(_epoch_progression_figure(records, epoch_path, 6))
            self.assertGreater(epoch_path.stat().st_size, 1000)

    def test_full_validation_stage_history_uses_refinement_metric_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_order = ["pre_box", "decoder_0", "decoder_1"]
            for epoch in (0, 1):
                target = root / "eval" / f"epoch_{epoch:04d}"
                target.mkdir(parents=True)
                stages = {
                    stage: {"metrics": {
                        "AP50_DOTA07": .4 + .1 * epoch + .02 * index,
                        "AP75_DOTA07": .2 + .1 * epoch + .02 * index,
                        "mAP50_75_DOTA07": .3 + .1 * epoch + .02 * index,
                    }}
                    for index, stage in enumerate(stage_order)
                }
                (target / "refinement_stages.json").write_text(json.dumps({
                    "stage_order": stage_order, "stages": stages,
                }), encoding="utf-8")
            output = root / "history.png"
            self.assertTrue(_stage_metric_history(root, output))
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
