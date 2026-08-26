import gzip
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from engine.data.dataset import CODroneDetection
from engine.data.dataset.codrone_tiling import (
    CODroneTilingProtocol,
    boundary_distances,
    generate_sliding_windows,
    parse_dota_annotations,
    polygon_window_iof,
    split_codrone_split,
)


def _read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CODroneTilingGeometryTest(unittest.TestCase):
    def test_compact_jpeg_is_the_manifested_formal_default(self):
        protocol = CODroneTilingProtocol()
        self.assertEqual(protocol.image_extension, ".jpg")
        self.assertEqual(protocol.image_quality, 95)
        self.assertEqual(protocol.to_manifest()["image_encoding"], "JPEG quality 95")

    def test_published_4k_window_layout_and_edge_overlap(self):
        protocol = CODroneTilingProtocol()
        windows = generate_sliding_windows(3840, 2160, protocol)
        self.assertEqual(protocol.step, 980)
        self.assertEqual(len(windows), 8)
        self.assertEqual(sorted({window.x_start for window in windows}), [0, 980, 1960, 2660])
        self.assertEqual(sorted({window.y_start for window in windows}), [0, 980])
        self.assertTrue(all(window.width == 1180 and window.height == 1180 for window in windows))
        self.assertTrue(all(window.effective_image_ratio == 1.0 for window in windows))
        # Boundary snapping makes the final horizontal overlap 480, not 200.
        self.assertEqual((1960 + 1180) - 2660, 480)

    def test_exact_iof_and_signed_boundary_distances(self):
        protocol = CODroneTilingProtocol(window_size=100, gap=20)
        window = generate_sliding_windows(180, 180, protocol)[0]
        polygon = [70, 20, 110, 20, 110, 60, 70, 60]
        self.assertAlmostEqual(polygon_window_iof(polygon, window), 0.75, places=8)
        center_distance, support_distance = boundary_distances(polygon, window)
        self.assertEqual(center_distance, 10.0)
        self.assertEqual(support_distance, -10.0)

    def test_small_image_fallback_retains_and_pads_one_window(self):
        protocol = CODroneTilingProtocol(window_size=100, gap=20)
        windows = generate_sliding_windows(60, 80, protocol)
        self.assertEqual(len(windows), 1)
        self.assertAlmostEqual(windows[0].effective_image_ratio, 0.48)
        self.assertEqual(windows[0].padding, (0, 0, 40, 20))

    def test_degenerate_annotation_is_audited_or_rejected_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.txt"
            path.write_text(
                "10 10 30 10 30 30 10 30 car 0\n"
                "5 5 5 5 5 5 5 5 truck 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Degenerate polygon"):
                parse_dota_annotations(path)
            invalid = []
            objects = parse_dota_annotations(path, strict=False, invalid_records=invalid)
            self.assertEqual(len(objects), 1)
            self.assertEqual(invalid[0]["line_number"], 2)
            self.assertEqual(invalid[0]["reason"], "degenerate_polygon")


class CODroneTilingIntegrationTest(unittest.TestCase):
    @staticmethod
    def _make_source(root, size=(180, 180)):
        source = root / "train"
        (source / "images").mkdir(parents=True)
        (source / "annfile").mkdir()
        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
        image[..., 2] = 80
        self_path = source / "images" / "scene.png"
        self_path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(self_path), image):
            raise RuntimeError("failed to create test image")
        annotations = "\n".join([
            "10 10 30 10 30 30 10 30 car 0",
            "70 20 110 20 110 60 70 60 truck 0",
            "40 40 50 40 50 50 40 50 ignored 0",
        ]) + "\n"
        (source / "annfile" / "scene.txt").write_text(annotations, encoding="utf-8")
        return source

    def test_materialization_metadata_annotations_and_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._make_source(root)
            output = root / "standard" / "train"
            protocol = CODroneTilingProtocol(
                window_size=100, gap=20, image_extension=".png")
            summary = split_codrone_split(
                source, output, protocol, nproc=1, preview_samples=1, preview_seed=3)

            self.assertTrue((output / "_SUCCESS").is_file())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["protocol"]["step"], 80)
            self.assertEqual(summary["source_images"], 1)
            self.assertEqual(summary["tiles"], 4)
            self.assertEqual(summary["regular_source_objects"], 2)
            self.assertEqual(summary["ignore_source_regions"], 1)
            self.assertEqual(summary["retained_assignments"], 4)
            self.assertEqual(summary["retained_regular_assignments"], 3)
            self.assertEqual(summary["retained_ignore_assignments"], 1)
            self.assertEqual(summary["truncated_retained_assignments"], 2)
            self.assertEqual(summary["truncated_regular_assignments"], 2)
            self.assertEqual(summary["truncated_ignore_assignments"], 0)
            self.assertEqual(summary["duplicated_regular_objects"], 1)
            self.assertEqual(summary["dropped_regular_objects"], 0)

            tile_records = _read_jsonl_gz(output / "diagnostics" / "tiles.jsonl.gz")
            object_records = _read_jsonl_gz(output / "diagnostics" / "objects.jsonl.gz")
            self.assertEqual(len(tile_records), 4)
            self.assertEqual(len(object_records), 3)
            truck = next(record for record in object_records if record["class_name"] == "truck")
            self.assertEqual(truck["retained_assignment_count"], 2)
            self.assertEqual(
                [round(item["iof"], 6) for item in truck["positive_window_assignments"]],
                [0.75, 0.75])
            self.assertTrue(all(item["truncated"] for item in truck["positive_window_assignments"]))
            self.assertTrue(all(item["support_to_boundary_px"] == -10.0
                                for item in truck["positive_window_assignments"]))

            annotation_files = sorted((output / "annfile").glob("*.txt"))
            self.assertEqual(len(annotation_files), 4)
            metadata_files = sorted((output / "metadata").glob("*.json"))
            self.assertEqual(len(metadata_files), 4)
            truck_lines = [
                line for path in annotation_files
                for line in path.read_text(encoding="utf-8").splitlines()
                if " truck " in line
            ]
            self.assertEqual(len(truck_lines), 2)
            self.assertTrue(all(line.endswith(" truck 2") for line in truck_lines))
            previews = list((output / "diagnostics" / "visualizations").glob("*.jpg"))
            self.assertEqual(len(previews), 1)
            self.assertIsNotNone(cv2.imread(str(previews[0])))

            dataset = CODroneDetection(output)
            self.assertEqual(len(dataset), 4)
            image, target = dataset.load_item(0)
            self.assertEqual(image.size, (100, 100))
            self.assertEqual(target["partition_id"], "codrone_dota_w100_g20_iof0p7")
            self.assertEqual(target["source_image_id"], "scene")
            self.assertEqual(target["tile_origin"].tolist(), [0, 0])
            self.assertEqual(target["source_object_index"].tolist(), [0, 1])
            self.assertEqual(target["source_tile_count"].tolist(), [1, 2])
            np.testing.assert_allclose(target["visible_ratio"].numpy(), [1.0, 0.75])
            np.testing.assert_allclose(target["boundary_distance_px"].numpy(), [10.0, -10.0])

    def test_degenerate_source_row_is_skipped_and_written_to_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._make_source(root)
            annotation = source / "annfile" / "scene.txt"
            annotation.write_text(
                annotation.read_text(encoding="utf-8")
                + "55 55 55 55 55 55 55 55 bus 0\n",
                encoding="utf-8",
            )
            output = root / "standard" / "train"
            summary = split_codrone_split(
                source, output, CODroneTilingProtocol(window_size=100, gap=20,
                                                      image_extension=".png"),
                nproc=1, preview_samples=0)
            self.assertEqual(summary["invalid_annotation_count"], 1)
            invalid = _read_jsonl_gz(
                output / "diagnostics" / "invalid_annotations.jsonl.gz")
            self.assertEqual(len(invalid), 1)
            self.assertEqual(invalid[0]["source_image_id"], "scene")
            self.assertEqual(invalid[0]["line_number"], 4)
            source_record = _read_jsonl_gz(output / "diagnostics" / "images.jsonl.gz")[0]
            self.assertEqual(source_record["invalid_annotation_lines"], [4])
            self.assertEqual(source_record["object_count"], 2)
            dataset = CODroneDetection(output)
            self.assertEqual(sum(len(dataset.get_ground_truth(i)["boxes"])
                                 for i in range(len(dataset))), 3)

    def test_padding_matches_official_opencv_channel_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._make_source(root, size=(60, 80))
            output = root / "standard" / "train"
            protocol = CODroneTilingProtocol(
                window_size=100, gap=20, image_extension=".png")
            summary = split_codrone_split(source, output, protocol, nproc=1, preview_samples=0)
            self.assertEqual(summary["padded_tiles"], 1)
            patch = cv2.imread(str(next((output / "images").glob("*.png"))))
            self.assertEqual(tuple(int(value) for value in patch[99, 99]), (104, 116, 124))
            self.assertEqual(tuple(int(value) for value in patch[79, 59]), (59, 79, 80))

    def test_outputs_are_deterministic_except_run_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._make_source(root)
            protocol = CODroneTilingProtocol(
                window_size=100, gap=20, image_extension=".png")
            outputs = [root / "run_a", root / "run_b"]
            for index, output in enumerate(outputs):
                split_codrone_split(
                    source, output, protocol, nproc=index + 1,
                    preview_samples=1, preview_seed=11)
            for relative in (
                "diagnostics/images.jsonl.gz",
                "diagnostics/tiles.jsonl.gz",
                "diagnostics/objects.jsonl.gz",
            ):
                self.assertEqual(_read_jsonl_gz(outputs[0] / relative),
                                 _read_jsonl_gz(outputs[1] / relative))
            first_annotations = {
                path.name: path.read_bytes() for path in (outputs[0] / "annfile").glob("*.txt")}
            second_annotations = {
                path.name: path.read_bytes() for path in (outputs[1] / "annfile").glob("*.txt")}
            self.assertEqual(first_annotations, second_annotations)
            first_metadata = {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in (outputs[0] / "metadata").glob("*.json")}
            second_metadata = {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in (outputs[1] / "metadata").glob("*.json")}
            self.assertEqual(first_metadata, second_metadata)

    def test_rejects_source_output_directory_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._make_source(root)
            with self.assertRaisesRegex(ValueError, "must not replace or contain"):
                split_codrone_split(source, source / "generated", preview_samples=0)


if __name__ == "__main__":
    unittest.main()
