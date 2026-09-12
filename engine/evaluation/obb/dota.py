"""Dataset-agnostic DOTA-style oriented-box evaluation.

The evaluator consumes a small dataset protocol instead of a concrete
dataset class.  Dataset adapters own parsing; this module owns AP, DOTA
Task1 export, tiled prediction merge, and the corresponding diagnostics.
"""

from __future__ import annotations

import copy
import json
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from ...core import register
from ...misc import dist_utils
from ...rtv4.rotated_box_ops import (
    class_aware_rotated_nms,
    rotated_iou,
    rbox_to_corners,
)
def _average_precision(recall, precision, use_07_metric=False):
    if len(recall) == 0:
        return 0.0
    if use_07_metric:
        return float(np.mean([
            np.max(precision[recall >= threshold]) if np.any(recall >= threshold) else 0.0
            for threshold in np.linspace(0, 1, 11)
        ]))
    recall_points = np.linspace(0, 1, 101)
    return float(np.mean([
        np.max(precision[recall >= threshold]) if np.any(recall >= threshold) else 0.0
        for threshold in recall_points
    ]))


@register()
class DotaOBBEvaluator:
    """Evaluate original-image pixel OBBs, respecting DOTA difficulty flags."""

    STAT_NAMES = ("mAP50_75_DOTA07", "AP50_DOTA07", "AP75_DOTA07", "mAP50_95", "AP50", "AP75")

    def __init__(self, dataset, iou_thresholds=None, use_07_metric=True,
                 selection_metric="mAP50_75_DOTA07"):
        if selection_metric not in self.STAT_NAMES:
            raise ValueError(f"Unknown OBB selection_metric: {selection_metric}")
        self.selection_metric = selection_metric
        self.selection_index = self.STAT_NAMES.index(selection_metric)
        self.dataset = dataset
        self.iou_thresholds = np.asarray(
            iou_thresholds if iou_thresholds is not None else np.arange(0.5, 0.96, 0.05),
            dtype=np.float64)
        self.use_07_metric = bool(use_07_metric)
        self.iou_types = ("rbox",)
        self.coco_eval = {}
        self._ground_truth_by_class = None
        self._positives_by_class = None
        self.cleanup()

    def cleanup(self):
        self.predictions = {}
        self.metrics = {}
        self.per_class = {}
        self.per_class_metrics = {}
        # stats[0] remains the default checkpoint metric; selection_index can
        # explicitly select another metric without changing this vector.
        # Keep it aligned with MMRotate DOTAMetric(iou_thrs=[0.5, 0.75]):
        # DOTA/VOC AP averaged over AP50 and AP75, not COCO AP@[.50:.95].
        self.stats = np.zeros(6, dtype=np.float64)

    def clone_empty(self):
        """Create an evaluator with identical protocol and no predictions.

        Shallow-copying is intentional: immutable dataset/protocol state and
        the read-only ground-truth cache are shared, while ``cleanup`` resets
        every prediction/result container.  This gives diagnostics a generic
        way to evaluate pre-box and decoder stages without rebuilding
        dataset-specific evaluators or duplicating their merge policy.
        """

        evaluator = copy.copy(self)
        evaluator.cleanup()
        return evaluator

    def reuse_ground_truth_cache_from(self, evaluator):
        self._ground_truth_by_class = evaluator._ground_truth_by_class
        self._positives_by_class = evaluator._positives_by_class

    def update(self, predictions):
        for image_id, prediction in predictions.items():
            self.predictions[int(image_id)] = {
                "boxes": prediction["boxes"].detach().float().cpu(),
                "scores": prediction["scores"].detach().float().cpu(),
                "labels": prediction["labels"].detach().long().cpu(),
            }

    def synchronize_between_processes(self):
        gathered = dist_utils.all_gather(self.predictions)
        self.predictions = {}
        for rank_predictions in gathered:
            self.predictions.update(rank_predictions)

    def _ground_truth_cache(self):
        if self._ground_truth_by_class is not None:
            return self._ground_truth_by_class, self._positives_by_class

        class_count = len(self.dataset.classes)
        ground_truth_by_class = [dict() for _ in range(class_count)]
        positives_by_class = np.zeros(class_count, dtype=np.int64)

        for image_id in range(len(self.dataset)):
            annotation = self.dataset.get_ground_truth(image_id)
            ignore_boxes = annotation.get(
                "ignore_boxes", annotation["boxes"].new_empty((0, 5))).float()
            for class_index in range(class_count):
                mask = annotation["labels"] == class_index
                boxes = annotation["boxes"][mask].float()
                difficult = annotation["difficulty"][mask].bool()
                if len(ignore_boxes):
                    boxes = torch.cat((boxes, ignore_boxes), dim=0)
                    difficult = torch.cat((
                        difficult,
                        torch.ones(len(ignore_boxes), dtype=torch.bool)))
                ground_truth_by_class[class_index][image_id] = {
                    "boxes": boxes,
                    "difficult": difficult,
                }
                positives_by_class[class_index] += int((~annotation["difficulty"][mask].bool()).sum())

        self._ground_truth_by_class = ground_truth_by_class
        self._positives_by_class = positives_by_class
        return ground_truth_by_class, positives_by_class

    def _class_detections(self, class_index):
        detections_by_image = {}
        scores, image_ids, local_indices = [], [], []
        for image_id, prediction in self.predictions.items():
            mask = prediction["labels"] == class_index
            if not bool(mask.any()):
                continue
            class_boxes = prediction["boxes"][mask].float()
            class_scores = prediction["scores"][mask].float()
            detections_by_image[image_id] = {"boxes": class_boxes, "scores": class_scores}
            scores.append(class_scores)
            image_ids.append(torch.full((len(class_scores),), image_id, dtype=torch.long))
            local_indices.append(torch.arange(len(class_scores), dtype=torch.long))

        if not scores:
            empty_long = torch.empty(0, dtype=torch.long)
            return detections_by_image, torch.empty(0), empty_long, empty_long

        scores = torch.cat(scores)
        image_ids = torch.cat(image_ids)
        local_indices = torch.cat(local_indices)
        order = torch.argsort(scores, descending=True)
        return detections_by_image, scores[order], image_ids[order], local_indices[order]

    def _class_precision_recall_inputs(self, class_index, thresholds):
        ground_truth_by_class, positives_by_class = self._ground_truth_cache()
        ground_truth = ground_truth_by_class[class_index]
        positives = int(positives_by_class[class_index])
        if positives == 0:
            return None, None, positives

        detections_by_image, _, image_ids, local_indices = self._class_detections(class_index)
        thresholds = np.asarray(thresholds, dtype=np.float64)
        true_positive = [[] for _ in thresholds]
        false_positive = [[] for _ in thresholds]
        detected = {
            image_id: torch.zeros((len(thresholds), len(record["boxes"])), dtype=torch.bool)
            for image_id, record in ground_truth.items()
            if len(record["boxes"])
        }

        best_overlaps, best_matches = {}, {}
        for image_id, detections in detections_by_image.items():
            record = ground_truth.get(image_id)
            num_detections = len(detections["boxes"])
            if record is None or len(record["boxes"]) == 0:
                best_overlaps[image_id] = torch.zeros(num_detections, dtype=torch.float32)
                best_matches[image_id] = torch.full((num_detections,), -1, dtype=torch.long)
                continue
            overlaps = rotated_iou(
                detections["boxes"], record["boxes"], model_space=False)
            best_overlaps[image_id], best_matches[image_id] = overlaps.max(dim=1)

        threshold_values = thresholds.tolist()
        for image_id_value, local_index_value in zip(image_ids.tolist(), local_indices.tolist()):
            record = ground_truth.get(image_id_value)
            if record is None or len(record["boxes"]) == 0:
                for threshold_index in range(len(threshold_values)):
                    true_positive[threshold_index].append(0.0)
                    false_positive[threshold_index].append(1.0)
                continue

            overlap = float(best_overlaps[image_id_value][local_index_value])
            match = int(best_matches[image_id_value][local_index_value])
            is_difficult = bool(record["difficult"][match])
            image_detected = detected[image_id_value]
            for threshold_index, threshold in enumerate(threshold_values):
                if overlap < threshold:
                    true_positive[threshold_index].append(0.0)
                    false_positive[threshold_index].append(1.0)
                elif is_difficult:
                    # Official DOTA/VOC behaviour: a difficult match is ignored.
                    continue
                elif not bool(image_detected[threshold_index, match]):
                    image_detected[threshold_index, match] = True
                    true_positive[threshold_index].append(1.0)
                    false_positive[threshold_index].append(0.0)
                else:
                    true_positive[threshold_index].append(0.0)
                    false_positive[threshold_index].append(1.0)

        return true_positive, false_positive, positives

    @staticmethod
    def _ap_from_detections(true_positive, false_positive, positives, use_07_metric=False):
        if positives == 0:
            return np.nan
        if not true_positive:
            return 0.0
        tp = np.cumsum(np.asarray(true_positive))
        fp = np.cumsum(np.asarray(false_positive))
        recall = tp / max(positives, np.finfo(np.float64).eps)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        return _average_precision(recall, precision, use_07_metric)

    @staticmethod
    def _nanmean_or_zero(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0 or bool(np.isnan(values).all()):
            return 0.0
        return float(np.nanmean(values))

    def _class_aps(self, class_index, thresholds):
        true_positive, false_positive, positives = self._class_precision_recall_inputs(
            class_index, thresholds)
        if positives == 0:
            return np.full(len(thresholds), np.nan, dtype=np.float64)
        return np.asarray([
            self._ap_from_detections(tp, fp, positives, use_07_metric=False)
            for tp, fp in zip(true_positive, false_positive)
        ], dtype=np.float64)

    def _class_ap(self, class_index, threshold, use_07_metric=False):
        true_positive, false_positive, positives = self._class_precision_recall_inputs(
            class_index, [threshold])
        if positives == 0:
            return np.nan
        return self._ap_from_detections(
            true_positive[0], false_positive[0], positives, use_07_metric)

    def accumulate(self, verbose=True):
        class_count = len(self.dataset.classes)
        aps = np.full((class_count, len(self.iou_thresholds)), np.nan, dtype=np.float64)
        dota_ap50 = np.full(class_count, np.nan, dtype=np.float64)
        dota_ap75 = np.full(class_count, np.nan, dtype=np.float64)
        total_predictions = sum(len(prediction["scores"]) for prediction in self.predictions.values())
        start = perf_counter()
        if verbose:
            print(
                f"DOTA OBB evaluator: computing AP for {len(self.predictions)} images, "
                f"{total_predictions} predictions")
        threshold_50 = int(np.argmin(np.abs(self.iou_thresholds - 0.5)))
        threshold_75 = int(np.argmin(np.abs(self.iou_thresholds - 0.75)))
        for class_index in range(class_count):
            true_positive, false_positive, positives = self._class_precision_recall_inputs(
                class_index, self.iou_thresholds)
            if positives > 0:
                for threshold_index in range(len(self.iou_thresholds)):
                    aps[class_index, threshold_index] = self._ap_from_detections(
                        true_positive[threshold_index],
                        false_positive[threshold_index],
                        positives,
                        use_07_metric=False)
                dota_ap50[class_index] = self._ap_from_detections(
                    true_positive[threshold_50],
                    false_positive[threshold_50],
                    positives,
                    use_07_metric=self.use_07_metric)
                dota_ap75[class_index] = self._ap_from_detections(
                    true_positive[threshold_75],
                    false_positive[threshold_75],
                    positives,
                    use_07_metric=self.use_07_metric)
            if verbose:
                print(
                    f"  [{class_index + 1:02d}/{class_count:02d}] "
                    f"{self.dataset.classes[class_index]} AP50={dota_ap50[class_index]:.4f} "
                    f"AP75={dota_ap75[class_index]:.4f}")
        dota_map50_75 = self._nanmean_or_zero(np.stack((dota_ap50, dota_ap75), axis=1))
        self.stats = np.asarray([
            dota_map50_75,
            self._nanmean_or_zero(dota_ap50),
            self._nanmean_or_zero(dota_ap75),
            self._nanmean_or_zero(aps),
            self._nanmean_or_zero(aps[:, threshold_50]),
            self._nanmean_or_zero(aps[:, threshold_75]),
        ], dtype=np.float64)
        self.metrics = {
            "mAP50_75_DOTA07": float(self.stats[0]),
            "AP50_DOTA07": float(self.stats[1]),
            "AP75_DOTA07": float(self.stats[2]),
            "mAP50_95": float(self.stats[3]),
            "AP50": float(self.stats[4]),
            "AP75": float(self.stats[5]),
        }
        self.per_class = {
            name: float(dota_ap50[index]) if not np.isnan(dota_ap50[index]) else None
            for index, name in enumerate(self.dataset.classes)
        }
        self.per_class_metrics = {
            name: {
                "AP50_DOTA07": float(dota_ap50[index])
                    if not np.isnan(dota_ap50[index]) else None,
                "AP75_DOTA07": float(dota_ap75[index])
                    if not np.isnan(dota_ap75[index]) else None,
                "mAP50_95": self._nanmean_or_zero(aps[index]),
            }
            for index, name in enumerate(self.dataset.classes)
        }
        if verbose:
            print(f"DOTA OBB evaluator: AP done in {perf_counter() - start:.1f}s")

    def summarize(self):
        print("DOTA OBB evaluation (original-image coordinates)")
        print("  mAP50/75 DOTA-07 (best): {:.4f}".format(
            self.metrics.get("mAP50_75_DOTA07", 0.0)))
        print("  AP50 / AP75 DOTA-07    : {:.4f} / {:.4f}".format(
            self.metrics.get("AP50_DOTA07", 0.0),
            self.metrics.get("AP75_DOTA07", 0.0)))
        print("  mAP@[.50:.95] diagnostic: {:.4f}".format(
            self.metrics.get("mAP50_95", 0.0)))
        print("  AP50 / AP75 diagnostic  : {:.4f} / {:.4f}".format(
            self.metrics.get("AP50", 0.0), self.metrics.get("AP75", 0.0)))
        valid = [(name, value) for name, value in self.per_class.items() if value is not None]
        if valid:
            print("  per-class AP50 DOTA-07: " + ", ".join(f"{name}={value:.3f}" for name, value in valid))

    def export_dota_results(self, output_dir):
        """Write official Task1-style per-class result text files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        handles = {
            class_index: (output_dir / f"Task1_{name}.txt").open("w", encoding="utf-8")
            for class_index, name in enumerate(self.dataset.classes)
        }
        try:
            for image_id, prediction in sorted(self.predictions.items()):
                name = self.dataset.image_ids[image_id]
                corners = rbox_to_corners(prediction["boxes"]).reshape(-1, 8)
                for polygon, score, label in zip(corners, prediction["scores"], prediction["labels"]):
                    coordinates = " ".join(f"{float(value):.2f}" for value in polygon)
                    handles[int(label)].write(f"{name} {float(score):.6f} {coordinates}\n")
        finally:
            for handle in handles.values():
                handle.close()
        return output_dir


@register()
class MergedDotaOBBEvaluator(DotaOBBEvaluator):
    """Merge local-tile predictions and evaluate once on original images.

    Per-tile postprocessing is left unchanged. Its surviving boxes are
    translated by the recorded tile origin, concatenated per source image,
    and subjected to class-aware rotated NMS, matching the DOTA/MMRotate
    protocol used by ``ai4rs``. All ranks gather tile predictions before the
    merge so an original image remains the indivisible metric unit.
    """

    def __init__(
        self,
        dataset,
        iou_thresholds=None,
        use_07_metric=True,
        merge_iou_threshold=0.1,
        max_detections_per_image=None,
        source_root=None,
        require_all_tiles=True,
        record_merge_candidates=True,
    ):
        self.tile_dataset = dataset
        self.diagnostic_dataset = dataset
        self.merge_iou_threshold = float(merge_iou_threshold)
        self.max_detections_per_image = (
            None if max_detections_per_image is None else int(max_detections_per_image))
        self.require_all_tiles = bool(require_all_tiles)
        self.record_merge_candidates = bool(record_merge_candidates)
        if not 0.0 <= self.merge_iou_threshold <= 1.0:
            raise ValueError("merge_iou_threshold must be in [0, 1]")
        if self.max_detections_per_image is not None and self.max_detections_per_image <= 0:
            raise ValueError("max_detections_per_image must be positive or null")
        if not hasattr(dataset, "get_partition_metadata"):
            raise TypeError(
                "MergedDotaOBBEvaluator requires a partition-aware DOTA dataset "
                "with get_partition_metadata(index)")
        if not hasattr(dataset, "build_source_dataset"):
            raise TypeError(
                "MergedDotaOBBEvaluator requires the dataset adapter to implement "
                "build_source_dataset(source_root)")

        embedded_manifest = getattr(dataset, "tiling_manifest", None)
        manifest_path = Path(dataset.root) / "manifest.json"
        if embedded_manifest is not None:
            self.tiling_manifest = dict(embedded_manifest)
        elif manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                self.tiling_manifest = json.load(handle)
        else:
            raise FileNotFoundError(
                f"Merged evaluation requires the tiling manifest: {manifest_path}")
        if self.tiling_manifest.get("status") != "complete":
            raise RuntimeError(f"Tiling output is incomplete: {manifest_path}")
        source_root = (
            source_root
            or getattr(dataset, "source_root", None)
            or self.tiling_manifest.get("source_split"))
        if not source_root:
            raise ValueError("source_root is absent from both evaluator config and tiling manifest")
        source_dataset = dataset.build_source_dataset(source_root)
        if tuple(source_dataset.classes) != tuple(dataset.classes):
            raise ValueError("Tile/source class orders do not match")
        self.source_index_by_name = {
            name: index for index, name in enumerate(source_dataset.image_ids)}
        super().__init__(
            source_dataset,
            iou_thresholds=iou_thresholds,
            use_07_metric=use_07_metric,
        )

    def cleanup(self):
        super().cleanup()
        self.tile_predictions = {}
        self.merge_candidate_records = []
        self.merge_image_records = []
        self.merge_summary = {}
        self.final_prediction_metadata = {}
        self._tile_predictions_merged = False

    @staticmethod
    def _cpu_prediction(prediction):
        required = {"boxes", "scores", "labels"}
        missing = required - set(prediction)
        if missing:
            raise KeyError(f"Tile prediction is missing fields: {sorted(missing)}")
        result = {
            "boxes": prediction["boxes"].detach().float().cpu(),
            "scores": prediction["scores"].detach().float().cpu(),
            "labels": prediction["labels"].detach().long().cpu(),
        }
        count = len(result["boxes"])
        if result["boxes"].shape != (count, 5):
            raise ValueError(f"Expected tile boxes shaped (N, 5), got {result['boxes'].shape}")
        if len(result["scores"]) != count or len(result["labels"]) != count:
            raise ValueError("Tile prediction boxes/scores/labels have inconsistent lengths")
        return result

    def update(self, predictions):
        if self._tile_predictions_merged:
            raise RuntimeError("Cannot update a merged evaluator before cleanup()")
        for image_id, prediction in predictions.items():
            image_id = int(image_id)
            if image_id < 0 or image_id >= len(self.tile_dataset):
                raise IndexError(f"Tile image id is out of range: {image_id}")
            self.tile_predictions[image_id] = self._cpu_prediction(prediction)

    def synchronize_between_processes(self):
        gathered = dist_utils.all_gather(self.tile_predictions)
        self.tile_predictions = {}
        for rank_predictions in gathered:
            self.tile_predictions.update(rank_predictions)

    def _global_nms_trace(self, boxes, scores, labels):
        keep_all = class_aware_rotated_nms(
            boxes, scores, labels, self.merge_iou_threshold, max_output=None)
        keep = keep_all
        if self.max_detections_per_image is not None:
            keep = keep_all[:self.max_detections_per_image]
        status = labels.new_full((len(labels),), 1)
        parent = labels.new_full((len(labels),), -1)
        parent_iou = scores.new_zeros((len(labels),))
        status[keep] = 0
        if len(keep) < len(keep_all):
            status[keep_all[len(keep):]] = 2

        for label in labels.unique(sorted=True):
            suppressed = torch.nonzero(
                (labels == label) & (status == 1), as_tuple=False).squeeze(1)
            retained = keep_all[labels[keep_all] == label]
            if not len(suppressed) or not len(retained):
                continue
            overlaps = rotated_iou(
                boxes[suppressed], boxes[retained], model_space=False)
            higher_score = scores[retained][None, :] >= scores[suppressed][:, None]
            eligible = higher_score & (overlaps > self.merge_iou_threshold)
            has_parent = eligible.any(dim=1)
            first_parent = eligible.to(torch.int8).argmax(dim=1)
            parent[suppressed[has_parent]] = retained[first_parent[has_parent]]
            parent_iou[suppressed[has_parent]] = overlaps[
                has_parent, first_parent[has_parent]]
        return keep, status, parent, parent_iou

    def _source_gt_matches(self, source_index, boxes, labels):
        best_iou = boxes.new_zeros(len(boxes))
        best_gt = labels.new_full((len(labels),), -1)
        best_source_object = labels.new_full((len(labels),), -1)
        best_gt_box = boxes.new_zeros((len(boxes), 5))
        annotation = self.dataset.get_ground_truth(source_index)
        for label in labels.unique(sorted=True):
            candidates = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
            gt_indices = torch.nonzero(
                annotation["labels"] == label, as_tuple=False).squeeze(1)
            if not len(candidates) or not len(gt_indices):
                continue
            overlaps = rotated_iou(
                boxes[candidates], annotation["boxes"][gt_indices].float(),
                model_space=False)
            values, local_indices = overlaps.max(dim=1)
            best_iou[candidates] = values
            best_gt[candidates] = gt_indices[local_indices]
            source_object_index = annotation.get(
                "source_object_index",
                torch.arange(len(annotation["boxes"]), dtype=torch.int64),
            )
            best_source_object[candidates] = source_object_index[gt_indices[local_indices]]
            best_gt_box[candidates] = annotation["boxes"][gt_indices[local_indices]]
        return best_iou, best_gt, best_source_object, best_gt_box

    @staticmethod
    def _candidate_boundary(local_boxes, tile_size):
        if not len(local_boxes):
            return local_boxes.new_empty(0), local_boxes.new_empty(0)
        width, height = float(tile_size[0]), float(tile_size[1])
        center = torch.stack((
            local_boxes[:, 0], local_boxes[:, 1],
            width - local_boxes[:, 0], height - local_boxes[:, 1],
        ), dim=1).min(dim=1).values
        corners = rbox_to_corners(local_boxes, normalized_angle=False)
        support = torch.stack((
            corners[..., 0].min(dim=1).values,
            corners[..., 1].min(dim=1).values,
            width - corners[..., 0].max(dim=1).values,
            height - corners[..., 1].max(dim=1).values,
        ), dim=1).min(dim=1).values
        return center, support

    def _merge_tile_predictions(self):
        if self._tile_predictions_merged:
            return
        if self.require_all_tiles:
            missing = sorted(set(range(len(self.tile_dataset))) - set(self.tile_predictions))
            if missing:
                preview = ", ".join(map(str, missing[:8]))
                raise RuntimeError(
                    f"Merged evaluation is missing {len(missing)} tile predictions: {preview}")

        collectors = defaultdict(lambda: {
            "boxes": [], "local_boxes": [], "scores": [], "labels": [],
            "tile_indices": [], "tile_detection_indices": [], "tile_ids": [],
            "tile_origins": [], "tile_sizes": [], "tile_count": 0,
        })
        for tile_index in sorted(self.tile_predictions):
            prediction = self.tile_predictions[tile_index]
            metadata = self.tile_dataset.get_partition_metadata(tile_index)
            source_name = metadata["source_image_id"]
            if source_name not in self.source_index_by_name:
                raise KeyError(
                    f"Tile {tile_index} refers to unknown source image {source_name!r}")
            source_index = self.source_index_by_name[source_name]
            local_boxes = prediction["boxes"]
            global_boxes = local_boxes.clone()
            if len(global_boxes):
                origin = torch.as_tensor(metadata["tile_origin"], dtype=global_boxes.dtype)
                global_boxes[:, :2] += origin
            count = len(global_boxes)
            collector = collectors[source_index]
            collector["tile_count"] += 1
            collector["boxes"].append(global_boxes)
            collector["local_boxes"].append(local_boxes)
            collector["scores"].append(prediction["scores"])
            collector["labels"].append(prediction["labels"])
            collector["tile_indices"].extend([tile_index] * count)
            collector["tile_detection_indices"].extend(range(count))
            collector["tile_ids"].extend([metadata["tile_id"]] * count)
            collector["tile_origins"].extend([metadata["tile_origin"]] * count)
            collector["tile_sizes"].extend([metadata["tile_size"]] * count)

        self.predictions = {}
        merge_started = perf_counter()
        total_candidates = total_kept = total_suppressed = total_limited = 0
        total_cross_tile = 0
        total_same_gt = total_different_gt = total_unmatched_gt = 0
        for source_index in range(len(self.dataset)):
            source_started = perf_counter()
            collector = collectors.get(source_index)
            if collector is None:
                boxes = torch.empty((0, 5), dtype=torch.float32)
                scores = torch.empty(0, dtype=torch.float32)
                labels = torch.empty(0, dtype=torch.long)
                local_boxes = boxes.clone()
                tile_indices, tile_detection_indices = [], []
                tile_ids, tile_origins, tile_sizes = [], [], []
                source_tile_count = 0
            else:
                boxes = torch.cat(collector["boxes"], dim=0)
                local_boxes = torch.cat(collector["local_boxes"], dim=0)
                scores = torch.cat(collector["scores"], dim=0)
                labels = torch.cat(collector["labels"], dim=0)
                tile_indices = collector["tile_indices"]
                tile_detection_indices = collector["tile_detection_indices"]
                tile_ids = collector["tile_ids"]
                tile_origins = collector["tile_origins"]
                tile_sizes = collector["tile_sizes"]
                source_tile_count = collector["tile_count"]

            if len(boxes):
                keep, status, parent, parent_iou = self._global_nms_trace(
                    boxes, scores, labels)
                if self.record_merge_candidates:
                    best_gt_iou, best_gt, best_source_object, best_gt_box = \
                        self._source_gt_matches(source_index, boxes, labels)
            else:
                keep = labels.new_empty(0)
                status = labels.new_empty(0)
                parent = labels.new_empty(0)
                parent_iou = scores.new_empty(0)
                if self.record_merge_candidates:
                    best_gt_iou = scores.new_empty(0)
                    best_gt = labels.new_empty(0)
                    best_source_object = labels.new_empty(0)
                    best_gt_box = boxes.new_empty((0, 5))
            self.predictions[source_index] = {
                "boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep],
            }
            keep_list = keep.tolist()
            self.final_prediction_metadata[source_index] = {
                "global_candidate_indices": keep.clone(),
                "tile_image_indices": torch.as_tensor(
                    [tile_indices[index] for index in keep_list], dtype=torch.int64),
                "tile_detection_indices": torch.as_tensor(
                    [tile_detection_indices[index] for index in keep_list], dtype=torch.int64),
                "tile_ids": [tile_ids[index] for index in keep_list],
                "tile_origins": (
                    torch.stack([
                        torch.as_tensor(tile_origins[index], dtype=torch.int64)
                        for index in keep_list
                    ]).reshape(-1, 2)
                    if keep_list else torch.empty((0, 2), dtype=torch.int64)),
                "tile_sizes": (
                    torch.stack([
                        torch.as_tensor(tile_sizes[index], dtype=torch.int64)
                        for index in keep_list
                    ]).reshape(-1, 2)
                    if keep_list else torch.empty((0, 2), dtype=torch.int64)),
                "local_boxes": local_boxes[keep].clone(),
            }

            source_name = self.dataset.image_ids[source_index]
            source_image_path = str(self.dataset.images[source_index])
            if self.record_merge_candidates:
                for candidate_index in range(len(boxes)):
                    parent_index = int(parent[candidate_index])
                    candidate_center, candidate_support = self._candidate_boundary(
                        local_boxes[candidate_index:candidate_index + 1],
                        tile_sizes[candidate_index])
                    cross_tile = (
                        parent_index >= 0
                        and tile_indices[parent_index] != tile_indices[candidate_index])
                    total_cross_tile += cross_tile
                    gt_index = int(best_gt[candidate_index])
                    source_object_index = int(best_source_object[candidate_index])
                    candidate_tp50 = gt_index >= 0 and float(best_gt_iou[candidate_index]) >= 0.5
                    suppressor_gt_index = int(best_gt[parent_index]) if parent_index >= 0 else -1
                    suppressor_source_object_index = (
                        int(best_source_object[parent_index]) if parent_index >= 0 else -1)
                    suppressor_tp50 = (
                        parent_index >= 0 and suppressor_gt_index >= 0
                        and float(best_gt_iou[parent_index]) >= 0.5)
                    same_gt_suppression = (
                        parent_index >= 0 and candidate_tp50 and suppressor_tp50
                        and source_object_index == suppressor_source_object_index)
                    different_gt_suppression = (
                        parent_index >= 0 and candidate_tp50 and suppressor_tp50
                        and source_object_index != suppressor_source_object_index)
                    unmatched_gt_suppression = (
                        parent_index >= 0
                        and not same_gt_suppression and not different_gt_suppression)
                    total_same_gt += same_gt_suppression
                    total_different_gt += different_gt_suppression
                    total_unmatched_gt += unmatched_gt_suppression
                    self.merge_candidate_records.append({
                    "source_image_id": source_name,
                    "source_image_index": source_index,
                    "global_candidate_index": candidate_index,
                    "tile_image_index": tile_indices[candidate_index],
                    "tile_id": tile_ids[candidate_index],
                    "tile_origin": tile_origins[candidate_index],
                    "tile_size": tile_sizes[candidate_index],
                    "tile_detection_index": tile_detection_indices[candidate_index],
                    "source_image_path": source_image_path,
                    "local_box": local_boxes[candidate_index],
                    "global_box": boxes[candidate_index],
                    "score": scores[candidate_index],
                    "label": int(labels[candidate_index]),
                    "class_name": self.dataset.classes[int(labels[candidate_index])],
                    "candidate_center_boundary_distance_px": candidate_center[0],
                    "candidate_support_boundary_distance_px": candidate_support[0],
                    "status": ("kept" if int(status[candidate_index]) == 0
                               else "max_detections" if int(status[candidate_index]) == 2
                               else "global_nms_overlap"),
                    "suppressed_by_global_candidate_index": (
                        parent_index if parent_index >= 0 else None),
                    "suppressor_tile_id": tile_ids[parent_index] if parent_index >= 0 else None,
                    "suppression_iou": parent_iou[candidate_index],
                    "cross_tile_suppression": cross_tile,
                    "best_source_gt_index": gt_index if gt_index >= 0 else None,
                    "best_source_gt_iou": best_gt_iou[candidate_index],
                    "best_source_gt_box": (
                        best_gt_box[candidate_index] if gt_index >= 0 else None),
                    "source_object_index": (
                        source_object_index if candidate_tp50 else None),
                    "source_object_uid": (
                        f"{source_name}:{source_object_index}"
                        if candidate_tp50 else None),
                    "source_tp50": candidate_tp50,
                    "suppressor_best_source_gt_index": (
                        suppressor_gt_index if suppressor_tp50 else None),
                    "suppressor_best_source_gt_iou": (
                        best_gt_iou[parent_index] if parent_index >= 0 else None),
                    "suppressor_best_source_gt_box": (
                        best_gt_box[parent_index] if suppressor_tp50 else None),
                    "suppressor_source_object_index": (
                        suppressor_source_object_index if suppressor_tp50 else None),
                    "suppressor_source_object_uid": (
                        f"{source_name}:{suppressor_source_object_index}"
                        if suppressor_tp50 else None),
                    "same_best_gt_suppression": same_gt_suppression,
                    "different_best_gt_suppression": different_gt_suppression,
                    "unmatched_best_gt_suppression": unmatched_gt_suppression,
                    })
            else:
                for candidate_index in torch.nonzero(
                        parent >= 0, as_tuple=False).squeeze(1).tolist():
                    parent_index = int(parent[candidate_index])
                    total_cross_tile += (
                        tile_indices[parent_index] != tile_indices[candidate_index])

            kept_count = int((status == 0).sum()) if len(status) else 0
            suppressed_count = int((status == 1).sum()) if len(status) else 0
            limited_count = int((status == 2).sum()) if len(status) else 0
            total_candidates += len(boxes)
            total_kept += kept_count
            total_suppressed += suppressed_count
            total_limited += limited_count
            self.merge_image_records.append({
                "source_image_id": source_name,
                "source_image_index": source_index,
                "source_image_path": source_image_path,
                "tile_count": source_tile_count,
                "candidate_count": len(boxes),
                "kept_count": kept_count,
                "global_nms_suppressed_count": suppressed_count,
                "max_detection_limited_count": limited_count,
                "merge_ms": (perf_counter() - source_started) * 1000.0,
            })

        self.merge_summary = {
            "source_images": len(self.dataset),
            "tile_predictions": len(self.tile_predictions),
            "candidate_count_after_local_nms": total_candidates,
            "kept_after_global_nms": total_kept,
            "global_nms_suppressed": total_suppressed,
            "max_detection_limited": total_limited,
            "cross_tile_suppressions": total_cross_tile,
            "same_best_gt_suppressions": (
                total_same_gt if self.record_merge_candidates else None),
            "different_best_gt_suppressions": (
                total_different_gt if self.record_merge_candidates else None),
            "unmatched_best_gt_suppressions": (
                total_unmatched_gt if self.record_merge_candidates else None),
            "merge_candidate_trace_recorded": self.record_merge_candidates,
            "merge_iou_threshold": self.merge_iou_threshold,
            "max_detections_per_image": self.max_detections_per_image,
            "merge_total_ms": (perf_counter() - merge_started) * 1000.0,
            "tiling_protocol": self.tiling_manifest.get("protocol", {}),
            "source_inventory_sha256": self.tiling_manifest.get("source_inventory_sha256"),
        }
        self._tile_predictions_merged = True

    def accumulate(self, verbose=True):
        self._merge_tile_predictions()
        super().accumulate(verbose=verbose)

    def summarize(self):
        super().summarize()
        if self.merge_summary:
            print(
                "  tile merge candidates/kept/global-NMS: "
                f"{self.merge_summary['candidate_count_after_local_nms']} / "
                f"{self.merge_summary['kept_after_global_nms']} / "
                f"{self.merge_summary['global_nms_suppressed']}")
