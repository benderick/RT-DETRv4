"""DOTA-style oriented-box evaluator for CODrone."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from time import perf_counter

from ...core import register
from ...misc import dist_utils
from ...rtv4.rotated_box_ops import rotated_iou
from ...rtv4.rotated_box_ops import rbox_to_corners


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
class CODroneEvaluator:
    """Evaluate original-image pixel OBBs, respecting DOTA difficulty flags."""

    def __init__(self, dataset, iou_thresholds=None, use_07_metric=True):
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
        # stats[0] is consumed by DetSolver as the checkpoint-selection metric.
        # Keep it aligned with CODrone/MMRotate DOTAMetric(iou_thrs=[0.5, 0.75]):
        # DOTA/VOC AP averaged over AP50 and AP75, not COCO AP@[.50:.95].
        self.stats = np.zeros(6, dtype=np.float64)

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
                detections["boxes"], record["boxes"], normalized_angle=False)
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

    def accumulate(self):
        class_count = len(self.dataset.classes)
        aps = np.full((class_count, len(self.iou_thresholds)), np.nan, dtype=np.float64)
        dota_ap50 = np.full(class_count, np.nan, dtype=np.float64)
        dota_ap75 = np.full(class_count, np.nan, dtype=np.float64)
        total_predictions = sum(len(prediction["scores"]) for prediction in self.predictions.values())
        start = perf_counter()
        print(
            f"CODrone OBB evaluator: computing AP for {len(self.predictions)} images, "
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
            print(
                f"  [{class_index + 1:02d}/{class_count:02d}] "
                f"{self.dataset.classes[class_index]} AP50={dota_ap50[class_index]:.4f} "
                f"AP75={dota_ap75[class_index]:.4f}")
        codrone_map50_75 = self._nanmean_or_zero(np.stack((dota_ap50, dota_ap75), axis=1))
        self.stats = np.asarray([
            codrone_map50_75,
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
        print(f"CODrone OBB evaluator: AP done in {perf_counter() - start:.1f}s")

    def summarize(self):
        print("CODrone OBB evaluation (original-image coordinates)")
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
