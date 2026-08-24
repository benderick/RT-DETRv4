"""Decode and restore D-FINE OBB predictions with an optional rotated NMS."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core import register
from .rotated_box_ops import class_aware_rotated_nms, regularize_rboxes, rotated_iou


@register()
class RotatedPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries"]

    def __init__(self, num_classes=12, use_focal_loss=True, num_top_queries=1000,
                 score_threshold=0.05, nms_iou_threshold=0.1, max_detections=500,
                 apply_nms=True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = int(num_top_queries)
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)
        self.apply_nms = bool(apply_nms)
        self.deploy_mode = False

    @staticmethod
    def _metadata(target_info, device, dtype):
        if isinstance(target_info, (list, tuple)) and target_info and isinstance(target_info[0], dict):
            canvas = torch.stack([target["size"] for target in target_info]).to(device=device, dtype=dtype)
            scale = torch.stack([target["scale_factor"] for target in target_info]).to(device=device, dtype=dtype)
            padding = torch.stack([target["padding"] for target in target_info]).to(device=device, dtype=dtype)
        else:
            canvas = torch.as_tensor(target_info, device=device, dtype=dtype)
            scale = torch.ones_like(canvas)
            padding = torch.zeros((len(canvas), 4), device=device, dtype=dtype)
        return canvas, scale, padding

    def restore_boxes(self, normalized_boxes, target_info):
        """Restore normalized canvas boxes to original-image pixel space."""
        canvas, scale, padding = self._metadata(
            target_info, normalized_boxes.device, normalized_boxes.dtype)
        boxes = normalized_boxes.clone()
        boxes[..., 0] *= canvas[:, None, 0]
        boxes[..., 1] *= canvas[:, None, 1]
        boxes[..., 2] *= canvas[:, None, 0]
        boxes[..., 3] *= canvas[:, None, 1]
        boxes[..., 4] *= math.pi
        boxes[..., 0] = (boxes[..., 0] - padding[:, None, 0]) / scale[:, None, 0]
        boxes[..., 1] = (boxes[..., 1] - padding[:, None, 1]) / scale[:, None, 1]
        length_scale = scale.mean(dim=1, keepdim=True)
        boxes[..., 2:4] /= length_scale.unsqueeze(-1)
        return regularize_rboxes(boxes)

    def _nms_trace(self, boxes, scores, labels):
        """Return NMS decisions and the retained box that suppressed each box.

        Status codes are stable on disk: 0=kept, 1=nms_overlap,
        2=max_detections.  The expensive overlap trace is only requested by
        diagnostic evaluation; normal inference keeps the original fast path.
        """
        keep_all = class_aware_rotated_nms(
            boxes, scores, labels, self.nms_iou_threshold, max_output=None)
        keep = keep_all[:self.max_detections]
        status = labels.new_full((len(labels),), 1)
        parent = labels.new_full((len(labels),), -1)
        parent_iou = scores.new_zeros((len(labels),))
        if len(keep):
            status[keep] = 0
        if len(keep_all) > self.max_detections:
            status[keep_all[self.max_detections:]] = 2

        # NMS processes retained boxes in descending-score order. Attribute a
        # deleted candidate to the first eligible retained box in that order,
        # rather than to the largest-IoU box (which may be encountered later
        # and therefore is not the causal suppressor).
        for label in labels.unique(sorted=True):
            candidates = torch.nonzero((labels == label) & (status == 1), as_tuple=False).squeeze(1)
            retained = keep_all[labels[keep_all] == label]
            if not len(candidates) or not len(retained):
                continue
            overlaps = rotated_iou(
                boxes[candidates], boxes[retained], normalized_angle=False)
            higher_score = scores[retained][None, :] >= scores[candidates][:, None]
            eligible = higher_score & (overlaps >= self.nms_iou_threshold)
            has_parent = eligible.any(dim=1)
            first_parent = eligible.to(torch.int8).argmax(dim=1)
            parent[candidates[has_parent]] = retained[first_parent[has_parent]]
            parent_iou[candidates[has_parent]] = overlaps[
                has_parent, first_parent[has_parent]]
        return keep, keep_all, status, parent, parent_iou

    def forward(
        self,
        outputs,
        target_info,
        return_diagnostics=False,
        apply_nms=None,
    ):
        """Restore predictions and select the final score-sorted candidates.

        ``apply_nms=False`` follows DETR's set-prediction post-processing: it
        retains the highest-scoring candidates up to ``max_detections`` but
        performs no overlap suppression.  Keeping this a per-call choice lets
        diagnostics compare the two policies on exactly the same frozen model
        outputs without silently changing the public inference default.
        """
        apply_nms = self.apply_nms if apply_nms is None else bool(apply_nms)
        logits, normalized_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        probabilities = logits.sigmoid() if self.use_focal_loss else logits.softmax(-1)
        top_count = min(self.num_top_queries, probabilities.shape[1] * probabilities.shape[2])
        scores, flat_indices = torch.topk(probabilities.flatten(1), top_count, dim=1)
        labels = flat_indices % self.num_classes
        query_indices = flat_indices // self.num_classes
        query_boxes = self.restore_boxes(normalized_boxes, target_info)
        boxes = query_boxes.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 5))

        results = []
        diagnostics = []
        for batch_index, (image_boxes, image_scores, image_labels, image_queries) in enumerate(
                zip(boxes, scores, labels, query_indices)):
            valid = image_scores >= self.score_threshold
            image_boxes, image_scores, image_labels, image_queries = (
                image_boxes[valid], image_scores[valid], image_labels[valid], image_queries[valid])
            if len(image_boxes) and apply_nms and return_diagnostics:
                keep, keep_all, status, parent, parent_iou = self._nms_trace(
                    image_boxes, image_scores, image_labels)
            elif len(image_boxes) and apply_nms:
                keep = class_aware_rotated_nms(
                    image_boxes, image_scores, image_labels, self.nms_iou_threshold,
                    self.max_detections)
            elif len(image_boxes):
                # ``torch.topk`` above already returned candidates in
                # descending score order.  This is the exact NMS-free control:
                # same logits, threshold, candidate budget, and coordinate
                # restoration, with only overlap suppression removed.
                keep_all = torch.arange(len(image_boxes), device=image_boxes.device)
                keep = keep_all[:self.max_detections]
                status = image_labels.new_full((len(image_labels),), 2)
                status[keep] = 0
                parent = image_labels.new_full((len(image_labels),), -1)
                parent_iou = image_scores.new_zeros((len(image_scores),))
            else:
                keep = image_labels.new_empty(0)
                keep_all = keep
                status = image_labels.new_empty(0)
                parent = image_labels.new_empty(0)
                parent_iou = image_scores.new_empty(0)
            results.append({"boxes": image_boxes[keep], "scores": image_scores[keep],
                            "labels": image_labels[keep]})
            if return_diagnostics:
                diagnostics.append({
                    "query_boxes": query_boxes[batch_index],
                    "probabilities": probabilities[batch_index],
                    "pre_nms_boxes": image_boxes,
                    "pre_nms_scores": image_scores,
                    "pre_nms_labels": image_labels,
                    "pre_nms_query_indices": image_queries,
                    "keep_indices": keep,
                    "nms_keep_all_indices": keep_all,
                    "status": status,
                    "suppressed_by": parent,
                    "suppression_iou": parent_iou,
                    "topk_count": int(top_count),
                    "score_filtered_count": int(valid.sum()),
                    "flat_candidate_count": int(probabilities.shape[1] * probabilities.shape[2]),
                    "score_threshold": self.score_threshold,
                    "apply_nms": bool(apply_nms),
                    "topk_min_score": scores[batch_index, -1] if top_count else None,
                })
        if self.deploy_mode:
            raise RuntimeError("Rotated NMS returns variable-length results; use normal inference mode")
        return (results, diagnostics) if return_diagnostics else results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
