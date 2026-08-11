"""Decode, restore, and rotated-NMS D-FINE OBB predictions."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..core import register
from .rotated_box_ops import class_aware_rotated_nms, regularize_rboxes


@register()
class RotatedPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries"]

    def __init__(self, num_classes=12, use_focal_loss=True, num_top_queries=1000,
                 score_threshold=0.05, nms_iou_threshold=0.1, max_detections=500):
        super().__init__()
        self.num_classes = int(num_classes)
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = int(num_top_queries)
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)
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

    def forward(self, outputs, target_info):
        logits, normalized_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        canvas, scale, padding = self._metadata(target_info, logits.device, normalized_boxes.dtype)
        probabilities = logits.sigmoid() if self.use_focal_loss else logits.softmax(-1)
        top_count = min(self.num_top_queries, probabilities.shape[1] * probabilities.shape[2])
        scores, flat_indices = torch.topk(probabilities.flatten(1), top_count, dim=1)
        labels = flat_indices % self.num_classes
        query_indices = flat_indices // self.num_classes
        boxes = normalized_boxes.gather(1, query_indices.unsqueeze(-1).expand(-1, -1, 5)).clone()
        boxes[..., 0] *= canvas[:, None, 0]
        boxes[..., 1] *= canvas[:, None, 1]
        boxes[..., 2] *= canvas[:, None, 0]
        boxes[..., 3] *= canvas[:, None, 1]
        boxes[..., 4] *= math.pi
        boxes[..., 0] = (boxes[..., 0] - padding[:, None, 0]) / scale[:, None, 0]
        boxes[..., 1] = (boxes[..., 1] - padding[:, None, 1]) / scale[:, None, 1]
        length_scale = scale.mean(dim=1, keepdim=True)
        boxes[..., 2:4] /= length_scale.unsqueeze(-1)
        boxes = regularize_rboxes(boxes)

        results = []
        for image_boxes, image_scores, image_labels in zip(boxes, scores, labels):
            valid = image_scores >= self.score_threshold
            image_boxes, image_scores, image_labels = (
                image_boxes[valid], image_scores[valid], image_labels[valid])
            keep = class_aware_rotated_nms(
                image_boxes, image_scores, image_labels, self.nms_iou_threshold,
                self.max_detections) if len(image_boxes) else image_labels.new_empty(0)
            results.append({"boxes": image_boxes[keep], "scores": image_scores[keep],
                            "labels": image_labels[keep]})
        if self.deploy_mode:
            raise RuntimeError("Rotated NMS returns variable-length results; use normal inference mode")
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
