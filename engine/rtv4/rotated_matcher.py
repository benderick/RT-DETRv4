"""Hungarian matching for normalized oriented boxes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..core import register
from .rotated_box_ops import angle_distance, pairwise_chamfer_cost, pairwise_kld_cost


@register()
class RotatedHungarianMatcher(nn.Module):
    __share__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=True, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weights = {
            "class": weight_dict.get("cost_class", 2.0),
            "bbox": weight_dict.get("cost_bbox", 5.0),
            "angle": weight_dict.get("cost_angle", 2.0),
            "kld": weight_dict.get("cost_kld", 2.0),
            "chamfer": weight_dict.get("cost_chamfer", 0.5),
        }
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        if not any(self.weights.values()):
            raise ValueError("At least one rotated matching cost must be non-zero")

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False):
        if return_topk:
            raise NotImplementedError("One-to-many matching is not used by the OBB baseline")
        probabilities = outputs["pred_logits"].sigmoid() if self.use_focal_loss \
            else outputs["pred_logits"].softmax(-1)
        predicted_boxes = outputs["pred_boxes"]
        result = []
        for probability, prediction, target in zip(probabilities, predicted_boxes, targets):
            target_boxes, target_labels = target["boxes"], target["labels"]
            if len(target_boxes) == 0 or len(prediction) == 0:
                empty = torch.empty(0, dtype=torch.int64)
                result.append((empty, empty.clone()))
                continue
            if self.use_focal_loss:
                selected = probability[:, target_labels]
                negative = (1 - self.alpha) * selected.pow(self.gamma) * (-(1 - selected + 1e-8).log())
                positive = self.alpha * (1 - selected).pow(self.gamma) * (-(selected + 1e-8).log())
                class_cost = positive - negative
            else:
                class_cost = -probability[:, target_labels]
            bbox_cost = torch.cdist(prediction[:, :4], target_boxes[:, :4], p=1)
            angle_cost = angle_distance(
                prediction[:, None, 4], target_boxes[None, :, 4], normalized=True)
            cost = self.weights["class"] * class_cost
            cost += self.weights["bbox"] * bbox_cost
            cost += self.weights["angle"] * angle_cost
            if self.weights["kld"]:
                cost += self.weights["kld"] * pairwise_kld_cost(prediction, target_boxes)
            if self.weights["chamfer"]:
                cost += self.weights["chamfer"] * pairwise_chamfer_cost(prediction, target_boxes)
            row, column = linear_sum_assignment(torch.nan_to_num(cost, nan=1e6).cpu())
            result.append((torch.as_tensor(row, dtype=torch.int64),
                           torch.as_tensor(column, dtype=torch.int64)))
        return {"indices": result}
