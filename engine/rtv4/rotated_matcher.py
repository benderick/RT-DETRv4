"""Hungarian matching for normalized oriented boxes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ..core import register
from .rotated_box_ops import (
    CHAMFER_DISTANCE_MODES,
    angle_distance,
    pairwise_chamfer_cost,
    pairwise_kld_cost,
)


@register()
class RotatedHungarianMatcher(nn.Module):
    __share__ = ["use_focal_loss"]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
        chamfer_distance="paper_squared",
        kld_sqrt=False,
        kld_fun="log1p",
        kld_tau=1.0,
    ):
        super().__init__()
        self.weights = {
            "class": weight_dict.get("cost_class", 2.0),
            "bbox": weight_dict.get("cost_bbox", 0.0),
            "angle": weight_dict.get("cost_angle", 0.0),
            "kld": weight_dict.get("cost_kld", 2.0),
            "chamfer": weight_dict.get("cost_chamfer", 5.0),
        }
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        if chamfer_distance not in CHAMFER_DISTANCE_MODES:
            raise ValueError(
                f"Unknown Chamfer distance {chamfer_distance!r}; "
                f"expected one of {CHAMFER_DISTANCE_MODES}")
        if kld_fun not in {"log1p", "sqrt", "none"}:
            raise ValueError(f"Unsupported KLD post-processing function: {kld_fun!r}")
        self.chamfer_distance = chamfer_distance
        self.kld_sqrt = bool(kld_sqrt)
        self.kld_fun = str(kld_fun)
        self.kld_tau = float(kld_tau)
        if not any(self.weights.values()):
            raise ValueError("At least one rotated matching cost must be non-zero")

    def _cost_components(self, probability, prediction, target):
        """Build the unweighted matching terms once.

        Keeping the individual terms available is important for diagnostics:
        a large final Hungarian cost alone cannot tell whether classification,
        centre/scale regression, or orientation caused the assignment.
        """
        target_boxes, target_labels = target["boxes"], target["labels"]
        if self.use_focal_loss:
            selected = probability[:, target_labels]
            negative = (1 - self.alpha) * selected.pow(self.gamma) * (-(1 - selected + 1e-8).log())
            positive = self.alpha * (1 - selected).pow(self.gamma) * (-(selected + 1e-8).log())
            class_cost = positive - negative
        else:
            class_cost = -probability[:, target_labels]
        components = {
            "class": class_cost,
            "bbox": torch.cdist(prediction[:, :4], target_boxes[:, :4], p=1),
            "angle": angle_distance(
                prediction[:, None, 4], target_boxes[None, :, 4], normalized=True),
        }
        if self.weights["kld"]:
            components["kld"] = pairwise_kld_cost(
                prediction,
                target_boxes,
                sqrt=self.kld_sqrt,
                fun=self.kld_fun,
                tau=self.kld_tau,
            )
        if self.weights["chamfer"]:
            components["chamfer"] = pairwise_chamfer_cost(
                prediction, target_boxes, distance_mode=self.chamfer_distance)
        return components

    @torch.no_grad()
    def forward(self, outputs, targets, return_topk=False, return_costs=False):
        if return_topk:
            raise NotImplementedError("One-to-many matching is not used by the OBB baseline")
        probabilities = outputs["pred_logits"].sigmoid() if self.use_focal_loss \
            else outputs["pred_logits"].softmax(-1)
        predicted_boxes = outputs["pred_boxes"]
        result = []
        matched_costs = []
        candidate_costs = []
        for probability, prediction, target in zip(probabilities, predicted_boxes, targets):
            target_boxes, target_labels = target["boxes"], target["labels"]
            if len(target_boxes) == 0 or len(prediction) == 0:
                empty = torch.empty(0, dtype=torch.int64)
                result.append((empty, empty.clone()))
                matched_costs.append({
                    name: prediction.new_empty(0)
                    for name in (*self.weights.keys(), "total")
                })
                candidate_costs.append({
                    "query_indices": torch.empty((len(target_boxes), 0), dtype=torch.long),
                    **{name: prediction.new_empty((len(target_boxes), 0))
                       for name in (*self.weights.keys(), "total")},
                })
                continue
            components = self._cost_components(probability, prediction, target)
            cost = sum(self.weights[name] * value for name, value in components.items())
            row, column = linear_sum_assignment(torch.nan_to_num(cost, nan=1e6).cpu())
            row = torch.as_tensor(row, dtype=torch.int64)
            column = torch.as_tensor(column, dtype=torch.int64)
            result.append((row, column))
            if return_costs:
                device_row = row.to(cost.device)
                device_column = column.to(cost.device)
                selected_costs = {
                    name: value[device_row, device_column].detach()
                    for name, value in components.items()
                }
                # Zero-weight terms are still explicit in the schema, which
                # makes runs with different matcher configurations comparable.
                for name in self.weights:
                    selected_costs.setdefault(name, cost.new_zeros(len(row)))
                selected_costs["total"] = cost[device_row, device_column].detach()
                matched_costs.append(selected_costs)
                top_count = min(5, len(prediction))
                safe_cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
                top_total, top_query = torch.topk(
                    safe_cost, top_count, dim=0, largest=False, sorted=True)
                candidates = {
                    "query_indices": top_query.transpose(0, 1).detach(),
                    "total": top_total.transpose(0, 1).detach(),
                }
                for name in self.weights:
                    component = components.get(name, cost.new_zeros(cost.shape))
                    candidates[name] = component.gather(0, top_query).transpose(0, 1).detach()
                candidate_costs.append(candidates)
        response = {"indices": result}
        if return_costs:
            response["matched_costs"] = matched_costs
            response["candidate_costs"] = candidate_costs
            response["weights"] = dict(self.weights)
            response["chamfer_distance"] = self.chamfer_distance
            response["chamfer_source_alignment"] = (
                "O2-DFINE paper Eq. 10"
                if self.chamfer_distance == "paper_squared"
                else "released O2-RTDETR source"
            )
            response["kld"] = {
                "sqrt": self.kld_sqrt,
                "fun": self.kld_fun,
                "tau": self.kld_tau,
                "source_alignment": "released O2-RTDETR configuration"
                    if (not self.kld_sqrt and self.kld_fun == "log1p"
                        and self.kld_tau == 1.0)
                    else "explicit local configuration",
            }
        return response
