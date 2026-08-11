"""Losses for the D-FINE oriented-box baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy
from .dfine_utils import bbox2distance
from .rotated_box_ops import aligned_kld_loss, angle_distance, rotated_iou


@register()
class RotatedRTv4Criterion(nn.Module):
    """Match and supervise class, FDR geometry, angle, and OBB geometry."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(self, matcher, weight_dict, losses=("focal", "boxes", "local"),
                 alpha=0.25, gamma=2.0, num_classes=12, reg_max=32, **kwargs):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.losses = tuple(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reg_max = reg_max

    @staticmethod
    def _matched_tensors(outputs, targets, indices):
        device = outputs["pred_boxes"].device
        batch_indices, source_indices, target_boxes, target_labels = [], [], [], []
        for batch_index, ((source, target_index), target) in enumerate(zip(indices, targets)):
            source = source.to(device)
            target_index = target_index.to(device)
            if len(source):
                batch_indices.append(torch.full_like(source, batch_index))
                source_indices.append(source)
                target_boxes.append(target["boxes"][target_index])
                target_labels.append(target["labels"][target_index])
        if not source_indices:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return (empty, empty), outputs["pred_boxes"].new_empty((0, 5)), empty
        return ((torch.cat(batch_indices), torch.cat(source_indices)),
                torch.cat(target_boxes), torch.cat(target_labels))

    def _normalizer(self, indices, device):
        count = sum(len(source) for source, _ in indices)
        value = torch.tensor([count], dtype=torch.float32, device=device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(value)
        return torch.clamp(value / get_world_size(), min=1).item()

    def _classification_loss(self, outputs, targets, indices, normalizer, kind):
        logits = outputs["pred_logits"]
        matched, target_boxes, target_labels = self._matched_tensors(outputs, targets, indices)
        target_classes = torch.full(logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=logits.device)
        target_classes[matched] = target_labels
        one_hot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1].to(logits.dtype)
        if kind == "vfl":
            quality = torch.zeros(logits.shape[:2], device=logits.device, dtype=logits.dtype)
            if len(target_boxes):
                pred_boxes = outputs["pred_boxes"][matched].detach()
                quality[matched] = rotated_iou(pred_boxes, target_boxes, aligned=True).clamp(0, 1)
            target_score = quality.unsqueeze(-1) * one_hot
            prediction_score = logits.sigmoid().detach()
            weight = self.alpha * prediction_score.pow(self.gamma) * (1 - one_hot) + target_score
            loss = F.binary_cross_entropy_with_logits(logits, target_score, weight=weight, reduction="none")
            loss = loss.mean(1).sum() * logits.shape[1] / normalizer
            return {"loss_vfl": loss}
        loss = torchvision.ops.sigmoid_focal_loss(
            logits, one_hot, self.alpha, self.gamma, reduction="none")
        loss = loss.mean(1).sum() * logits.shape[1] / normalizer
        return {"loss_focal": loss}

    def _box_losses(self, outputs, targets, indices, normalizer):
        matched, target_boxes, _ = self._matched_tensors(outputs, targets, indices)
        pred_boxes = outputs["pred_boxes"][matched]
        if not len(target_boxes):
            zero = outputs["pred_boxes"].sum() * 0
            return {"loss_bbox": zero, "loss_angle": zero, "loss_kld": zero}
        return {
            "loss_bbox": F.l1_loss(pred_boxes[:, :4], target_boxes[:, :4], reduction="sum") / normalizer,
            "loss_angle": angle_distance(pred_boxes[:, 4], target_boxes[:, 4]).sum() / normalizer,
            "loss_kld": aligned_kld_loss(pred_boxes, target_boxes).sum() / normalizer,
        }

    def _local_loss(self, outputs, targets, indices, normalizer):
        if "pred_corners" not in outputs or "ref_points" not in outputs:
            return {}
        matched, target_boxes, _ = self._matched_tensors(outputs, targets, indices)
        if not len(target_boxes):
            return {"loss_fgl": outputs["pred_corners"].sum() * 0}
        pred_distribution = outputs["pred_corners"][matched].reshape(-1, self.reg_max + 1)
        references = outputs["ref_points"][matched][..., :4].detach()
        labels, weight_right, weight_left = bbox2distance(
            references, box_cxcywh_to_xyxy(target_boxes[:, :4]), self.reg_max,
            outputs["reg_scale"], outputs["up"])
        left = labels.long()
        right = left + 1
        loss = F.cross_entropy(pred_distribution, left, reduction="none") * weight_left.reshape(-1)
        loss += F.cross_entropy(pred_distribution, right, reduction="none") * weight_right.reshape(-1)
        with torch.no_grad():
            quality = rotated_iou(outputs["pred_boxes"][matched].detach(), target_boxes,
                                  aligned=True).clamp(0, 1)
            quality = quality[:, None].expand(-1, 4).reshape(-1)
        return {"loss_fgl": (loss * quality).sum() / normalizer}

    def _compute(self, outputs, targets, indices, suffix="", loss_names=None):
        normalizer = self._normalizer(indices, outputs["pred_logits"].device)
        values = {}
        for loss_name in self.losses if loss_names is None else loss_names:
            if loss_name in {"focal", "vfl"}:
                values.update(self._classification_loss(outputs, targets, indices, normalizer, loss_name))
            elif loss_name == "boxes":
                values.update(self._box_losses(outputs, targets, indices, normalizer))
            elif loss_name == "local":
                values.update(self._local_loss(outputs, targets, indices, normalizer))
            else:
                raise ValueError(f"Unsupported rotated loss: {loss_name}")
        return {f"{name}{suffix}": value * self.weight_dict.get(name, 1.0)
                for name, value in values.items()}

    @staticmethod
    def _dn_indices(meta, targets):
        device = targets[0]["labels"].device
        indices = []
        target_indices = meta.get("dn_target_idx")
        for batch_index, positive in enumerate(meta["dn_positive_idx"]):
            if target_indices is None:
                gt = torch.arange(len(targets[batch_index]["labels"]), device=device)
                gt = gt.tile(meta["dn_num_group"])
            else:
                gt = target_indices[batch_index].to(device)
            indices.append((positive.to(device), gt))
        return indices

    def forward(self, outputs, targets, **kwargs):
        losses = {}
        main_indices = self.matcher(outputs, targets)["indices"]
        losses.update(self._compute(outputs, targets, main_indices))
        for index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            auxiliary["up"], auxiliary["reg_scale"] = outputs["up"], outputs["reg_scale"]
            aux_indices = self.matcher(auxiliary, targets)["indices"]
            losses.update(self._compute(auxiliary, targets, aux_indices, f"_aux_{index}"))
        if "pre_outputs" in outputs:
            pre = outputs["pre_outputs"]
            pre_indices = self.matcher(pre, targets)["indices"]
            pre_losses = tuple(loss for loss in self.losses if loss != "local")
            losses.update(self._compute(pre, targets, pre_indices, "_pre", pre_losses))
        for index, encoder_output in enumerate(outputs.get("enc_aux_outputs", [])):
            encoder_indices = self.matcher(encoder_output, targets)["indices"]
            encoder_losses = tuple(loss for loss in self.losses if loss != "local")
            losses.update(self._compute(
                encoder_output, targets, encoder_indices, f"_enc_{index}", encoder_losses))
        if "dn_outputs" in outputs:
            dn_indices = self._dn_indices(outputs["dn_meta"], targets)
            for index, dn_output in enumerate(outputs["dn_outputs"]):
                dn_output["up"], dn_output["reg_scale"] = outputs["up"], outputs["reg_scale"]
                losses.update(self._compute(dn_output, targets, dn_indices, f"_dn_{index}"))
            if "dn_pre_outputs" in outputs:
                dn_pre_losses = tuple(loss for loss in self.losses if loss != "local")
                losses.update(self._compute(
                    outputs["dn_pre_outputs"], targets, dn_indices, "_dn_pre", dn_pre_losses))
        return {name: torch.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4)
                for name, value in losses.items()}
