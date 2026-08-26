"""Losses for the D-FINE oriented-box baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou
from .dfine_utils import bbox2distance
from .obb.methods.o2.adr import (
    adr_orthogonality_error,
    adr_target_residual,
    apply_adr_residuals,
    distribution_integral,
    rbox_to_adr,
    translate_with_project,
)
from .rotated_box_ops import aligned_kld_loss, angle_distance, rotated_iou


@register()
class RotatedRTv4Criterion(nn.Module):
    """Match and supervise class, FDR geometry, angle, and OBB geometry."""

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses=("focal", "boxes", "local"),
        alpha=0.25,
        gamma=2.0,
        num_classes=12,
        reg_max=32,
        use_uni_set=True,
        kld_sqrt=False,
        kld_fun="log1p",
        kld_tau=1.0,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = dict(weight_dict)
        self.losses = tuple(losses)
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_uni_set = bool(use_uni_set)
        self.kld_sqrt = bool(kld_sqrt)
        self.kld_fun = str(kld_fun)
        self.kld_tau = float(kld_tau)
        if self.kld_fun not in {"log1p", "sqrt", "none"}:
            raise ValueError(f"Unsupported KLD post-processing function: {self.kld_fun!r}")
        self.last_diagnostics = None
        self.last_nonfinite_losses = None

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

    @staticmethod
    def _aligned_xywh_quality(pred_boxes, target_boxes):
        """Return the source-aligned D-FINE localization quality.

        D-FINE weights FGL with the aligned IoU obtained by interpreting the
        first four ``(cx, cy, w, h)`` parameters as horizontal boxes.  Every
        released O²-RTDETR recipe makes the same explicit ``hbox_iou`` choice.
        O²-DFINE does not publish decoder source, so inheriting this quality
        definition is the only choice supported independently by both parent
        implementations.  Rotation quality remains supervised by L1/KLD and
        is reported separately with rotated IoU in diagnostics/evaluation.
        """

        if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 5:
            raise ValueError(
                "Aligned OBB quality expects equal (..., 5) tensors, got "
                f"{tuple(pred_boxes.shape)} and {tuple(target_boxes.shape)}")
        if pred_boxes.numel() == 0:
            return pred_boxes.new_empty(pred_boxes.shape[:-1])
        overlaps, _ = box_iou(
            box_cxcywh_to_xyxy(pred_boxes.reshape(-1, 5)[:, :4]),
            box_cxcywh_to_xyxy(target_boxes.reshape(-1, 5)[:, :4]),
        )
        return overlaps.diagonal().reshape(pred_boxes.shape[:-1])

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
                matched_quality = self._aligned_xywh_quality(
                    pred_boxes, target_boxes).clamp(0, 1)
                quality[matched] = matched_quality.to(dtype=quality.dtype)
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
        # The released O² head applies ordinary L1 to all five normalized box
        # parameters.  Keep that exact source semantics here; its disagreement
        # with periodic OBB geometry is exposed by diagnostics, not hidden by a
        # local loss variant inside the reproduction baseline.
        angle_loss = (pred_boxes[:, 4] - target_boxes[:, 4]).abs()
        return {
            "loss_bbox": F.l1_loss(pred_boxes[:, :4], target_boxes[:, :4], reduction="sum") / normalizer,
            "loss_angle": angle_loss.sum() / normalizer,
            "loss_kld": aligned_kld_loss(
                pred_boxes,
                target_boxes,
                sqrt=self.kld_sqrt,
                fun=self.kld_fun,
                tau=self.kld_tau,
            ).sum() / normalizer,
        }

    def _local_loss(self, outputs, targets, indices, normalizer, suffix=""):
        if "pred_corners" not in outputs or "ref_points" not in outputs:
            return {}
        matched, target_boxes, _ = self._matched_tensors(outputs, targets, indices)
        if not len(target_boxes):
            return {"loss_fgl": outputs["pred_corners"].sum() * 0}
        matched_distributions = outputs["pred_corners"][matched]
        references = outputs["ref_points"][matched].detach()
        bins = self.reg_max + 1
        if matched_distributions.shape[-1] % bins:
            raise ValueError("Fine-grained logits are not divisible by reg_max + 1")
        components = matched_distributions.shape[-1] // bins
        if components == 6:
            if "adr_project" not in outputs:
                raise KeyError("Six-component refinement requires adr_project")
            targets_residual = adr_target_residual(references, target_boxes)
            left, weight_right, weight_left = translate_with_project(
                targets_residual, outputs["adr_project"])
            right = left + 1
        elif components == 4:
            labels, weight_right, weight_left = bbox2distance(
                references[..., :4], box_cxcywh_to_xyxy(target_boxes[:, :4]), self.reg_max,
                outputs["reg_scale"], outputs["up"])
            left = labels.long()
            right = left + 1
            weight_left, weight_right = weight_left.reshape(-1), weight_right.reshape(-1)
        else:
            raise ValueError(f"Expected four FDR or six ADR components, got {components}")
        pred_distribution = matched_distributions.reshape(-1, bins)
        loss = F.cross_entropy(pred_distribution, left, reduction="none") * weight_left.reshape(-1)
        loss += F.cross_entropy(pred_distribution, right, reduction="none") * weight_right.reshape(-1)
        with torch.no_grad():
            quality = self._aligned_xywh_quality(
                outputs["pred_boxes"][matched].detach(), target_boxes,
            ).clamp(0, 1)
            quality = quality[:, None].expand(-1, components).reshape(-1)
        return {"loss_fgl": (loss * quality).sum() / normalizer}

    def _compute(
        self,
        outputs,
        targets,
        indices,
        suffix="",
        loss_names=None,
        geometry_indices=None,
    ):
        values = {}
        for loss_name in self.losses if loss_names is None else loss_names:
            selected_indices = geometry_indices \
                if geometry_indices is not None and loss_name in {"boxes", "local"} \
                else indices
            normalizer = self._normalizer(
                selected_indices, outputs["pred_logits"].device)
            if loss_name in {"focal", "vfl"}:
                values.update(self._classification_loss(
                    outputs, targets, selected_indices, normalizer, loss_name))
            elif loss_name == "boxes":
                values.update(self._box_losses(
                    outputs, targets, selected_indices, normalizer))
            elif loss_name == "local":
                values.update(self._local_loss(
                    outputs, targets, selected_indices, normalizer, suffix=suffix))
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

    @staticmethod
    def _get_union_indices(indices, additional_indices):
        """Build D-FINE's decoder-wide geometric matching union.

        Classification retains each layer's one-to-one Hungarian assignment;
        box and fine-grained losses use every distinct query/GT pair observed
        across decoder, pre-decoder, and encoder assignments.  If a query was
        assigned to several GTs, the most frequent pair is retained, matching
        the original D-FINE criterion semantics.
        """

        results = []
        all_layers = [indices, *additional_indices]
        for batch_index in range(len(indices)):
            pairs = [
                torch.stack((layer[batch_index][0], layer[batch_index][1]), dim=1)
                for layer in all_layers if len(layer[batch_index][0])
            ]
            if not pairs:
                empty = indices[batch_index][0].new_empty(0, dtype=torch.long)
                results.append((empty, empty.clone()))
                continue
            unique, counts = torch.unique(
                torch.cat(pairs, dim=0), return_counts=True, dim=0)
            order = torch.argsort(counts, descending=True, stable=True)
            chosen = {}
            for pair in unique[order]:
                query, target = int(pair[0]), int(pair[1])
                chosen.setdefault(query, target)
            device = unique.device
            results.append((
                torch.tensor(list(chosen), dtype=torch.long, device=device),
                torch.tensor(list(chosen.values()), dtype=torch.long, device=device),
            ))
        return results

    def forward(self, outputs, targets, **kwargs):
        self.last_nonfinite_losses = None
        collect_diagnostics = bool(kwargs.get("collect_diagnostics", False))
        main_indices = self.matcher(outputs, targets)["indices"]
        auxiliaries = outputs.get("aux_outputs", [])
        auxiliary_indices = [
            self.matcher(auxiliary, targets)["indices"]
            for auxiliary in auxiliaries
        ]
        pre_indices = self.matcher(outputs["pre_outputs"], targets)["indices"] \
            if "pre_outputs" in outputs else None
        encoder_indices = [
            self.matcher(encoder_output, targets)["indices"]
            for encoder_output in outputs.get("enc_aux_outputs", [])
        ]
        union_sources = [*auxiliary_indices, *encoder_indices]
        if pre_indices is not None:
            union_sources.append(pre_indices)
        geometry_indices = self._get_union_indices(main_indices, union_sources) \
            if self.use_uni_set else None

        losses = {}
        losses.update(self._compute(
            outputs, targets, main_indices, geometry_indices=geometry_indices))
        for index, (auxiliary, aux_indices) in enumerate(
                zip(auxiliaries, auxiliary_indices)):
            auxiliary["up"], auxiliary["reg_scale"] = outputs["up"], outputs["reg_scale"]
            for key in ("adr_project", "refinement_kind"):
                if key in outputs:
                    auxiliary[key] = outputs[key]
            losses.update(self._compute(
                auxiliary,
                targets,
                aux_indices,
                f"_aux_{index}",
                geometry_indices=geometry_indices,
            ))
        decoder_layer_indices = [*auxiliary_indices, main_indices]
        if pre_indices is not None:
            pre = outputs["pre_outputs"]
            pre_losses = tuple(loss for loss in self.losses if loss != "local")
            losses.update(self._compute(
                pre,
                targets,
                pre_indices,
                "_pre",
                pre_losses,
                geometry_indices=geometry_indices,
            ))
        for index, (encoder_output, enc_indices) in enumerate(
                zip(outputs.get("enc_aux_outputs", []), encoder_indices)):
            encoder_losses = tuple(loss for loss in self.losses if loss != "local")
            losses.update(self._compute(
                encoder_output,
                targets,
                enc_indices,
                f"_enc_{index}",
                encoder_losses,
                geometry_indices=geometry_indices,
            ))
        if "dn_outputs" in outputs:
            dn_indices = self._dn_indices(outputs["dn_meta"], targets)
            for index, dn_output in enumerate(outputs["dn_outputs"]):
                dn_output["up"], dn_output["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for key in ("adr_project", "refinement_kind"):
                    if key in outputs:
                        dn_output[key] = outputs[key]
                losses.update(self._compute(dn_output, targets, dn_indices, f"_dn_{index}"))
            if "dn_pre_outputs" in outputs:
                dn_pre_losses = tuple(loss for loss in self.losses if loss != "local")
                losses.update(self._compute(
                    outputs["dn_pre_outputs"], targets, dn_indices, "_dn_pre", dn_pre_losses))
        if collect_diagnostics:
            self.last_diagnostics = self._main_match_diagnostics(
                outputs, targets, main_indices)
            self.last_diagnostics["assignment_instability"] = self._assignment_instability(
                decoder_layer_indices, targets)
            self.last_diagnostics["training_semantics"] = {
                "dfine_union_matching": self.use_uni_set,
                "union_match_count": sum(len(source) for source, _ in geometry_indices)
                    if geometry_indices is not None else None,
                "kld": {
                    "sqrt": self.kld_sqrt,
                    "fun": self.kld_fun,
                    "tau": self.kld_tau,
                },
                "angle_loss": {
                    "mode": "raw_normalized_l1",
                    "source_alignment": "shared five-parameter OBB L1",
                },
                "localization_quality": {
                    "mode": "aligned_xywh_iou",
                    "source_alignment": "D-FINE FGL and released O2 hbox_iou",
                },
                "refinement_kind": outputs.get("refinement_kind"),
                "adr_geometry_contract": outputs.get("adr_geometry_contract"),
            }
        else:
            self.last_diagnostics = None
        nonfinite = {}
        for name, value in losses.items():
            invalid = ~torch.isfinite(value.detach())
            if invalid.any():
                nonfinite[name] = {
                    "element_count": int(value.numel()),
                    "nonfinite_count": int(invalid.sum()),
                    "has_nan": bool(torch.isnan(value.detach()).any()),
                    "has_posinf": bool(torch.isposinf(value.detach()).any()),
                    "has_neginf": bool(torch.isneginf(value.detach()).any()),
                }
        if nonfinite:
            self.last_nonfinite_losses = nonfinite
            details = ", ".join(
                f"{name}={record}" for name, record in nonfinite.items())
            raise FloatingPointError(
                "Rotated criterion produced non-finite raw losses; refusing "
                f"to sanitize them because that preserves invalid gradients: {details}"
            )
        return losses

    @staticmethod
    @torch.no_grad()
    def _assignment_instability(layer_indices, targets):
        """Compute the O^2 query-index instability over decoder layers.

        For every ground truth, this records whether its matched query index
        changes anywhere in the layer sequence.  Adjacent-layer rates are
        retained as well, which makes the aggregate value auditable.
        """

        layer_count = len(layer_indices)
        transition_changes = [0 for _ in range(max(layer_count - 1, 0))]
        transition_totals = [0 for _ in transition_changes]
        total_ground_truth = 0
        total_changed = 0
        per_image = []
        for batch_index, target in enumerate(targets):
            count = len(target["labels"])
            assignments = torch.full((layer_count, count), -1, dtype=torch.long)
            for layer_index, indices in enumerate(layer_indices):
                source, target_index = indices[batch_index]
                if len(target_index):
                    assignments[layer_index, target_index.cpu()] = source.cpu()
            if layer_count > 1 and count:
                adjacent = assignments[1:] != assignments[:-1]
                changed = adjacent.any(dim=0)
                for transition_index in range(layer_count - 1):
                    transition_changes[transition_index] += int(adjacent[transition_index].sum())
                    transition_totals[transition_index] += count
            else:
                changed = torch.zeros(count, dtype=torch.bool)
            changed_count = int(changed.sum())
            total_ground_truth += count
            total_changed += changed_count
            per_image.append({
                "ground_truth_count": count,
                "changed_count": changed_count,
                "instability": changed_count / max(count, 1),
            })
        transitions = [
            {
                "from_layer": index,
                "to_layer": index + 1,
                "changed_count": changed,
                "ground_truth_count": total,
                "instability": changed / max(total, 1),
            }
            for index, (changed, total) in enumerate(
                zip(transition_changes, transition_totals))
        ]
        return {
            "definition": "fraction of GT whose matched query index changes across decoder layers",
            "decoder_layer_count": layer_count,
            "ground_truth_count": total_ground_truth,
            "changed_count": total_changed,
            "instability": total_changed / max(total_ground_truth, 1),
            "adjacent_transitions": transitions,
            "per_image": per_image,
        }

    @torch.no_grad()
    def _main_match_diagnostics(self, outputs, targets, indices):
        matched, target_boxes, target_labels = self._matched_tensors(outputs, targets, indices)
        pred_boxes = outputs["pred_boxes"][matched]
        if not len(target_boxes):
            return {"matched_count": 0}
        probabilities = outputs["pred_logits"][matched].sigmoid()
        target_scores = probabilities.gather(1, target_labels[:, None]).squeeze(1)
        top_scores, top_labels = probabilities.max(dim=1)
        center_error = torch.linalg.vector_norm(pred_boxes[:, :2] - target_boxes[:, :2], dim=1)
        scale = torch.linalg.vector_norm(target_boxes[:, 2:4], dim=1).clamp_min(1e-7)
        geometry_angle = angle_distance(pred_boxes[:, 4], target_boxes[:, 4])
        loss_angle = (pred_boxes[:, 4] - target_boxes[:, 4]).abs()
        target_anisotropy = (
            (target_boxes[:, 2] - target_boxes[:, 3]).abs() /
            (target_boxes[:, 2] + target_boxes[:, 3]).clamp_min(1e-7)
        )
        values = {
            "matched_count": len(target_boxes),
            "center_error_normalized": center_error / scale,
            "angle_error_deg": geometry_angle * 180.0,
            "angle_loss_error_deg": loss_angle * 180.0,
            "target_anisotropy": target_anisotropy,
            "rotated_iou": rotated_iou(pred_boxes, target_boxes, aligned=True).clamp(0, 1),
            "target_class_score": target_scores,
            "top_score": top_scores,
            "class_correct": (top_labels == target_labels).float(),
        }

        # ADR removes the scalar-angle seam from the final OBB, but its
        # ``top``/``right`` gliding-vertex chart still has identified
        # endpoints.  Around an image-axis crossing, epsilon/Wr and eta/Hr
        # can jump between values close to zero and one while the rectangle
        # itself changes continuously.  Record that exposure explicitly so a
        # run can be analysed without regenerating training samples later.
        adr_chart = None
        if "adr_project" in outputs and "ref_points" in outputs:
            target_adr, target_external_scale = rbox_to_adr(target_boxes)
            offset_fraction = (
                target_adr[:, 4:] / target_external_scale.clamp_min(1e-7)
            ).clamp(0, 1)
            offset_endpoint_distance = torch.minimum(
                offset_fraction, 1 - offset_fraction)
            seam_distance = offset_endpoint_distance.amin(dim=-1)
            reference_boxes = outputs["ref_points"][matched]
            target_residual = adr_target_residual(reference_boxes, target_boxes)
            project = outputs["adr_project"].detach()
            outside_codebook = (
                (target_residual < project.min()) |
                (target_residual > project.max())
            )
            values.update({
                "adr_epsilon_fraction": offset_fraction[:, 0],
                "adr_eta_fraction": offset_fraction[:, 1],
                "adr_chart_seam_distance": seam_distance,
                "adr_target_residual_abs": target_residual.abs().mean(dim=-1),
                "adr_target_offset_residual_abs": target_residual[:, 4:].abs().mean(dim=-1),
            })
            if "pred_corners" in outputs:
                predicted_distribution = outputs["pred_corners"][matched]
                predicted_residual = distribution_integral(
                    predicted_distribution, project, components=6)
                predicted_values = apply_adr_residuals(
                    reference_boxes, predicted_residual)
                raw_orthogonality = adr_orthogonality_error(predicted_values)
                values["adr_raw_orthogonality_error"] = raw_orthogonality
            else:
                raw_orthogonality = None
            adr_chart = {
                "definition": (
                    "min over epsilon/Wr, eta/Hr of distance to the identified "
                    "linear-chart endpoints {0,1}"
                ),
                "known_limitation": (
                    "top/right vertex identity can switch at image-axis crossings; "
                    "OBB geometry stays continuous but ADR offset targets need not"
                ),
                "near_endpoint_count_0p001": int((seam_distance <= 0.001).sum()),
                "near_endpoint_count_0p01": int((seam_distance <= 0.01).sum()),
                "near_endpoint_count_0p05": int((seam_distance <= 0.05).sum()),
                "outside_codebook_component_count": int(outside_codebook.sum()),
                "outside_codebook_box_count": int(outside_codebook.any(dim=-1).sum()),
                "raw_orthogonality_error_definition": (
                    "absolute cosine between consecutive raw gliding-vertex "
                    "edges before equal-diagonal rectangle completion"
                ),
                "raw_orthogonality_error_mean": (
                    float(raw_orthogonality.float().mean())
                    if raw_orthogonality is not None and len(raw_orthogonality)
                    else None
                ),
                "matched_count": int(len(target_boxes)),
            }
        summary = {"matched_count": values.pop("matched_count")}
        for name, tensor in values.items():
            finite = tensor[torch.isfinite(tensor)].float()
            if not len(finite):
                continue
            summary[name] = {
                "mean": float(finite.mean()),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }
        summary["square_symmetry"] = {
            "loss_mode": "raw_normalized_l1",
            "anisotropy_definition": "abs(width-height)/(width+height)",
            "exact_square_count": int((target_anisotropy <= 1e-7).sum()),
            "angle_seam_disagreement_count": int(
                (loss_angle - geometry_angle > 1e-6).sum()),
            "known_limitation": (
                "five-parameter L1 is not invariant to the angle seam or the "
                "quarter-turn equivalence of exact squares"
            ),
        }
        if adr_chart is not None:
            summary["adr_chart_seam"] = adr_chart
        return summary
