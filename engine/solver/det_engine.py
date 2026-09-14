"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import sys
import math
import time
from collections import OrderedDict
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..rtv4.rotated_box_ops import angle_distance, rotated_iou


def _refinement_stage_outputs(outputs):
    """Return the common pre-box -> decoder-stage prediction contract."""

    required = (
        "diagnostic_pre_logits", "diagnostic_pre_boxes",
        "diagnostic_layer_logits", "diagnostic_layer_boxes",
    )
    if any(name not in outputs for name in required):
        return OrderedDict()
    layer_logits = outputs["diagnostic_layer_logits"]
    layer_boxes = outputs["diagnostic_layer_boxes"]
    if layer_logits.shape[:3] != layer_boxes.shape[:3]:
        raise RuntimeError(
            "Diagnostic layer logits and boxes have inconsistent shapes: "
            f"{tuple(layer_logits.shape)} versus {tuple(layer_boxes.shape)}")
    stages = OrderedDict([
        ("pre_box", {
            "pred_logits": outputs["diagnostic_pre_logits"],
            "pred_boxes": outputs["diagnostic_pre_boxes"],
        })
    ])
    for layer_index in range(len(layer_boxes)):
        stages[f"decoder_{layer_index}"] = {
            "pred_logits": layer_logits[layer_index],
            "pred_boxes": layer_boxes[layer_index],
        }
    return stages


def _evaluator_metrics(evaluator):
    return {
        "metrics": getattr(evaluator, "metrics", {}),
        "per_class": getattr(evaluator, "per_class", {}),
        "per_class_metrics": getattr(evaluator, "per_class_metrics", {}),
        "stats": getattr(evaluator, "stats", []),
        "prediction_count": int(sum(
            len(prediction["scores"])
            for prediction in getattr(evaluator, "predictions", {}).values()
        )),
    }


def _synchronize_for_measurement(device, enabled):
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _distribution(tensor):
    tensor = tensor.detach().float().reshape(-1)
    tensor = tensor[torch.isfinite(tensor)]
    if not len(tensor):
        return {"count": 0}
    quantiles = torch.quantile(tensor, tensor.new_tensor([0.0, 0.25, 0.5, 0.75, 0.95, 1.0]))
    return {
        "count": len(tensor), "mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False)),
        "min": float(quantiles[0]), "p25": float(quantiles[1]),
        "median": float(quantiles[2]), "p75": float(quantiles[3]),
        "p95": float(quantiles[4]), "max": float(quantiles[5]),
    }


def _fine_grained_distribution_observations(outputs):
    logits = outputs.get("pred_corners")
    project = outputs.get("distribution_project")
    if logits is None or project is None:
        return None
    project = project.detach().float().reshape(-1)
    if not len(project) or logits.shape[-1] % len(project):
        return {"error": "distribution logits/codebook shape mismatch"}
    components = logits.shape[-1] // len(project)
    names = list(outputs.get("distribution_names", ()))
    if len(names) != components:
        names = [f"component_{index}" for index in range(components)]
    values = logits.detach().float().reshape(*logits.shape[:-1], components, len(project))
    probabilities = values.softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    peak_probability, peak_bin = probabilities.max(dim=-1)
    expectation = (probabilities * project).sum(dim=-1)
    variance = (probabilities * (project - expectation[..., None]).square()).sum(dim=-1)
    return {
        "refinement_kind": outputs.get("refinement_kind"),
        "refinement_mode": outputs.get("refinement_mode"),
        "bin_values": project,
        "components": {
            name: {
                "logit": _distribution(values[..., index, :]),
                "entropy": _distribution(entropy[..., index]),
                "peak_probability": _distribution(peak_probability[..., index]),
                "peak_bin_histogram": torch.bincount(
                    peak_bin[..., index].reshape(-1), minlength=len(project)),
                "expected_residual": _distribution(expectation[..., index]),
                "variance": _distribution(variance[..., index]),
            }
            for index, name in enumerate(names)
        },
    }


def _box_geometry_observations(boxes):
    boxes = boxes.detach()
    finite_box = torch.isfinite(boxes).all(dim=-1)
    finite = boxes[finite_box].float()
    result = {
        "box_count": int(finite_box.numel()),
        "nonfinite_box_count": int((~finite_box).sum()),
    }
    if not len(finite):
        return result
    width_height = finite[..., 2:4]
    minor_side = width_height.amin(dim=-1)
    major_side = width_height.amax(dim=-1)
    angle = torch.remainder(finite[..., 4], 1.0)
    result.update({
        "center_x": _distribution(finite[..., 0]),
        "center_y": _distribution(finite[..., 1]),
        "width": _distribution(finite[..., 2]),
        "height": _distribution(finite[..., 3]),
        "minor_side": _distribution(minor_side),
        "area_normalized": _distribution(width_height.prod(dim=-1)),
        "aspect_ratio": _distribution(major_side / minor_side.clamp_min(1e-12)),
        "anisotropy": _distribution(
            (finite[..., 2] - finite[..., 3]).abs() /
            width_height.sum(dim=-1).clamp_min(1e-12)),
        "angle_normalized": _distribution(angle),
        "angle_seam_distance": _distribution(torch.minimum(angle, 1.0 - angle)),
    })
    return result


@torch.no_grad()
def _denoising_recovery_observations(outputs, targets):
    """Measure what every decoder stage recovers from its DN references."""

    meta = outputs.get("dn_meta")
    stages = []
    if "dn_pre_outputs" in outputs:
        stages.append(("pre_box", outputs["dn_pre_outputs"]))
    stages.extend(
        (f"decoder_{index}", stage)
        for index, stage in enumerate(outputs.get("dn_outputs", [])))
    if not meta or not stages:
        return None
    groups = int(meta["dn_num_group"])
    total = int(meta["dn_num_split"][0])
    if groups <= 0 or total % (2 * groups):
        return {"error": "invalid denoising group layout"}
    max_gt = total // (2 * groups)
    target_indices = meta.get("dn_target_idx")
    records = []
    for stage_name, stage in stages:
        positive_boxes, target_boxes = [], []
        positive_target_scores, negative_target_scores = [], []
        positive_max_scores, negative_max_scores = [], []
        for batch_index, positive in enumerate(meta["dn_positive_idx"]):
            positive = positive.to(stage["pred_boxes"].device)
            if not len(positive):
                continue
            if target_indices is None:
                target_index = torch.arange(
                    len(targets[batch_index]["boxes"]), device=positive.device
                ).tile(groups)
            else:
                target_index = target_indices[batch_index].to(positive.device)
            negative = positive + max_gt
            labels = targets[batch_index]["labels"][target_index]
            logits = stage["pred_logits"][batch_index]
            probabilities = logits.sigmoid()
            positive_target_scores.append(probabilities[positive, labels])
            negative_target_scores.append(probabilities[negative, labels])
            positive_max_scores.append(probabilities[positive].amax(dim=-1))
            negative_max_scores.append(probabilities[negative].amax(dim=-1))
            positive_boxes.append(stage["pred_boxes"][batch_index, positive])
            target_boxes.append(targets[batch_index]["boxes"][target_index])
        if not positive_boxes:
            continue
        predicted = torch.cat(positive_boxes)
        target = torch.cat(target_boxes)
        target_diagonal = torch.linalg.vector_norm(
            target[:, 2:4], dim=-1).clamp_min(1e-12)
        records.append({
            "stage": stage_name,
            "matched_positive_count": len(predicted),
            "positive_rotated_iou": _distribution(rotated_iou(
                predicted, target, aligned=True, model_space=True)),
            "positive_center_error_target_diagonal": _distribution(
                torch.linalg.vector_norm(
                    predicted[:, :2] - target[:, :2], dim=-1
                ) / target_diagonal),
            "positive_log_size_error": _distribution((
                predicted[:, 2:4].clamp_min(1e-12).log()
                - target[:, 2:4].clamp_min(1e-12).log()
            ).abs().mean(dim=-1)),
            "positive_angle_error_deg": _distribution(
                angle_distance(predicted[:, 4], target[:, 4]) * 180.0),
            "positive_target_class_score": _distribution(
                torch.cat(positive_target_scores)),
            "negative_target_class_score": _distribution(
                torch.cat(negative_target_scores)),
            "positive_max_class_score": _distribution(
                torch.cat(positive_max_scores)),
            "negative_max_class_score": _distribution(
                torch.cat(negative_max_scores)),
        })
    return records


@torch.no_grad()
def _gradient_statistics(model):
    broad_accumulators = {}
    component_accumulators = {}

    component_prefixes = (
        ("encoder_box_head", "decoder.enc_bbox_head."),
        ("pre_box_head", "decoder.pre_bbox_head."),
        ("box_refinement_heads", "decoder.dec_bbox_head."),
        ("angle_refinement_heads", "decoder.dec_angle_head."),
        ("classification_heads", "decoder.dec_score_head."),
        ("location_quality_heads", "decoder.decoder.lqe_layers."),
    )
    owner = getattr(model, "module", model)
    method_component_markers = getattr(
        owner, "diagnostic_gradient_groups", {})
    if not isinstance(method_component_markers, dict) or not all(
            isinstance(name, str) and isinstance(marker, str) and marker
            for name, marker in method_component_markers.items()):
        raise TypeError(
            "diagnostic_gradient_groups must map names to non-empty parameter "
            "name markers")
    reserved_components = {name for name, _ in component_prefixes}
    if reserved_components.intersection(method_component_markers):
        raise ValueError(
            "diagnostic_gradient_groups cannot replace stable component names")

    def accumulate(accumulators, group, grad):
        entry = accumulators.setdefault(group, {
            "square_sum": grad.new_zeros(()), "max_abs": grad.new_zeros(()),
            "element_count": 0, "nonfinite_count": 0, "parameter_tensors": 0,
        })
        entry["square_sum"] += grad.square().sum()
        entry["max_abs"] = torch.maximum(entry["max_abs"], grad.abs().max())
        entry["element_count"] += grad.numel()
        entry["nonfinite_count"] += int((~torch.isfinite(grad)).sum())
        entry["parameter_tensors"] += 1

    def finalize(accumulators):
        return {
            group: {
                "l2_norm": float(entry["square_sum"].sqrt()),
                "max_abs": float(entry["max_abs"]),
                "element_count": entry["element_count"],
                "parameter_tensors": entry["parameter_tensors"],
                "nonfinite_count": entry["nonfinite_count"],
            }
            for group, entry in accumulators.items()
        }

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        clean_name = name.removeprefix("module.")
        group = next((candidate for candidate in ("backbone", "encoder", "decoder")
                      if clean_name.startswith(candidate + ".")), "other")
        grad = parameter.grad.detach().float()
        accumulate(broad_accumulators, group, grad)
        component = next((component for component, prefix in component_prefixes
                          if clean_name.startswith(prefix)), None)
        if component is not None:
            accumulate(component_accumulators, component, grad)
        for component, marker in method_component_markers.items():
            if marker in clean_name:
                accumulate(component_accumulators, component, grad)

    result = finalize(broad_accumulators)
    total_square = sum(
        (entry["square_sum"] for entry in broad_accumulators.values()),
        start=torch.zeros(()) if not broad_accumulators else
        next(iter(broad_accumulators.values()))["square_sum"].new_zeros(()),
    )
    result["total_l2_norm"] = float(total_square.sqrt())
    # These decoder sub-groups are the earliest warning that an OBB head is
    # disconnected even when the aggregate decoder gradient remains healthy.
    result["components"] = finalize(component_accumulators)
    return result


@torch.no_grad()
def _training_observations(samples, targets, outputs):
    probabilities = outputs["pred_logits"].sigmoid()
    top_scores, top_labels = probabilities.max(dim=-1)
    boxes = outputs["pred_boxes"]
    target_boxes = torch.cat([target["boxes"] for target in targets], dim=0) \
        if any(len(target["boxes"]) for target in targets) else boxes.new_empty((0, 5))
    target_labels = torch.cat([target["labels"] for target in targets], dim=0) \
        if any(len(target["labels"]) for target in targets) else top_labels.new_empty(0)
    class_histogram = torch.bincount(target_labels, minlength=probabilities.shape[-1])
    augmentation_keys = sorted({
        key for target in targets for key in target if key.startswith("aug_")
    })
    denoising_meta = outputs.get("dn_meta", {})
    decoder_outputs = [*outputs.get("aux_outputs", []), outputs]
    denoising_outputs = outputs.get("dn_outputs", [])
    target_width_height = target_boxes[:, 2:4]
    target_minor_side = target_width_height.amin(dim=-1) \
        if len(target_boxes) else target_boxes.new_empty(0)
    target_major_side = target_width_height.amax(dim=-1) \
        if len(target_boxes) else target_boxes.new_empty(0)
    target_angle = torch.remainder(target_boxes[:, 4], 1.0) \
        if len(target_boxes) else target_boxes.new_empty(0)
    query_width_height = boxes[..., 2:4]
    query_minor_side = query_width_height.amin(dim=-1)
    query_major_side = query_width_height.amax(dim=-1)
    query_angle = torch.remainder(boxes[..., 4], 1.0)
    return {
        "numerics": {
            "samples_dtype": str(samples.dtype),
            "pred_logits_dtype": str(outputs["pred_logits"].dtype),
            "pred_boxes_dtype": str(outputs["pred_boxes"].dtype),
            "pred_distributions_dtype": str(outputs["pred_corners"].dtype)
                if "pred_corners" in outputs else None,
            "reference_boxes_dtype": str(outputs["ref_points"].dtype)
                if "ref_points" in outputs else None,
            "target_boxes_dtypes": sorted({
                str(target["boxes"].dtype) for target in targets
                if "boxes" in target
            }),
        },
        "batch": {
            "image_shape": list(samples.shape),
            "gt_per_image": [len(target["boxes"]) for target in targets],
            "gt_class_histogram": class_histogram,
            "empty_image_count": sum(not len(target["boxes"]) for target in targets),
            "augmentation": [
                {key: target[key] for key in augmentation_keys if key in target}
                for target in targets
            ],
        },
        "targets": {
            "center_x": _distribution(target_boxes[:, 0] if len(target_boxes) else target_boxes),
            "center_y": _distribution(target_boxes[:, 1] if len(target_boxes) else target_boxes),
            "width": _distribution(target_boxes[:, 2] if len(target_boxes) else target_boxes),
            "height": _distribution(target_boxes[:, 3] if len(target_boxes) else target_boxes),
            "minor_side": _distribution(target_minor_side),
            "area_normalized": _distribution(
                target_width_height.prod(dim=-1)
                if len(target_boxes) else target_boxes.new_empty(0)),
            "aspect_ratio": _distribution(
                target_major_side / target_minor_side.clamp_min(1e-12)),
            "anisotropy": _distribution(
                (target_boxes[:, 2] - target_boxes[:, 3]).abs() /
                target_width_height.sum(dim=-1).clamp_min(1e-12)
                if len(target_boxes) else target_boxes.new_empty(0)),
            "angle_normalized": _distribution(target_angle),
            "angle_seam_distance": _distribution(
                torch.minimum(target_angle, 1.0 - target_angle)),
        },
        "queries": {
            "top_score": _distribution(top_scores),
            "top_class_histogram": torch.bincount(
                top_labels.reshape(-1), minlength=probabilities.shape[-1]),
            "center_x": _distribution(boxes[..., 0]),
            "center_y": _distribution(boxes[..., 1]),
            "width": _distribution(boxes[..., 2]),
            "height": _distribution(boxes[..., 3]),
            "minor_side": _distribution(query_minor_side),
            "area_normalized": _distribution(query_width_height.prod(dim=-1)),
            "aspect_ratio": _distribution(
                query_major_side / query_minor_side.clamp_min(1e-12)),
            "anisotropy": _distribution(
                (boxes[..., 2] - boxes[..., 3]).abs() /
                query_width_height.sum(dim=-1).clamp_min(1e-12)),
            "angle_normalized": _distribution(query_angle),
            "angle_seam_distance": _distribution(
                torch.minimum(query_angle, 1.0 - query_angle)),
        },
        "decoder_box_geometry": [
            {"layer": index, **_box_geometry_observations(layer["pred_boxes"])}
            for index, layer in enumerate(decoder_outputs)
        ],
        "denoising_box_geometry": [
            {"layer": index, **_box_geometry_observations(layer["pred_boxes"])}
            for index, layer in enumerate(denoising_outputs)
        ] if denoising_outputs else None,
        "denoising_pre_box_geometry": _box_geometry_observations(
            outputs["dn_pre_outputs"]["pred_boxes"]
        ) if "dn_pre_outputs" in outputs else None,
        "denoising_recovery": _denoising_recovery_observations(outputs, targets),
        "fine_grained_distributions": _fine_grained_distribution_observations(outputs),
        # Optional research modules expose already-aggregated, detached
        # observations through this generic extension point.  Stable models do
        # not emit the key, so their forward and log schemas are unchanged.
        "method_diagnostics": outputs.get("method_train_diagnostics"),
        "denoising": {
            "mode": denoising_meta.get("dn_noise_mode"),
            "lambdas": denoising_meta.get("dn_noise_lambdas"),
            "num_group": denoising_meta.get("dn_num_group"),
            "query_split": denoising_meta.get("dn_num_split"),
            "crowded_policy": denoising_meta.get("dn_crowded_policy"),
            "group_base_count": denoising_meta.get("dn_group_base_count"),
            "requested_query_budget": denoising_meta.get(
                "dn_requested_query_budget"),
            "actual_query_count": denoising_meta.get("dn_actual_query_count"),
            "budget_exceeded": denoising_meta.get("dn_budget_exceeded"),
            "original_gt_counts": denoising_meta.get("dn_original_gt_counts"),
            "selected_gt_counts": denoising_meta.get("dn_selected_gt_counts"),
            "dropped_gt_counts": denoising_meta.get("dn_dropped_gt_counts"),
            "selected_target_indices": denoising_meta.get(
                "dn_selected_target_idx"),
            "dropped_target_indices": denoising_meta.get(
                "dn_dropped_target_idx"),
        } if denoising_meta else None,
    }

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        # Support both DDP ('module.') and non-DDP naming
        if name.startswith('module.encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # Gradient Analysis
    encoder_grad_percentages = []
    cur_iters = epoch * len(data_loader)

    teacher_model = kwargs.get('teacher_model', None)
    diagnostics = kwargs.get('diagnostics', None)
    amp_scale_min = math.inf
    amp_skipped_steps = 0
    loader_wait_start = time.perf_counter()

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if lr_warmup_scheduler is not None and hasattr(lr_warmup_scheduler, "prepare_step"):
            lr_warmup_scheduler.prepare_step()
        yielded_at = time.perf_counter()
        data_loader_wait_ms = (yielded_at - loader_wait_start) * 1000.0
        global_step = epoch * len(data_loader) + i
        diagnostic_step = diagnostics is not None and diagnostics.enabled and (
            diagnostics.should_log_train(global_step)
        )
        train_decoder = getattr(dist_utils.de_parallel(model), "decoder", None)
        if train_decoder is not None and getattr(train_decoder, "query_adapter", None) is not None:
            train_decoder.collect_train_diagnostics = bool(diagnostic_step)
        if diagnostic_step and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        step_start = _synchronize_for_measurement(device, diagnostic_step)
        samples = samples.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]
        transfer_end = _synchronize_for_measurement(device, diagnostic_step)
        metas = dict(epoch=epoch, step=i, global_step=global_step,
                     epoch_step=len(data_loader), collect_diagnostics=diagnostic_step)

        teacher_encoder_output_for_distillation = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_encoder_output_for_distillation = teacher_model(samples).detach()
        teacher_end = _synchronize_for_measurement(device, diagnostic_step)

        gradient_before_clip = None
        gradient_after_clip = None
        amp_scale_before = None
        amp_scale_after = None
        optimizer_step_skipped = False

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets,
                                teacher_encoder_output=teacher_encoder_output_for_distillation)
            forward_end = _synchronize_for_measurement(device, diagnostic_step)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            loss_end = _synchronize_for_measurement(device, diagnostic_step)

            loss = sum(loss_dict.values())
            amp_scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            backward_end = _synchronize_for_measurement(device, diagnostic_step)

            if max_norm > 0 or diagnostic_step:
                scaler.unscale_(optimizer)
            if diagnostic_step:
                gradient_before_clip = _gradient_statistics(model)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if diagnostic_step:
                    gradient_after_clip = _gradient_statistics(model)

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            gradient_end = _synchronize_for_measurement(device, diagnostic_step)
            scaler.step(optimizer)
            scaler.update()
            amp_scale_after = float(scaler.get_scale())
            # GradScaler applies its backoff factor only when found_inf caused
            # the optimizer step to be skipped. Scale growth/equality means a
            # real step was taken.
            optimizer_step_skipped = amp_scale_after < amp_scale_before
            amp_skipped_steps += int(optimizer_step_skipped)
            amp_scale_min = min(amp_scale_min, amp_scale_after)
            optimizer.zero_grad()
            optimizer_end = _synchronize_for_measurement(device, diagnostic_step)

        else:
            outputs = model(samples, targets=targets,
                            teacher_encoder_output=teacher_encoder_output_for_distillation) # NEW kwarg
            forward_end = _synchronize_for_measurement(device, diagnostic_step)
            loss_dict = criterion(outputs, targets, **metas)
            loss_end = _synchronize_for_measurement(device, diagnostic_step)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            backward_end = _synchronize_for_measurement(device, diagnostic_step)

            if diagnostic_step:
                gradient_before_clip = _gradient_statistics(model)

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if diagnostic_step:
                    gradient_after_clip = _gradient_statistics(model)

            gradient_end = _synchronize_for_measurement(device, diagnostic_step)
            optimizer.step()
            optimizer_end = _synchronize_for_measurement(device, diagnostic_step)

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        if diagnostic_step:
            end_time = _synchronize_for_measurement(device, True)
            observations = _training_observations(samples, targets, outputs)
            diagnostics.record_train_step({
                "epoch": epoch, "step": i, "global_step": global_step,
                "loss_total": loss_value,
                "losses": loss_dict_reduced,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "weight_decays": [group.get("weight_decay", 0.0) for group in optimizer.param_groups],
                "loss_weights": getattr(criterion, "weight_dict", None),
                "amp_enabled": scaler is not None,
                "amp": {
                    "scale_before": amp_scale_before,
                    "scale_after": amp_scale_after,
                    "optimizer_step_skipped": optimizer_step_skipped,
                },
                "clip_max_norm": max_norm,
                "timing_ms": {
                    "data_loader_wait": data_loader_wait_ms,
                    "host_to_device": (transfer_end - step_start) * 1000.0,
                    "teacher": (teacher_end - transfer_end) * 1000.0,
                    "forward": (forward_end - teacher_end) * 1000.0,
                    "criterion": (loss_end - forward_end) * 1000.0,
                    "backward": (backward_end - loss_end) * 1000.0,
                    "gradient_inspection_and_clip": (gradient_end - backward_end) * 1000.0,
                    "optimizer": (optimizer_end - gradient_end) * 1000.0,
                    "step_total": (end_time - step_start) * 1000.0,
                    "throughput_images_per_second": len(samples) /
                        max(end_time - step_start, 1e-9),
                    "cuda_synchronized": device.type == "cuda",
                },
                "gpu_memory": diagnostics.memory_snapshot(device),
                "gradients_before_clip": gradient_before_clip,
                "gradients_after_clip": gradient_after_clip,
                "main_hungarian_matches": getattr(criterion, "last_diagnostics", None),
                **observations,
            })
        loader_wait_start = time.perf_counter()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    statistics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if scaler is not None:
        statistics.update({
            "amp_scale_final": float(scaler.get_scale()),
            "amp_scale_min": amp_scale_min,
            "amp_skipped_steps": amp_skipped_steps,
        })
    return statistics, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor,
             data_loader, coco_evaluator: CocoEvaluator, device, **kwargs):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    diagnostics = kwargs.get("diagnostics")
    epoch = kwargs.get("epoch")
    diagnostic_dataset = getattr(
        coco_evaluator, "diagnostic_dataset", coco_evaluator.dataset)
    diagnostic_eval = bool(
        diagnostics is not None and diagnostics.enabled and "rbox" in iou_types)
    full_eval_records = diagnostic_eval and diagnostics.needs_full_eval_records()
    model_module = dist_utils.de_parallel(model)
    decoder = getattr(model_module, "decoder", None)
    previous_diagnostic_mode = None
    previous_attention_mode = None
    stage_evaluators = None
    if diagnostic_eval:
        diagnostics.start_evaluation(
            epoch, diagnostic_dataset, split="val",
            model_source=kwargs.get("model_source"))
        if decoder is not None and hasattr(decoder, "set_diagnostic_mode"):
            previous_diagnostic_mode = bool(
                getattr(getattr(decoder, "decoder", None), "diagnostic_mode", False))
            previous_attention_mode = bool(
                getattr(getattr(decoder, "decoder", None),
                        "diagnostic_attention_mode", False))

    loader_wait_start = time.perf_counter()
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        yielded_at = time.perf_counter()
        data_loader_wait_ms = (yielded_at - loader_wait_start) * 1000.0
        if diagnostic_eval and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        batch_start = _synchronize_for_measurement(device, diagnostic_eval)
        samples = samples.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()}
                   for t in targets]
        transfer_end = _synchronize_for_measurement(device, diagnostic_eval)

        if diagnostic_eval and decoder is not None and hasattr(decoder, "set_diagnostic_mode"):
            capture_attention = diagnostics.needs_detailed_eval_layers()
            decoder.set_diagnostic_mode(
                diagnostics.needs_layerwise_eval() or capture_attention,
                capture_attention=capture_attention,
            )
        if getattr(dist_utils.de_parallel(model), "requires_image_context", False):
            outputs = model(samples, targets=targets)
        else:
            outputs = model(samples)
        forward_end = _synchronize_for_measurement(device, diagnostic_eval)

        if 'rbox' in iou_types:
            if full_eval_records:
                results, post_diagnostics = postprocessor(
                    outputs, targets, return_diagnostics=True)
            else:
                results = postprocessor(outputs, targets)
        else:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)
        postprocess_end = _synchronize_for_measurement(device, diagnostic_eval)

        if full_eval_records:
            matching = criterion.matcher(outputs, targets, return_costs=True)
            matching_end = _synchronize_for_measurement(device, True)
            diagnostics.record_evaluation_batch(
                outputs, targets, results, post_diagnostics, matching,
                diagnostic_dataset,
                timings={
                    "batch_size": len(targets),
                    "data_loader_wait_ms": data_loader_wait_ms,
                    "host_to_device_ms": (transfer_end - batch_start) * 1000.0,
                    "forward_ms": (forward_end - transfer_end) * 1000.0,
                    "postprocess_and_nms_ms": (postprocess_end - forward_end) * 1000.0,
                    "diagnostic_matching_ms": (matching_end - postprocess_end) * 1000.0,
                    "forward_images_per_second": len(targets) /
                        max(forward_end - transfer_end, 1e-9),
                    "cuda_synchronized": device.type == "cuda",
                },
                device=device,
            )

        record_end = _synchronize_for_measurement(device, diagnostic_eval)
        if diagnostic_eval:
            if not full_eval_records:
                diagnostics.record_compact_predictions(outputs, targets, results, diagnostic_dataset, postprocessor)
            serialization_end = _synchronize_for_measurement(device, True)
            diagnostics.record_evaluation_performance({
                "batch_size":len(targets), "data_loader_wait_ms":data_loader_wait_ms,
                "host_to_device_ms":(transfer_end-batch_start)*1000,
                "forward_ms":(forward_end-transfer_end)*1000,
                "postprocess_and_nms_ms":(postprocess_end-forward_end)*1000,
                "diagnostic_matching_and_records_ms":(record_end-postprocess_end)*1000,
                "prediction_archive_ms":(serialization_end-record_end)*1000,
            })
            if diagnostics.needs_layerwise_eval():
                stage_outputs = _refinement_stage_outputs(outputs)
                if not stage_outputs:
                    raise RuntimeError(
                        "Full-validation layerwise diagnostics were requested, "
                        "but the model did not expose the refinement-stage contract")
                if stage_evaluators is None:
                    if not hasattr(coco_evaluator, "clone_empty"):
                        raise TypeError(
                            "Layerwise diagnostics require an evaluator implementing clone_empty()")
                    stage_evaluators = OrderedDict(
                        (name, coco_evaluator.clone_empty()) for name in stage_outputs)
                if tuple(stage_evaluators) != tuple(stage_outputs):
                    raise RuntimeError("Refinement stage order changed between validation batches")
                for name, stage_output in stage_outputs.items():
                    stage_predictions = postprocessor(stage_output, targets)
                    stage_evaluators[name].update({
                        int(target["image_id"].item()): prediction
                        for target, prediction in zip(targets, stage_predictions)
                    })

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        loader_wait_start = time.perf_counter()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    refinement_stages = None
    if stage_evaluators is not None:
        for evaluator in stage_evaluators.values():
            evaluator.synchronize_between_processes()
            if hasattr(evaluator, "reuse_ground_truth_cache_from"):
                evaluator.reuse_ground_truth_cache_from(coco_evaluator)
            evaluator.accumulate(verbose=False)
        stage_records = OrderedDict(
            (name, _evaluator_metrics(evaluator))
            for name, evaluator in stage_evaluators.items()
        )
        primary_metric = getattr(coco_evaluator, "selection_metric", "mAP50_75_DOTA07")
        first_name, final_name = next(iter(stage_records)), next(reversed(stage_records))
        first_value = stage_records[first_name]["metrics"].get(primary_metric)
        final_value = stage_records[final_name]["metrics"].get(primary_metric)
        refinement_stages = {
            "stage_order": list(stage_records),
            "stages": stage_records,
            "primary_metric": primary_metric,
            "pre_to_final_delta": (
                final_value - first_value
                if first_value is not None and final_value is not None else None),
            "final_stage_matches_primary_evaluator": (
                list(stage_records[final_name]["stats"])
                == list(getattr(coco_evaluator, "stats", []))
            ),
        }
    if diagnostic_eval:
        diagnostics.record_global_merge(coco_evaluator)
        diagnostics.finish_evaluation(coco_evaluator, refinement_stages)
        if previous_diagnostic_mode is not None:
            decoder.set_diagnostic_mode(
                previous_diagnostic_mode,
                capture_attention=previous_attention_mode,
            )

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if 'rbox' in iou_types:
            stats['dota_eval_rbox'] = coco_evaluator.stats.tolist()

    return stats, coco_evaluator
