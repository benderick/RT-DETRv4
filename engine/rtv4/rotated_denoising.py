"""Capped contrastive denoising queries for oriented D-FINE."""

from __future__ import annotations

import math

import torch

from .rotated_box_ops import regularize_rboxes
from .utils import inverse_sigmoid


OCD_MODES = ("standard", "box", "angle", "geometric", "probability", "none")
OCD_CROWDED_POLICIES = ("released_dynamic", "strict_budget_random")


def _signed_annulus_like(value, inner, outer):
    """Sample uniformly from ``[-outer,-inner) U [inner,outer)``."""

    if outer < inner or inner < 0:
        raise ValueError("OCD negative-noise bounds require 0 <= inner <= outer")
    magnitude = inner + (outer - inner) * torch.rand_like(value)
    sign = torch.where(torch.rand_like(value) < 0.5, -torch.ones_like(value),
                       torch.ones_like(value))
    return sign * magnitude


def apply_ocd_box_noise(boxes, negative, lambda1=1.0, lambda2=2.0, scale=1.0):
    """Apply the paper's vertex-coordinate box noise in the local box frame."""

    if lambda1 < 0 or lambda2 < lambda1:
        raise ValueError("OCD box noise requires 0 <= lambda1 <= lambda2")
    xyxy = torch.cat((boxes[..., :2] - boxes[..., 2:4] / 2,
                      boxes[..., :2] + boxes[..., 2:4] / 2), dim=-1)
    coordinate_scale = torch.cat((boxes[..., 2:4], boxes[..., 2:4]), dim=-1) / 2
    positive_delta = (2 * torch.rand_like(xyxy) - 1) * lambda1
    negative_delta = _signed_annulus_like(xyxy, lambda1, lambda2)
    delta = torch.where(negative.expand_as(xyxy).bool(), negative_delta, positive_delta)
    xyxy = xyxy + delta * coordinate_scale * scale
    first, second = xyxy[..., :2], xyxy[..., 2:]
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    result = boxes.clone()
    result[..., :2] = (low + high) / 2
    result[..., 2:4] = (high - low).clamp_min(1e-5)
    return result


def apply_ocd_angle_noise(boxes, negative, lambda3=9.0, lambda4=18.0, scale=1.0):
    """Apply O^2 angle noise; normalized angles preserve the paper's ratio."""

    if lambda3 < 0 or lambda4 < lambda3:
        raise ValueError("OCD angle noise requires 0 <= lambda3 <= lambda4")
    theta = boxes[..., 4:5]
    positive_factor = (2 * torch.rand_like(theta) - 1) * lambda3
    negative_factor = _signed_annulus_like(theta, lambda3, lambda4)
    factor = torch.where(negative.bool(), negative_factor, positive_factor)
    result = boxes.clone()
    result[..., 4:5] = theta + factor * theta * (scale / 18.0)
    return result


def apply_ocd_probability_noise(
    boxes, negative, lambda5=0.3, lambda6=0.6, scale=1.0,
):
    """Perturb the Gaussian covariance and reconstruct an oriented box.

    The reconstruction uses the original maximum side as the scale discarded
    by the normalized covariance in the paper.
    """

    if lambda5 < 0 or lambda6 < lambda5 or lambda6 >= 1:
        raise ValueError("OCD probability noise requires 0 <= lambda5 <= lambda6 < 1")
    theta = boxes[..., 4] * math.pi
    cosine, sine = theta.cos(), theta.sin()
    rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        *theta.shape, 2, 2)
    maximum_side = boxes[..., 2:4].amax(dim=-1, keepdim=True).clamp_min(1e-7)
    normalized_sides = boxes[..., 2:4] / maximum_side
    eigenvalues = normalized_sides.square() / 4
    covariance = rotation @ torch.diag_embed(eigenvalues) @ rotation.transpose(-1, -2)

    positive_lambda = torch.rand_like(theta) * lambda5
    negative_lambda = lambda5 + torch.rand_like(theta) * (lambda6 - lambda5)
    mixture = torch.where(negative.squeeze(-1).bool(), negative_lambda, positive_lambda)
    mixture = (mixture * scale).clamp(0, 1 - 1e-6)
    identity = torch.eye(2, dtype=boxes.dtype, device=boxes.device)
    covariance = ((1 - mixture)[..., None, None] * covariance +
                  mixture[..., None, None] * identity)
    values, vectors = torch.linalg.eigh(covariance)
    order = values.argsort(dim=-1, descending=True)
    values = values.gather(-1, order).clamp_min(1e-12)
    vectors = vectors.gather(-1, order[..., None, :].expand_as(vectors))
    major = vectors[..., :, 0]

    result = boxes.clone()
    result[..., 2:4] = 2 * values.sqrt() * maximum_side
    result[..., 4] = torch.remainder(torch.atan2(major[..., 1], major[..., 0]) / math.pi, 1.0)
    return result


def _apply_standard_noise(boxes, negative, scale):
    noise_strength = 1.0 + negative
    center_noise = (torch.rand_like(boxes[..., :2]) * 2 - 1) * boxes[..., 2:4] * 0.5
    boxes[..., :2] += center_noise * noise_strength * scale
    size_noise = (torch.rand_like(boxes[..., 2:4]) * 2 - 1) * 0.5
    boxes[..., 2:4] *= torch.exp(size_noise * noise_strength * scale)
    angle_noise = (torch.rand_like(boxes[..., 4:5]) * 2 - 1) * 0.25
    boxes[..., 4:5] += angle_noise * noise_strength * scale
    return boxes


def get_rotated_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    mode="standard",
    lambda1=1.0,
    lambda2=2.0,
    lambda3=9.0,
    lambda4=18.0,
    lambda5=0.3,
    lambda6=0.6,
    crowded_policy="strict_budget_random",
):
    if num_denoising <= 0 or not targets:
        return None, None, None, None
    if mode not in OCD_MODES:
        raise ValueError(f"Unknown OCD mode {mode!r}; expected one of {OCD_MODES}")
    if crowded_policy not in OCD_CROWDED_POLICIES:
        raise ValueError(
            f"Unknown OCD crowded policy {crowded_policy!r}; "
            f"expected one of {OCD_CROWDED_POLICIES}")
    device = targets[0]["labels"].device
    max_selected = max(1, num_denoising // 2)
    selected_indices = []
    for target in targets:
        count = len(target["labels"])
        if crowded_policy == "strict_budget_random" and count > max_selected:
            selected_indices.append(torch.randperm(count, device=device)[:max_selected].sort().values)
        else:
            selected_indices.append(torch.arange(count, device=device))
    counts = [len(index) for index in selected_indices]
    max_gt = max(counts)
    if max_gt == 0:
        return None, None, None, None

    # One group contains a positive and a negative copy. ``released_dynamic``
    # follows the released dynamic-group behavior: a crowded image can exceed
    # the requested query budget instead of silently losing GT supervision.
    # ``strict_budget_random`` instead caps the selected GT set and records
    # every dropped count/target index in metadata.
    num_group = max(1, num_denoising // (2 * max_gt))
    batch_size = len(targets)
    classes = torch.full((batch_size, max_gt), num_classes, dtype=torch.long, device=device)
    boxes = torch.zeros((batch_size, max_gt, 5), device=device)
    valid = torch.zeros((batch_size, max_gt), dtype=torch.bool, device=device)
    for batch_index, (target, chosen) in enumerate(zip(targets, selected_indices)):
        count = len(chosen)
        if count:
            classes[batch_index, :count] = target["labels"][chosen]
            boxes[batch_index, :count] = target["boxes"][chosen]
            valid[batch_index, :count] = True

    classes = classes.tile(1, 2 * num_group)
    boxes = boxes.tile(1, 2 * num_group, 1)
    valid = valid.tile(1, 2 * num_group)
    negative = torch.zeros((batch_size, max_gt * 2, 1), device=device)
    negative[:, max_gt:] = 1
    negative = negative.tile(1, num_group, 1)
    positive = (1 - negative).squeeze(-1).bool() & valid
    positive_indices = torch.nonzero(positive, as_tuple=False)
    dn_positive_idx = []
    for batch_index in range(batch_size):
        dn_positive_idx.append(positive_indices[positive_indices[:, 0] == batch_index, 1])

    if label_noise_ratio > 0:
        change = torch.rand_like(classes, dtype=torch.float) < label_noise_ratio * 0.5
        random_labels = torch.randint(0, num_classes, classes.shape, device=device)
        classes = torch.where(change & valid, random_labels, classes)

    if box_noise_scale > 0:
        if mode == "standard":
            boxes = _apply_standard_noise(boxes, negative, box_noise_scale)
        elif mode in {"box", "geometric"}:
            boxes = apply_ocd_box_noise(
                boxes, negative, lambda1, lambda2, box_noise_scale)
        if mode in {"angle", "geometric"}:
            boxes = apply_ocd_angle_noise(
                boxes, negative, lambda3, lambda4, box_noise_scale)
        elif mode == "probability":
            boxes = apply_ocd_probability_noise(
                boxes, negative, lambda5, lambda6, box_noise_scale)
        if mode != "none":
            boxes[..., :4].clamp_(min=1e-5, max=1 - 1e-5)
            boxes[..., :2].clamp_(min=1e-5, max=1 - 1e-5)
            boxes = regularize_rboxes(boxes, normalized_angle=True).clamp(1e-5, 1 - 1e-5)

    query_logits = class_embed(classes)
    query_boxes_unact = inverse_sigmoid(boxes.clamp(1e-5, 1 - 1e-5))
    total_dn = int(2 * max_gt * num_group)
    total_queries = total_dn + num_queries
    attention_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
    attention_mask[total_dn:, :total_dn] = True
    group_size = 2 * max_gt
    for group_index in range(num_group):
        start, end = group_index * group_size, (group_index + 1) * group_size
        attention_mask[start:end, :start] = True
        attention_mask[start:end, end:total_dn] = True

    dn_target_idx = [index.tile(num_group) for index in selected_indices]
    dropped_indices = []
    for target, selected in zip(targets, selected_indices):
        kept = torch.zeros(len(target["labels"]), dtype=torch.bool, device=device)
        kept[selected] = True
        dropped_indices.append(torch.nonzero(~kept, as_tuple=False).squeeze(1))
    meta = {
        "dn_positive_idx": tuple(dn_positive_idx),
        "dn_target_idx": tuple(dn_target_idx),
        "dn_num_group": num_group,
        "dn_num_split": [total_dn, num_queries],
        "dn_noise_mode": mode,
        "dn_noise_lambdas": [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6],
        "dn_crowded_policy": crowded_policy,
        "dn_requested_query_budget": int(num_denoising),
        "dn_actual_query_count": total_dn,
        "dn_budget_exceeded": total_dn > num_denoising,
        "dn_original_gt_counts": [len(target["labels"]) for target in targets],
        "dn_selected_gt_counts": counts,
        "dn_dropped_gt_counts": [
            len(target["labels"]) - selected
            for target, selected in zip(targets, counts)
        ],
        "dn_selected_target_idx": tuple(selected_indices),
        "dn_dropped_target_idx": tuple(dropped_indices),
    }
    return query_logits, query_boxes_unact, attention_mask, meta
