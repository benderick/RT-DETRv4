"""Capped contrastive denoising queries for oriented D-FINE."""

from __future__ import annotations

import torch

from .rotated_box_ops import regularize_rboxes
from .utils import inverse_sigmoid


def get_rotated_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
):
    if num_denoising <= 0 or not targets:
        return None, None, None, None
    device = targets[0]["labels"].device
    max_selected = max(1, num_denoising // 2)
    selected_indices = []
    for target in targets:
        count = len(target["labels"])
        if count > max_selected:
            selected_indices.append(torch.randperm(count, device=device)[:max_selected].sort().values)
        else:
            selected_indices.append(torch.arange(count, device=device))
    counts = [len(index) for index in selected_indices]
    max_gt = max(counts)
    if max_gt == 0:
        return None, None, None, None

    # One group contains a positive and a negative copy. Unlike the original
    # implementation this bound is strict even for a crowded 400-object image.
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
        noise_strength = 1.0 + negative
        center_noise = (torch.rand_like(boxes[..., :2]) * 2 - 1) * boxes[..., 2:4] * 0.5
        boxes[..., :2] += center_noise * noise_strength * box_noise_scale
        size_noise = (torch.rand_like(boxes[..., 2:4]) * 2 - 1) * 0.5
        boxes[..., 2:4] *= torch.exp(size_noise * noise_strength * box_noise_scale)
        angle_noise = (torch.rand_like(boxes[..., 4:5]) * 2 - 1) * 0.25
        boxes[..., 4:5] += angle_noise * noise_strength * box_noise_scale
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
    meta = {
        "dn_positive_idx": tuple(dn_positive_idx),
        "dn_target_idx": tuple(dn_target_idx),
        "dn_num_group": num_group,
        "dn_num_split": [total_dn, num_queries],
    }
    return query_logits, query_boxes_unact, attention_mask, meta
