"""Rotated bounding-box geometry used by the CODrone OBB baseline.

The model-facing convention is ``(cx, cy, w, h, angle / pi)`` where the
first four values are normalized by the image width/height, the angle is in
``[0, 1)``, and the pixel-space representation has ``w >= h``.  Public
helpers that operate on pixel boxes use radians in ``[0, pi)``.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import Tensor


ANGLE_PERIOD = math.pi


def _empty_like(boxes: Tensor, last_dim: int) -> Tensor:
    return boxes.new_empty((*boxes.shape[:-1], last_dim))


def regularize_rboxes(boxes: Tensor, normalized_angle: bool = False) -> Tensor:
    """Canonicalize boxes to long-edge form without changing their geometry."""
    if boxes.shape[-1] != 5:
        raise ValueError(f"Expected (..., 5) rotated boxes, got {tuple(boxes.shape)}")
    if boxes.numel() == 0:
        return boxes.clone()

    period = 1.0 if normalized_angle else ANGLE_PERIOD
    out = boxes.clone()
    # Clone the views: the width assignment below is in-place and must not
    # change the value later used to write the canonical height.
    w, h, angle = out[..., 2].clone(), out[..., 3].clone(), out[..., 4].clone()
    swap = h > w
    out[..., 2] = torch.where(swap, h, w).clamp_min(1e-7)
    out[..., 3] = torch.where(swap, w, h).clamp_min(1e-7)
    out[..., 4] = torch.remainder(
        angle + swap.to(angle.dtype) * (period / 2.0), period)
    return out


def angle_distance(angle1: Tensor, angle2: Tensor, normalized: bool = True) -> Tensor:
    """Shortest unsigned distance for half-turn-periodic box orientations."""
    period = 1.0 if normalized else ANGLE_PERIOD
    delta = torch.remainder(torch.abs(angle1 - angle2), period)
    return torch.minimum(delta, period - delta)


def rbox_to_corners(boxes: Tensor, normalized_angle: bool = False) -> Tensor:
    """Convert ``(..., 5)`` boxes to ordered ``(..., 4, 2)`` corners."""
    if boxes.shape[-1] != 5:
        raise ValueError(f"Expected (..., 5) rotated boxes, got {tuple(boxes.shape)}")
    if boxes.numel() == 0:
        return _empty_like(boxes, 8).reshape(*boxes.shape[:-1], 4, 2)

    angle = boxes[..., 4] * ANGLE_PERIOD if normalized_angle else boxes[..., 4]
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    template = boxes.new_tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    local = template * torch.stack((w, h), dim=-1).unsqueeze(-2)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (cos_a, -sin_a, sin_a, cos_a), dim=-1).reshape(*angle.shape, 2, 2)
    corners = torch.matmul(local, rotation.transpose(-1, -2))
    return corners + torch.stack((cx, cy), dim=-1).unsqueeze(-2)


def corners_to_rboxes(corners: Tensor) -> Tensor:
    """Convert ordered rectangular quadrilaterals to canonical pixel rboxes.

    CODrone/DOTA annotations list consecutive vertices.  Computing the two
    adjacent edges directly avoids OpenCV angle-version conventions.
    """
    if corners.shape[-2:] != (4, 2):
        raise ValueError(f"Expected (..., 4, 2) corners, got {tuple(corners.shape)}")
    if corners.numel() == 0:
        return corners.new_empty((*corners.shape[:-2], 5))

    center = corners.mean(dim=-2)
    edge1 = corners[..., 1, :] - corners[..., 0, :]
    edge2 = corners[..., 2, :] - corners[..., 1, :]
    len1 = torch.linalg.vector_norm(edge1, dim=-1).clamp_min(1e-7)
    len2 = torch.linalg.vector_norm(edge2, dim=-1).clamp_min(1e-7)
    use_first = len1 >= len2
    long_edge = torch.where(use_first.unsqueeze(-1), edge1, edge2)
    width = torch.maximum(len1, len2)
    height = torch.minimum(len1, len2)
    angle = torch.remainder(torch.atan2(long_edge[..., 1], long_edge[..., 0]), ANGLE_PERIOD)
    return torch.cat(
        (center, width.unsqueeze(-1), height.unsqueeze(-1), angle.unsqueeze(-1)),
        dim=-1)


def polygon_to_rbox(coords: Iterable[float], dtype: torch.dtype = torch.float32) -> Tensor:
    values = torch.as_tensor(list(coords), dtype=dtype)
    if values.numel() != 8:
        raise ValueError(f"DOTA polygon must contain 8 coordinates, got {values.numel()}")
    return corners_to_rboxes(values.reshape(1, 4, 2))[0]


def normalize_rboxes(boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """Normalize pixel boxes for a ``(width, height)`` canvas."""
    width, height = image_size
    factor = boxes.new_tensor([width, height, width, height, ANGLE_PERIOD])
    return regularize_rboxes(boxes, normalized_angle=False) / factor


def denormalize_rboxes(boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
    width, height = image_size
    factor = boxes.new_tensor([width, height, width, height, ANGLE_PERIOD])
    return regularize_rboxes(boxes * factor, normalized_angle=False)


def _gaussian_parameters(boxes: Tensor, normalized_angle: bool = True):
    angle = boxes[..., 4] * ANGLE_PERIOD if normalized_angle else boxes[..., 4]
    wh = boxes[..., 2:4].clamp_min(1e-7)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (cos_a, -sin_a, sin_a, cos_a), dim=-1).reshape(*angle.shape, 2, 2)
    scale = torch.diag_embed(0.5 * wh)
    covariance = rotation @ scale.square() @ rotation.transpose(-1, -2)
    eye = torch.eye(2, device=boxes.device, dtype=boxes.dtype)
    covariance = covariance + eye * 1e-9
    return boxes[..., :2], covariance


def _postprocess_distance(distance: Tensor, sqrt: bool = True) -> Tensor:
    if sqrt:
        distance = distance.clamp_min(1e-7).sqrt()
    distance = torch.log1p(distance.clamp_min(0))
    return 1.0 - 1.0 / (1.0 + distance)


def aligned_kld_loss(pred: Tensor, target: Tensor, normalized_angle: bool = True) -> Tensor:
    """Differentiable KLD geometry loss for aligned rotated boxes."""
    if pred.shape != target.shape or pred.shape[-1] != 5:
        raise ValueError(f"Aligned KLD expects equal (..., 5) shapes, got {pred.shape}, {target.shape}")
    if pred.numel() == 0:
        return pred.new_empty(pred.shape[:-1])
    xy_p, sigma_p = _gaussian_parameters(pred, normalized_angle)
    xy_t, sigma_t = _gaussian_parameters(target, normalized_angle)
    inv_p = torch.linalg.inv(sigma_p)
    delta = (xy_p - xy_t).unsqueeze(-1)
    center = 0.5 * (delta.transpose(-1, -2) @ inv_p @ delta).squeeze(-1).squeeze(-1)
    trace = 0.5 * torch.diagonal(inv_p @ sigma_t, dim1=-2, dim2=-1).sum(-1)
    log_det = 0.5 * (torch.linalg.slogdet(sigma_p).logabsdet -
                     torch.linalg.slogdet(sigma_t).logabsdet)
    distance = (center + trace + log_det - 1.0).clamp_min(0)
    return _postprocess_distance(distance)


def pairwise_kld_cost(pred: Tensor, target: Tensor, normalized_angle: bool = True) -> Tensor:
    """Pairwise KLD cost with shape ``(num_pred, num_target)``."""
    if pred.shape[-1] != 5 or target.shape[-1] != 5:
        raise ValueError("Pairwise KLD expects five-parameter rotated boxes")
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros((pred.shape[0], target.shape[0]))
    p = pred[:, None, :].expand(-1, target.shape[0], -1)
    t = target[None, :, :].expand(pred.shape[0], -1, -1)
    return aligned_kld_loss(p, t, normalized_angle)


def pairwise_chamfer_cost(pred: Tensor, target: Tensor, normalized_angle: bool = True) -> Tensor:
    """Bidirectional mean corner Chamfer distance used by O2 matching."""
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros((pred.shape[0], target.shape[0]))
    corners1 = rbox_to_corners(pred, normalized_angle=normalized_angle)
    corners2 = rbox_to_corners(target, normalized_angle=normalized_angle)
    distances = torch.linalg.vector_norm(
        corners1[:, None, :, None, :] - corners2[None, :, None, :, :], dim=-1)
    return distances.min(dim=-1).values.mean(dim=-1) + \
        distances.min(dim=-2).values.mean(dim=-1)


def rotated_iou(boxes1: Tensor, boxes2: Tensor, aligned: bool = False,
                normalized_angle: bool = True) -> Tensor:
    """Rotated IoU backed by MMCV's tested CPU/CUDA operator."""
    try:
        from mmcv.ops import box_iou_rotated
    except ImportError as exc:  # pragma: no cover - explicit runtime diagnosis
        raise RuntimeError(
            "CODrone OBB requires MMCV built with rotated ops (box_iou_rotated).") from exc
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        shape = (boxes1.shape[0],) if aligned else (boxes1.shape[0], boxes2.shape[0])
        return boxes1.new_zeros(shape)
    b1 = boxes1.clone()
    b2 = boxes2.clone()
    if normalized_angle:
        b1[..., 4] *= ANGLE_PERIOD
        b2[..., 4] *= ANGLE_PERIOD
    overlaps = box_iou_rotated(b1.float(), b2.float(), aligned=aligned, clockwise=True)
    return overlaps.to(dtype=boxes1.dtype)


def class_aware_rotated_nms(boxes: Tensor, scores: Tensor, labels: Tensor,
                            iou_threshold: float, max_output: int | None = None) -> Tensor:
    """Return score-sorted indices after per-class rotated NMS."""
    try:
        from mmcv.ops import nms_rotated
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "CODrone OBB requires MMCV built with rotated ops (nms_rotated).") from exc
    keeps = []
    for label in labels.unique(sorted=True):
        class_idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        if class_idx.numel() == 0:
            continue
        _, local_keep = nms_rotated(
            boxes[class_idx].float(), scores[class_idx].float(), iou_threshold)
        keeps.append(class_idx[local_keep])
    if not keeps:
        return labels.new_empty((0,), dtype=torch.long)
    keep = torch.cat(keeps)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    return keep[:max_output] if max_output is not None else keep
