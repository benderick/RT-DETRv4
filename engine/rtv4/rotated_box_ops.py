"""Shared rotated bounding-box geometry.

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

_ROTATED_IOU_PAIR_CHUNK = 65536
_GEOMETRY_TOLERANCE = 1e-10


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


def _postprocess_distance(
    distance: Tensor,
    *,
    sqrt: bool = True,
    fun: str = "log1p",
    tau: float = 1.0,
) -> Tensor:
    """Apply the configurable MMRotate Gaussian-distance post-processing.

    ``sqrt`` belongs to KLD itself and is deliberately applied before
    ``fun``.  O^2's released configurations use
    ``sqrt=False, fun='log1p', tau=1``. Other choices remain explicit
    experiment settings.
    """

    if sqrt:
        distance = distance.clamp_min(1e-7).sqrt()
    if fun == "log1p":
        distance = torch.log1p(distance)
    elif fun == "sqrt":
        distance = distance.clamp_min(1e-7).sqrt()
    elif fun != "none":
        raise ValueError(f"Unsupported KLD post-processing function: {fun!r}")
    return 1.0 - 1.0 / (tau + distance) if tau >= 1.0 else distance


def aligned_kld_loss(
    pred: Tensor,
    target: Tensor,
    normalized_angle: bool = True,
    *,
    sqrt: bool = True,
    fun: str = "log1p",
    tau: float = 1.0,
) -> Tensor:
    """Differentiable KLD geometry loss for aligned rotated boxes.

    The argument names and operation order mirror MMRotate's public
    ``kld_loss`` so experiments can state their KLD semantics exactly.
    """
    if pred.shape != target.shape or pred.shape[-1] != 5:
        raise ValueError(f"Aligned KLD expects equal (..., 5) shapes, got {pred.shape}, {target.shape}")
    if pred.numel() == 0:
        return pred.new_empty(pred.shape[:-1])
    # Evaluate the same Gaussian KLD directly in the predicted box's local
    # frame.  Forming a rotated covariance and then dividing its adjugate by
    # ``det(Sigma)`` is algebraically redundant: for a thin rotated box the
    # determinant becomes a subtraction of nearly equal float32 products and
    # can round to zero although both side lengths are finite.  The closed
    # form below has the same values/gradients on ordinary boxes and remains
    # defined over the full clamped OBB domain.
    working_dtype = torch.promote_types(pred.dtype, target.dtype)
    pred = pred.to(dtype=working_dtype)
    target = target.to(dtype=working_dtype)
    period = ANGLE_PERIOD if normalized_angle else 1.0
    pred_angle = pred[..., 4] * period
    relative_angle = (target[..., 4] - pred[..., 4]) * period
    pred_wh = pred[..., 2:4].clamp(min=1e-7, max=1e7)
    target_wh = target[..., 2:4].clamp(min=1e-7, max=1e7)
    pred_w, pred_h = pred_wh.unbind(-1)
    target_w, target_h = target_wh.unbind(-1)

    delta = pred[..., :2] - target[..., :2]
    cos_p, sin_p = pred_angle.cos(), pred_angle.sin()
    local_x = cos_p * delta[..., 0] + sin_p * delta[..., 1]
    local_y = -sin_p * delta[..., 0] + cos_p * delta[..., 1]
    center = 2.0 * (
        (local_x / pred_w).square() + (local_y / pred_h).square()
    )

    cos_r, sin_r = relative_angle.cos(), relative_angle.sin()
    cos2, sin2 = cos_r.square(), sin_r.square()
    trace = 0.5 * (
        target_w.square() * (cos2 / pred_w.square() + sin2 / pred_h.square())
        + target_h.square() * (sin2 / pred_w.square() + cos2 / pred_h.square())
    )
    log_det = torch.log(
        (pred_w * pred_h) / (target_w * target_h)
    )
    distance = center + trace + log_det - 1.0
    return _postprocess_distance(
        distance, sqrt=sqrt, fun=fun, tau=float(tau))


def pairwise_kld_cost(
    pred: Tensor,
    target: Tensor,
    normalized_angle: bool = True,
    *,
    sqrt: bool = True,
    fun: str = "log1p",
    tau: float = 1.0,
) -> Tensor:
    """Pairwise KLD cost with shape ``(num_pred, num_target)``."""
    if pred.shape[-1] != 5 or target.shape[-1] != 5:
        raise ValueError("Pairwise KLD expects five-parameter rotated boxes")
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros((pred.shape[0], target.shape[0]))
    p = pred[:, None, :].expand(-1, target.shape[0], -1)
    t = target[None, :, :].expand(pred.shape[0], -1, -1)
    return aligned_kld_loss(
        p, t, normalized_angle, sqrt=sqrt, fun=fun, tau=tau)


def pairwise_chamfer_cost(
    pred: Tensor,
    target: Tensor,
    normalized_angle: bool = True,
) -> Tensor:
    """Bidirectional mean Euclidean corner Chamfer distance.

    This is the executable definition in the authors' released O² detector.
    """
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return pred.new_zeros((pred.shape[0], target.shape[0]))
    corners1 = rbox_to_corners(pred, normalized_angle=normalized_angle)
    corners2 = rbox_to_corners(target, normalized_angle=normalized_angle)
    delta = corners1[:, None, :, None, :] - corners2[None, :, None, :, :]
    distances = torch.linalg.vector_norm(delta, dim=-1)
    return distances.min(dim=-1).values.mean(dim=-1) + \
        distances.min(dim=-2).values.mean(dim=-1)


def _cross_2d(first: Tensor, second: Tensor) -> Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _quadrilateral_area(corners: Tensor) -> Tensor:
    return 0.5 * _cross_2d(
        corners, torch.roll(corners, shifts=-1, dims=-2)).sum(dim=-1).abs()


def _points_inside_quadrilateral(points: Tensor, polygon: Tensor) -> Tensor:
    """Return a mask for convex quadrilaterals in either winding order."""

    edges = torch.roll(polygon, shifts=-1, dims=-2) - polygon
    relative = points.unsqueeze(-2) - polygon.unsqueeze(-3)
    sides = _cross_2d(edges.unsqueeze(-3), relative)
    return ((sides >= -_GEOMETRY_TOLERANCE).all(dim=-1) |
            (sides <= _GEOMETRY_TOLERANCE).all(dim=-1))


def _aligned_quadrilateral_iou(first: Tensor, second: Tensor) -> Tensor:
    """Robust IoU for aligned convex quadrilateral pairs.

    Every pair is translated and isotropically normalized before geometric
    predicates are evaluated in float64.  Its intersection polygon consists
    exactly of contained corners and segment intersections; sorting those
    candidates around their centroid then gives its area.  This avoids the
    coincident-edge failure of float32 rotated-box kernels, where two boxes
    differing by only a few ulps can incorrectly return IoU 0 or 1/3.
    """

    if first.shape != second.shape or first.shape[-2:] != (4, 2):
        raise ValueError(
            "Aligned quadrilateral IoU expects equal (..., 4, 2) shapes")
    if not len(first):
        return first.new_empty(0)

    first = first.to(torch.float64)
    second = second.to(torch.float64)
    combined = torch.cat((first, second), dim=-2)
    origin = combined.mean(dim=-2, keepdim=True)
    span = (combined.amax(dim=-2) - combined.amin(dim=-2)).amax(dim=-1)
    scale = span.clamp_min(torch.finfo(torch.float64).tiny)
    first = (first - origin) / scale[:, None, None]
    second = (second - origin) / scale[:, None, None]

    first_inside = _points_inside_quadrilateral(first, second)
    second_inside = _points_inside_quadrilateral(second, first)

    first_start = first[:, :, None, :]
    first_edge = (
        torch.roll(first, shifts=-1, dims=-2) - first)[:, :, None, :]
    second_start = second[:, None, :, :]
    second_edge = (
        torch.roll(second, shifts=-1, dims=-2) - second)[:, None, :, :]
    denominator = _cross_2d(first_edge, second_edge)
    relative_start = second_start - first_start
    nonparallel = denominator.abs() > 1e-14
    safe_denominator = torch.where(
        nonparallel, denominator, torch.ones_like(denominator))
    first_fraction = _cross_2d(relative_start, second_edge) / safe_denominator
    second_fraction = _cross_2d(relative_start, first_edge) / safe_denominator
    intersects = (
        nonparallel &
        (first_fraction >= -_GEOMETRY_TOLERANCE) &
        (first_fraction <= 1.0 + _GEOMETRY_TOLERANCE) &
        (second_fraction >= -_GEOMETRY_TOLERANCE) &
        (second_fraction <= 1.0 + _GEOMETRY_TOLERANCE)
    )
    intersections = first_start + first_fraction[..., None] * first_edge

    candidates = torch.cat((
        first,
        second,
        intersections.reshape(len(first), 16, 2),
    ), dim=-2)
    valid = torch.cat((
        first_inside,
        second_inside,
        intersects.reshape(len(first), 16),
    ), dim=-1)
    count = valid.sum(dim=-1)
    centroid = (
        (candidates * valid[..., None]).sum(dim=-2) /
        count.clamp_min(1)[..., None]
    )
    angles = torch.atan2(
        candidates[..., 1] - centroid[:, None, 1],
        candidates[..., 0] - centroid[:, None, 0],
    )
    angles = torch.where(
        valid, angles, torch.full_like(angles, torch.inf))
    order = angles.argsort(dim=-1)
    ordered = candidates.gather(
        -2, order[..., None].expand(-1, -1, 2))
    ordered_valid = valid.gather(-1, order)

    adjacent_valid = ordered_valid[:, :-1] & ordered_valid[:, 1:]
    signed_twice_area = (
        _cross_2d(ordered[:, :-1], ordered[:, 1:]) * adjacent_valid
    ).sum(dim=-1)
    last_index = (count - 1).clamp_min(0)
    last = ordered.gather(
        1, last_index[:, None, None].expand(-1, 1, 2)).squeeze(1)
    signed_twice_area += _cross_2d(last, ordered[:, 0]) * (count >= 2)
    intersection = torch.where(
        count >= 3,
        0.5 * signed_twice_area.abs(),
        torch.zeros_like(signed_twice_area),
    )
    first_area = _quadrilateral_area(first)
    second_area = _quadrilateral_area(second)
    intersection = torch.minimum(intersection, torch.minimum(first_area, second_area))
    union = (first_area + second_area - intersection).clamp_min(
        torch.finfo(torch.float64).tiny)
    return (intersection / union).clamp(0.0, 1.0)


@torch.no_grad()
def rotated_iou(boxes1: Tensor, boxes2: Tensor, aligned: bool = False,
                model_space: bool = True) -> Tensor:
    """Numerically robust rotated IoU with an explicit coordinate contract.

    ``model_space=True`` accepts normalized ``(cx, cy, w, h, theta / pi)``;
    ``model_space=False`` accepts original-image pixels and radians.  The
    result is computed from convex polygons in pair-normalized float64 space,
    making pixel/model results invariant to translation and common scale.
    """

    if boxes1.ndim != 2 or boxes2.ndim != 2 or \
            boxes1.shape[-1] != 5 or boxes2.shape[-1] != 5:
        raise ValueError(
            "Rotated IoU expects two matrices shaped (num_boxes, 5)")
    if boxes1.device != boxes2.device:
        raise ValueError("Rotated IoU inputs must be on the same device")
    if aligned and len(boxes1) != len(boxes2):
        raise ValueError("Aligned rotated IoU requires equal box counts")
    output_shape = (len(boxes1),) if aligned else (len(boxes1), len(boxes2))
    if not len(boxes1) or not len(boxes2):
        return boxes1.new_zeros(output_shape)

    # Promote before trigonometry and centre-plus-corner construction.  A
    # float64 polygon kernel cannot recover a thin edge that was already lost
    # while forming float32 corners at large pixel coordinates.
    first_corners = rbox_to_corners(
        boxes1.to(torch.float64), normalized_angle=model_space)
    second_corners = rbox_to_corners(
        boxes2.to(torch.float64), normalized_angle=model_space)
    values = []
    if aligned:
        for start in range(0, len(boxes1), _ROTATED_IOU_PAIR_CHUNK):
            end = min(start + _ROTATED_IOU_PAIR_CHUNK, len(boxes1))
            values.append(_aligned_quadrilateral_iou(
                first_corners[start:end], second_corners[start:end]))
    else:
        pair_count = len(boxes1) * len(boxes2)
        for start in range(0, pair_count, _ROTATED_IOU_PAIR_CHUNK):
            end = min(start + _ROTATED_IOU_PAIR_CHUNK, pair_count)
            flat_index = torch.arange(start, end, device=boxes1.device)
            first_index = torch.div(
                flat_index, len(boxes2), rounding_mode="floor")
            second_index = torch.remainder(flat_index, len(boxes2))
            values.append(_aligned_quadrilateral_iou(
                first_corners[first_index], second_corners[second_index]))
    return torch.cat(values).reshape(output_shape).to(dtype=boxes1.dtype)


def class_aware_rotated_nms(boxes: Tensor, scores: Tensor, labels: Tensor,
                            iou_threshold: float, max_output: int | None = None) -> Tensor:
    """Return score-sorted indices after robust per-class rotated NMS."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Rotated NMS IoU threshold must be in [0, 1]")
    keeps = []
    for label in labels.unique(sorted=True):
        class_idx = torch.nonzero(labels == label, as_tuple=False).squeeze(1)
        if class_idx.numel() == 0:
            continue
        order = torch.argsort(scores[class_idx], descending=True, stable=True)
        class_boxes = boxes[class_idx]
        corners = rbox_to_corners(
            class_boxes.to(torch.float64), normalized_angle=False)
        bounds_min = corners.amin(dim=-2)
        bounds_max = corners.amax(dim=-2)
        local_keep = []
        while len(order):
            current = order[0]
            local_keep.append(current)
            rest = order[1:]
            if not len(rest):
                break
            # Axis-aligned support boxes are an exact broad phase: rotated
            # rectangles with disjoint supports cannot overlap.
            support_overlap = (
                (bounds_min[rest, 0] <= bounds_max[current, 0]) &
                (bounds_max[rest, 0] >= bounds_min[current, 0]) &
                (bounds_min[rest, 1] <= bounds_max[current, 1]) &
                (bounds_max[rest, 1] >= bounds_min[current, 1])
            )
            suppress = torch.zeros_like(support_overlap)
            candidates = torch.nonzero(
                support_overlap, as_tuple=False).squeeze(1)
            if len(candidates):
                overlaps = rotated_iou(
                    class_boxes[current].unsqueeze(0),
                    class_boxes[rest[candidates]],
                    model_space=False,
                )[0]
                suppress[candidates] = overlaps > iou_threshold
            order = rest[~suppress]
        keeps.append(class_idx[torch.stack(local_keep)])
    if not keeps:
        return labels.new_empty((0,), dtype=torch.long)
    keep = torch.cat(keeps)
    keep = keep[torch.argsort(scores[keep], descending=True, stable=True)]
    return keep[:max_output] if max_output is not None else keep
