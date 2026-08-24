"""Geometry primitives for O^2-DFINE Angle Distribution Refinement.

The paper represents an oriented rectangle by the four distances from its
centre to the edges of its external axis-aligned rectangle and two gliding
vertex offsets.  This module keeps that representation isolated from the
decoder so its encode/decode contract can be tested independently.

Model-space boxes are ``(cx, cy, w, h, theta / pi)``.  The implementation
assumes an isotropic model canvas (current OBB recipes pad to a square)
because an angle is not preserved by anisotropic coordinate normalisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ....rotated_box_ops import corners_to_rboxes


ADR_COMPONENT_NAMES = (
    "external_left",
    "external_top",
    "external_right",
    "external_bottom",
    "vertex_epsilon",
    "vertex_eta",
)


def o2_weighting_function(
    reg_max: int,
    a: float = 0.5,
    c: float = 0.25,
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> Tensor:
    """Return the paper's non-uniform :math:`A(n)` for ``N + 1`` bins.

    ``reg_max`` is the paper's ``N`` and must be even so that the central bin
    has exactly zero residual.  The published O^2-DFINE setting is
    ``N=32, a=1/2, c=1/4``.
    """

    if reg_max < 4 or reg_max % 2:
        raise ValueError("O^2 ADR requires an even reg_max >= 4")
    if a <= 0 or c <= 0:
        raise ValueError("O^2 ADR weighting parameters a and c must be positive")
    indices = torch.arange(reg_max + 1, dtype=dtype, device=device)
    centered = indices - reg_max / 2
    exponent = 2 * centered.abs() / (reg_max - 2)
    values = torch.sign(centered) * c * (
        torch.pow(torch.as_tensor(1 + a / c, dtype=dtype, device=device), exponent) - 1
    )
    endpoints = centered.abs() == reg_max / 2
    values = torch.where(endpoints, torch.sign(centered) * (2 * a), values)
    values[reg_max // 2] = 0
    return values


def distribution_integral(logits: Tensor, project: Tensor, components: int = 6) -> Tensor:
    """Decode residual logits by expectation over the ADR codebook."""

    if logits.shape[-1] != components * project.numel():
        raise ValueError(
            f"Expected {components} x {project.numel()} ADR logits, got {logits.shape[-1]}"
        )
    working_project = project.to(dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(
        logits.reshape(
            *logits.shape[:-1], components, working_project.numel()
        ),
        dim=-1,
    )
    return torch.sum(probabilities * working_project, dim=-1)


def _side_vertices(corners: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select side vertices using exact lexicographic tie breaks.

    ``torch.isclose`` must not be used here.  Its relative tolerance depends
    on the absolute image coordinate, so a near-axis-aligned box can change
    vertex merely after translation.  Axis round-off is handled consistently
    when constructing centre-relative corners, so only a true equality is a
    tie here.  The secondary coordinate gives its deterministic convention.
    """

    x, y = corners[..., 0], corners[..., 1]
    inf = torch.full_like(x, torch.inf)
    min_y = y.min(dim=-1, keepdim=True).values
    max_y = y.max(dim=-1, keepdim=True).values
    min_x = x.min(dim=-1, keepdim=True).values
    max_x = x.max(dim=-1, keepdim=True).values

    # Axis-aligned rectangles contain two vertices on every extreme side.
    # Choosing top-left, top-right, bottom-right and bottom-left respectively
    # makes their gliding offsets reconstruct the original rectangle exactly.
    top_index = torch.where(y == min_y, x, inf).argmin(dim=-1)
    right_index = torch.where(x == max_x, y, inf).argmin(dim=-1)
    bottom_index = torch.where(y == max_y, -x, inf).argmin(dim=-1)
    left_index = torch.where(x == min_x, -y, inf).argmin(dim=-1)

    def gather(index: Tensor) -> Tensor:
        return corners.gather(
            -2, index[..., None, None].expand(*index.shape, 1, 2)
        ).squeeze(-2)

    return gather(top_index), gather(right_index), gather(bottom_index), gather(left_index)


def _centered_rbox_corners(boxes: Tensor, normalized_angle: bool) -> Tensor:
    """Construct relative corners with one coherent machine-axis snap.

    A floating representation of ``pi/2`` has a tiny non-zero cosine.  Letting
    four independent extrema interpret that residue can select a top vertex
    from one side of the axis and a right vertex from the other.  Snapping the
    trigonometric pair once, within eight machine epsilons, makes the choice
    coherent.  Since no absolute centre is involved, it is translation
    invariant.  Genuine rotations outside machine precision are untouched.
    """

    angle = boxes[..., 4] * math.pi if normalized_angle else boxes[..., 4]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    tolerance = 8 * torch.finfo(boxes.dtype).eps
    cos_a = torch.where(cos_a.abs() <= tolerance, torch.zeros_like(cos_a), cos_a)
    sin_a = torch.where(sin_a.abs() <= tolerance, torch.zeros_like(sin_a), sin_a)
    cos_a = torch.where(
        (cos_a.abs() - 1).abs() <= tolerance, torch.sign(cos_a), cos_a
    )
    sin_a = torch.where(
        (sin_a.abs() - 1).abs() <= tolerance, torch.sign(sin_a), sin_a
    )

    template = boxes.new_tensor(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    )
    local = template * boxes[..., 2:4].unsqueeze(-2)
    rotation = torch.stack(
        (cos_a, -sin_a, sin_a, cos_a), dim=-1
    ).reshape(*angle.shape, 2, 2)
    return torch.matmul(local, rotation.transpose(-1, -2))


def _positive_span(value: Tensor) -> Tensor:
    """Project only the joint external span to the valid OBB domain.

    The two signed distances remain untouched, so their difference still
    carries the external-rectangle centre correction.  There is no
    dtype-dependent continuation and no precision promotion hidden in the
    geometry contract.
    """

    return value.clamp_min(1e-7)


def rbox_to_adr(boxes: Tensor, normalized_angle: bool = True) -> tuple[Tensor, Tensor]:
    """Encode OBBs into six ADR values and external-rectangle ``(Wr, Hr)``.

    The four edge distances are measured from the OBB centre.  ``epsilon`` is
    the horizontal distance from the top vertex to the top-right external
    corner; ``eta`` is the vertical distance from the right vertex to the
    bottom-right corner, following the definitions in O^2-DFINE.
    """

    # Work in a centre-relative frame.  Besides being cheaper for the four
    # symmetric edge distances, this avoids losing a tiny near-axis vertex
    # separation when a float32 centre is added and subtracted again.
    corners = _centered_rbox_corners(boxes, normalized_angle=normalized_angle)
    x, y = corners[..., 0], corners[..., 1]
    x1, x2 = x.min(dim=-1).values, x.max(dim=-1).values
    y1, y2 = y.min(dim=-1).values, y.max(dim=-1).values
    top, right, _, _ = _side_vertices(corners)
    edges = torch.stack(
        (-x1, -y1, x2, y2), dim=-1
    )
    offsets = torch.stack((x2 - top[..., 0], y2 - right[..., 1]), dim=-1)
    scale = torch.stack((x2 - x1, y2 - y1), dim=-1).clamp_min(1e-7)
    return torch.cat((edges, offsets), dim=-1), scale


def adr_values_to_rbox(
    reference_centers: Tensor,
    values: Tensor,
    normalized_angle: bool = True,
) -> Tensor:
    """Decode absolute ADR chart values around a reference centre.

    O² predicts residuals relative to an initial box, but the geometric
    decoder only needs the resulting six chart values and the centre about
    which the four external-edge distances are measured.  Exposing that
    operation keeps representation-only diagnostics honest: they can perturb
    ADR itself without leaking the clean reference width, height, or angle
    into the decoder.
    """

    if reference_centers.shape[:-1] != values.shape[:-1]:
        raise ValueError("ADR centres and values must have matching leading shapes")
    if reference_centers.shape[-1] != 2 or values.shape[-1] != 6:
        raise ValueError("ADR value decode expects (..., 2) centres and (..., 6) values")

    left, top_distance, right, bottom_distance = values[..., :4].unbind(-1)
    width = _positive_span(left + right)
    height = _positive_span(top_distance + bottom_distance)

    # Difference controls the new external-rectangle centre even when one
    # signed side distance crosses zero.  Repairing the joint span rather
    # than individual sides preserves this correction.
    center = torch.stack(
        (reference_centers[..., 0] + 0.5 * (right - left),
         reference_centers[..., 1] + 0.5 * (bottom_distance - top_distance)),
        dim=-1,
    )
    half_size = 0.5 * torch.stack((width, height), dim=-1)
    x1y1 = center - half_size
    x2y2 = center + half_size
    epsilon, eta = values[..., 4:].unbind(-1)

    raw_corners = torch.stack(
        (
            torch.stack((x2y2[..., 0] - epsilon, x1y1[..., 1]), dim=-1),
            torch.stack((x2y2[..., 0], x2y2[..., 1] - eta), dim=-1),
            torch.stack((x1y1[..., 0] + epsilon, x2y2[..., 1]), dim=-1),
            torch.stack((x1y1[..., 0], x1y1[..., 1] + eta), dim=-1),
        ),
        dim=-2,
    )
    # The paper defines these as the ordered top/right/bottom/left vertices.
    # For a valid ADR state they are already an exact rectangle.  During
    # learning the six independent expectations can be inconsistent; the
    # minimal geometric completion is to use the two consecutive edge
    # lengths and the longer edge direction, exactly as the framework's
    # ordered-quadrilateral OBB conversion does.  In particular, do not
    # import midpoint-offset radial projection here: that coder solves a
    # different representation and can collapse two adjacent ADR vertices
    # onto an almost collinear state.
    boxes = corners_to_rboxes(raw_corners)
    if normalized_angle:
        boxes = boxes.clone()
        boxes[..., 4] /= math.pi
    return boxes


def adr_to_rbox(
    reference_boxes: Tensor,
    residuals: Tensor,
    normalized_angle: bool = True,
) -> Tensor:
    """Apply six ADR residuals and decode the ordered gliding vertices.

    The paper does not specify how an inconsistent intermediate six-tuple is
    converted to five OBB parameters.  We use the framework's direct ordered
    quadrilateral conversion; no unrelated coder or radial projection is
    inserted into the paper-derived path.
    """

    if reference_boxes.shape[:-1] != residuals.shape[:-1] or residuals.shape[-1] != 6:
        raise ValueError("ADR decode expects matching (..., 5) boxes and (..., 6) residuals")
    base, external_scale = rbox_to_adr(reference_boxes, normalized_angle=normalized_angle)
    scale6 = torch.cat((external_scale, external_scale, external_scale), dim=-1)
    values = base + scale6 * residuals
    return adr_values_to_rbox(
        reference_boxes[..., :2], values, normalized_angle=normalized_angle)


def adr_target_residual(
    reference_boxes: Tensor,
    target_boxes: Tensor,
    normalized_angle: bool = True,
) -> Tensor:
    """Encode target ADR corrections relative to the initial reference OBB."""

    if reference_boxes.shape != target_boxes.shape or reference_boxes.shape[-1] != 5:
        raise ValueError("ADR targets require equal (..., 5) reference and target boxes")
    reference_values, external_scale = rbox_to_adr(
        reference_boxes, normalized_angle=normalized_angle
    )
    target_values, _ = rbox_to_adr(target_boxes, normalized_angle=normalized_angle)

    # Edge distances are relative to the *initial* centre, as in D-FINE's
    # point-to-boundary target, so they also encode centre displacement.
    # Convert the target's centre-relative distances to distances from the
    # initial reference centre.  This avoids absolute-coordinate cancellation
    # for float32 boxes close to an axis.
    target_edges_from_center = target_values[..., :4]
    center_delta = target_boxes[..., :2] - reference_boxes[..., :2]
    target_edges = torch.stack(
        (target_edges_from_center[..., 0] - center_delta[..., 0],
         target_edges_from_center[..., 1] - center_delta[..., 1],
         target_edges_from_center[..., 2] + center_delta[..., 0],
         target_edges_from_center[..., 3] + center_delta[..., 1]), dim=-1
    )
    target_values = torch.cat((target_edges, target_values[..., 4:]), dim=-1)
    scale6 = torch.cat((external_scale, external_scale, external_scale), dim=-1)
    return (target_values - reference_values) / scale6.clamp_min(1e-7)


def translate_with_project(target: Tensor, project: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Map continuous residual targets to adjacent ADR bins without bias."""

    target = target.reshape(-1)
    insertion = torch.searchsorted(project, target, right=True) - 1
    left = insertion.clamp(0, project.numel() - 2)
    right = left + 1
    left_value, right_value = project[left], project[right]
    denominator = (right_value - left_value).clamp_min(1e-12)
    weight_right = ((target - left_value) / denominator).clamp(0, 1)
    weight_left = 1 - weight_right
    below = target <= project[0]
    above = target >= project[-1]
    left = torch.where(below, torch.zeros_like(left), left)
    left = torch.where(above, torch.full_like(left, project.numel() - 2), left)
    weight_left = torch.where(below, torch.ones_like(weight_left), weight_left)
    weight_right = torch.where(below, torch.zeros_like(weight_right), weight_right)
    weight_left = torch.where(above, torch.zeros_like(weight_left), weight_left)
    weight_right = torch.where(above, torch.ones_like(weight_right), weight_right)
    return left, weight_right, weight_left
