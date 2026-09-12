"""Small dependency-free OBB visualization helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .rotated_box_ops import rbox_to_corners


_COLORS = (
    "#ff4d4f", "#40a9ff", "#73d13d", "#ffc53d", "#9254de", "#36cfc9",
    "#ff7a45", "#597ef7", "#bae637", "#f759ab", "#13c2c2", "#faad14",
)


def to_preview_image(image):
    """Render non-RGB tensors as explicitly labelled band-0 previews."""
    if isinstance(image, (str, Path)) and Path(image).suffix.lower() == ".npy":
        from ..data.dataset.moda_dataset import load_moda_image
        image = load_moda_image(image)
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError("Preview tensor must be CHW")
        if tensor.dtype != torch.uint8:
            tensor = (tensor.clamp(0, 1) * 255).round().to(torch.uint8)
        if tensor.shape[0] == 3:
            return Image.fromarray(tensor.permute(1, 2, 0).numpy())
        preview = Image.fromarray(tensor[0].numpy()).convert("RGB")
        ImageDraw.Draw(preview).text((4, 4), "Band 0 (grayscale preview)", fill="yellow")
        return preview
    if isinstance(image, Image.Image):
        return image.convert("RGB").copy()
    return Image.open(image).convert("RGB")


def draw_obbs(image, boxes, labels, scores=None, class_names=None,
              score_threshold=0.0, line_width=3):
    """Draw pixel-space ``cx,cy,w,h,theta(rad)`` boxes on a PIL image."""
    image = to_preview_image(image)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).cpu().reshape(-1, 5)
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().reshape(-1)
    show_scores = scores is not None
    if scores is None:
        scores = torch.ones(len(boxes))
    scores = torch.as_tensor(scores, dtype=torch.float32).cpu().reshape(-1)
    corners = rbox_to_corners(boxes).round().to(torch.int64)
    drawing = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for polygon, label, score in zip(corners, labels, scores):
        if float(score) < score_threshold:
            continue
        color = _COLORS[int(label) % len(_COLORS)]
        points = [tuple(point.tolist()) for point in polygon]
        drawing.line(points + [points[0]], fill=color, width=line_width, joint="curve")
        class_name = class_names[int(label)] if class_names is not None else str(int(label))
        text = f"{class_name} {float(score):.2f}" if show_scores else class_name
        x, y = points[0]
        text_box = drawing.textbbox((x, y), text, font=font)
        drawing.rectangle(text_box, fill=color)
        drawing.text((x, y), text, fill="black", font=font)
    return image


def save_obb_visualization(path, image, boxes, labels, scores=None, class_names=None,
                           score_threshold=0.0, line_width=3):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = draw_obbs(image, boxes, labels, scores, class_names,
                         score_threshold, line_width)
    rendered.save(output)
    return output
