"""Geometry-safe augmentations for five-parameter oriented boxes."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

from ...core import register
from ...rtv4.rotated_box_ops import ANGLE_PERIOD, regularize_rboxes, rbox_to_corners


def _unpack(sample):
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise TypeError("Rotated transforms expect (image, target, dataset)")
    return sample[0], sample[1], sample[2]


def _filter_instances(target, keep):
    count = int(keep.numel())
    for key, value in list(target.items()):
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == count and key not in {
            "image_id", "idx", "orig_size", "size", "scale_factor", "padding"
        }:
            target[key] = value[keep]
    return target


@register()
class RotatedResizePad(nn.Module):
    """Keep aspect ratio, resize, and pad at the right/bottom."""

    def __init__(self, size=(1024, 1024), fill=114, interpolation="bilinear"):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2:
            raise ValueError("size must be (width, height)")
        self.fill = fill
        self.interpolation = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
        }[interpolation]

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        old_width, old_height = image.size
        canvas_width, canvas_height = self.size
        scale = min(canvas_width / old_width, canvas_height / old_height)
        new_width = max(1, min(canvas_width, round(old_width * scale)))
        new_height = max(1, min(canvas_height, round(old_height * scale)))
        scale_x, scale_y = new_width / old_width, new_height / old_height
        resized = image.resize((new_width, new_height), self.interpolation)
        image = Image.new("RGB", self.size, color=(self.fill,) * 3)
        image.paste(resized, (0, 0))

        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            # Keep-ratio resize should be isotropic.  Rounding makes the two
            # factors microscopically different, so use the mean for lengths.
            length_scale = 0.5 * (scale_x + scale_y)
            boxes[:, 2:4] = target["boxes"][:, 2:4] * length_scale
        target["boxes"] = boxes
        target["area"] = boxes[:, 2] * boxes[:, 3]
        target["size"] = torch.tensor([canvas_width, canvas_height], dtype=torch.int64)
        target["scale_factor"] = torch.tensor([scale_x, scale_y], dtype=torch.float32)
        target["padding"] = torch.tensor(
            [0, 0, canvas_width - new_width, canvas_height - new_height], dtype=torch.int64)
        return image, target, dataset


@register()
class RotatedPhotometricDistort(nn.Module):
    def __init__(self, p=0.8, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        if random.random() < self.p:
            operations = [
                lambda im: TF.adjust_brightness(im, random.uniform(1 - self.brightness, 1 + self.brightness)),
                lambda im: TF.adjust_contrast(im, random.uniform(1 - self.contrast, 1 + self.contrast)),
                lambda im: TF.adjust_saturation(im, random.uniform(1 - self.saturation, 1 + self.saturation)),
                lambda im: TF.adjust_hue(im, random.uniform(-self.hue, self.hue)),
            ]
            random.shuffle(operations)
            for operation in operations:
                image = operation(image)
        return image, target, dataset


@register()
class RotatedRandomFlip(nn.Module):
    def __init__(self, p=0.75, directions=("horizontal", "vertical", "diagonal")):
        super().__init__()
        self.p = p
        self.directions = tuple(directions)
        invalid = set(self.directions) - {"horizontal", "vertical", "diagonal"}
        if invalid:
            raise ValueError(f"Unsupported flip directions: {sorted(invalid)}")

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        if random.random() >= self.p:
            return image, target, dataset
        direction = random.choice(self.directions)
        width, height = image.size
        boxes = target["boxes"].clone()
        if direction in {"horizontal", "diagonal"}:
            image = TF.hflip(image)
            boxes[:, 0] = width - boxes[:, 0]
        if direction in {"vertical", "diagonal"}:
            image = TF.vflip(image)
            boxes[:, 1] = height - boxes[:, 1]
        if direction == "horizontal":
            boxes[:, 4] = torch.remainder(ANGLE_PERIOD - boxes[:, 4], ANGLE_PERIOD)
        elif direction == "vertical":
            boxes[:, 4] = torch.remainder(-boxes[:, 4], ANGLE_PERIOD)
        target["boxes"] = regularize_rboxes(boxes)
        return image, target, dataset


@register()
class RotatedRandomRotate(nn.Module):
    """Rotate the square canvas and boxes about its center."""

    def __init__(self, p=0.5, angle_range=180.0, fill=114):
        super().__init__()
        self.p = p
        self.angle_range = float(angle_range)
        self.fill = fill

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        if random.random() >= self.p:
            return image, target, dataset
        degrees = random.uniform(-self.angle_range, self.angle_range)
        image = image.rotate(degrees, resample=Image.Resampling.BILINEAR,
                             expand=False, fillcolor=(self.fill,) * 3)
        boxes = target["boxes"].clone()
        if boxes.numel():
            width, height = image.size
            radians = math.radians(degrees)
            cos_a, sin_a = math.cos(radians), math.sin(radians)
            # Clone both views before writing the transformed x coordinate.
            offset_x = (boxes[:, 0] - width / 2.0).clone()
            offset_y = (boxes[:, 1] - height / 2.0).clone()
            boxes[:, 0] = cos_a * offset_x + sin_a * offset_y + width / 2.0
            boxes[:, 1] = -sin_a * offset_x + cos_a * offset_y + height / 2.0
            boxes[:, 4] = torch.remainder(boxes[:, 4] - radians, ANGLE_PERIOD)
        target["boxes"] = regularize_rboxes(boxes)
        return image, target, dataset


@register()
class RotatedSanitizeBoxes(nn.Module):
    """Remove invalid/mostly cropped boxes without axis-aligned clipping."""

    def __init__(self, min_size=1.0, min_visible=0.2):
        super().__init__()
        self.min_size = float(min_size)
        self.min_visible = float(min_visible)

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        boxes = regularize_rboxes(target["boxes"])
        width, height = image.size
        keep = torch.isfinite(boxes).all(dim=1)
        keep &= (boxes[:, 2] >= self.min_size) & (boxes[:, 3] >= self.min_size)
        keep &= (boxes[:, 0] >= 0) & (boxes[:, 0] <= width)
        keep &= (boxes[:, 1] >= 0) & (boxes[:, 1] <= height)
        if self.min_visible > 0 and boxes.numel():
            canvas = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
            corners = rbox_to_corners(boxes).detach().cpu().numpy().astype(np.float32)
            visible = []
            for box, polygon in zip(boxes, corners):
                intersection, _ = cv2.intersectConvexConvex(polygon, canvas)
                visible.append(float(intersection) / max(float(box[2] * box[3]), 1e-7))
            keep &= torch.tensor(visible, device=keep.device) >= self.min_visible
        target["boxes"] = boxes
        target = _filter_instances(target, keep)
        target["area"] = target["boxes"][:, 2] * target["boxes"][:, 3]
        return image, target, dataset


@register()
class RotatedConvertToTensor(nn.Module):
    def __init__(self, normalize_boxes=True):
        super().__init__()
        self.normalize_boxes = normalize_boxes

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        width, height = image.size
        image = TF.pil_to_tensor(image).float().div_(255.0)
        boxes = regularize_rboxes(target["boxes"])
        if self.normalize_boxes:
            factor = boxes.new_tensor([width, height, width, height, ANGLE_PERIOD])
            boxes = boxes / factor
            boxes = regularize_rboxes(boxes, normalized_angle=True)
        target["boxes"] = boxes
        return image, target, dataset
