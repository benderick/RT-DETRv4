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
from torchvision.transforms import InterpolationMode

from ...core import register
from ...rtv4.rotated_box_ops import ANGLE_PERIOD, regularize_rboxes, rbox_to_corners


def _unpack(sample):
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise TypeError("Rotated transforms expect (image, target, dataset)")
    return sample[0], sample[1], sample[2]


def image_size(image):
    """Width, height for PIL or arbitrary-channel CHW tensors."""
    if torch.is_tensor(image):
        if image.ndim != 3:
            raise ValueError("Rotated image tensors must have shape [C,H,W]")
        return int(image.shape[-1]), int(image.shape[-2])
    return image.size


def _filter_instances(target, keep):
    count = int(keep.numel())
    image_metadata = {
        "image_id", "idx", "orig_size", "size", "scale_factor", "padding",
        "source_image_size", "tile_origin", "tile_size", "tile_overlap", "tile_step",
        "effective_image_ratio", "source_padding_ltrb",
        "valid_mask",
    }
    for key, value in list(target.items()):
        if (
            torch.is_tensor(value) and value.ndim > 0
            and value.shape[0] == count and key not in image_metadata
        ):
            target[key] = value[keep]
    return target


@register()
class RotatedResize(nn.Module):
    """Keep aspect ratio and resize without changing the image canvas early.

    Rotation augmentation must see the resized image itself, not an already
    padded square.  Otherwise a non-square image is rotated around the wrong
    centre and its valid-content boundary no longer matches the source O²
    training pipeline.
    """

    def __init__(self, size=(1024, 1024), interpolation="bilinear"):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2:
            raise ValueError("size must be (width, height)")
        self.interpolation = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
        }[interpolation]
        self.tensor_interpolation = InterpolationMode(interpolation)

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        old_width, old_height = image_size(image)
        canvas_width, canvas_height = self.size
        scale = min(canvas_width / old_width, canvas_height / old_height)
        new_width = max(1, min(canvas_width, round(old_width * scale)))
        new_height = max(1, min(canvas_height, round(old_height * scale)))
        scale_x, scale_y = new_width / old_width, new_height / old_height
        if torch.is_tensor(image):
            image = TF.resize(image, [new_height, new_width], self.tensor_interpolation, antialias=True)
        else:
            image = image.resize((new_width, new_height), self.interpolation)
        if "valid_mask" in target:
            target["valid_mask"] = TF.resize(target["valid_mask"][None].to(torch.uint8),
                [new_height, new_width], InterpolationMode.NEAREST)[0].bool()

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
        target["size"] = torch.tensor([new_width, new_height], dtype=torch.int64)
        target["scale_factor"] = torch.tensor([scale_x, scale_y], dtype=torch.float32)
        target["padding"] = torch.zeros(4, dtype=torch.int64)
        return image, target, dataset


@register()
class RotatedPad(nn.Module):
    """Pad the right/bottom after all geometry-changing augmentation."""

    def __init__(self, size=(1024, 1024), fill=114):
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if len(self.size) != 2:
            raise ValueError("size must be (width, height)")
        self.fill = int(fill)

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        old_width, old_height = image_size(image)
        canvas_width, canvas_height = self.size
        if old_width > canvas_width or old_height > canvas_height:
            raise ValueError(
                "RotatedPad cannot crop an oversized image: "
                f"image={(old_width, old_height)}, canvas={self.size}")
        pad = [0, 0, canvas_width - old_width, canvas_height - old_height]
        if torch.is_tensor(image):
            padded = TF.pad(image, pad, fill=self.fill)
        else:
            padded = Image.new("RGB", self.size, color=(self.fill,) * 3)
            padded.paste(image, (0, 0))
        if "valid_mask" in target:
            target["valid_mask"] = TF.pad(target["valid_mask"], pad, fill=0)
        target["size"] = torch.tensor(
            [canvas_width, canvas_height], dtype=torch.int64)
        target["padding"] = torch.tensor(
            [0, 0, canvas_width - old_width, canvas_height - old_height],
            dtype=torch.int64)
        return padded, target, dataset


@register()
class RotatedResizePad(nn.Module):
    """Keep aspect ratio, then pad at the right/bottom in one transform."""

    def __init__(self, size=(1024, 1024), fill=114, interpolation="bilinear"):
        super().__init__()
        self.resize = RotatedResize(size=size, interpolation=interpolation)
        self.pad = RotatedPad(size=size, fill=fill)

    def forward(self, sample):
        return self.pad(self.resize(sample))


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
        applied = random.random() < self.p
        brightness = contrast = saturation = 1.0
        hue = 0.0
        order_code = -1
        if applied:
            brightness = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
            saturation = random.uniform(1 - self.saturation, 1 + self.saturation)
            hue = random.uniform(-self.hue, self.hue)
            operations = [
                (1, lambda im: TF.adjust_brightness(im, brightness)),
                (2, lambda im: TF.adjust_contrast(im, contrast)),
                (3, lambda im: TF.adjust_saturation(im, saturation)),
                (4, lambda im: TF.adjust_hue(im, hue)),
            ]
            random.shuffle(operations)
            order_code = sum(code * (10 ** index) for index, (code, _) in enumerate(operations))
            for _, operation in operations:
                image = operation(image)
        target["aug_photometric_applied"] = torch.tensor(applied)
        target["aug_brightness_factor"] = torch.tensor(brightness, dtype=torch.float32)
        target["aug_contrast_factor"] = torch.tensor(contrast, dtype=torch.float32)
        target["aug_saturation_factor"] = torch.tensor(saturation, dtype=torch.float32)
        target["aug_hue_factor"] = torch.tensor(hue, dtype=torch.float32)
        target["aug_photometric_order_code"] = torch.tensor(order_code, dtype=torch.int64)
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
            target["aug_flip_code"] = torch.tensor(0, dtype=torch.int64)
            return image, target, dataset
        direction = random.choice(self.directions)
        target["aug_flip_code"] = torch.tensor(
            {"horizontal": 1, "vertical": 2, "diagonal": 3}[direction], dtype=torch.int64)
        width, height = image_size(image)
        boxes = target["boxes"].clone()
        if direction in {"horizontal", "diagonal"}:
            image = TF.hflip(image)
            if "valid_mask" in target:
                target["valid_mask"] = TF.hflip(target["valid_mask"])
            boxes[:, 0] = width - boxes[:, 0]
        if direction in {"vertical", "diagonal"}:
            image = TF.vflip(image)
            if "valid_mask" in target:
                target["valid_mask"] = TF.vflip(target["valid_mask"])
            boxes[:, 1] = height - boxes[:, 1]
        if direction == "horizontal":
            boxes[:, 4] = torch.remainder(ANGLE_PERIOD - boxes[:, 4], ANGLE_PERIOD)
        elif direction == "vertical":
            boxes[:, 4] = torch.remainder(-boxes[:, 4], ANGLE_PERIOD)
        target["boxes"] = regularize_rboxes(boxes)
        return image, target, dataset


@register()
class RotatedRandomRotate(nn.Module):
    """Rotate the current image and boxes about its centre."""

    def __init__(self, p=0.5, angle_range=180.0, fill=114):
        super().__init__()
        self.p = p
        self.angle_range = float(angle_range)
        self.fill = fill

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        if random.random() >= self.p:
            target["aug_rotation_degrees"] = torch.tensor(0.0, dtype=torch.float32)
            target["aug_rotation_applied"] = torch.tensor(False)
            return image, target, dataset
        degrees = random.uniform(-self.angle_range, self.angle_range)
        target["aug_rotation_degrees"] = torch.tensor(degrees, dtype=torch.float32)
        target["aug_rotation_applied"] = torch.tensor(True)
        if torch.is_tensor(image):
            image = TF.rotate(image, degrees, InterpolationMode.BILINEAR,
                              expand=False, fill=self.fill)
        else:
            image = image.rotate(degrees, resample=Image.Resampling.BILINEAR,
                                 expand=False, fillcolor=(self.fill,) * 3)
        if "valid_mask" in target:
            target["valid_mask"] = TF.rotate(target["valid_mask"][None].to(torch.uint8),
                degrees, InterpolationMode.NEAREST, fill=0)[0].bool()
        boxes = target["boxes"].clone()
        if boxes.numel():
            width, height = image_size(image)
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
        width, height = image_size(image)
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
    def __init__(self, normalize_boxes=True, box_coordinate_mode="per_axis"):
        super().__init__()
        self.normalize_boxes = normalize_boxes
        if box_coordinate_mode not in {"per_axis", "isotropic"}:
            raise ValueError("Unknown box_coordinate_mode")
        self.box_coordinate_mode = box_coordinate_mode

    def forward(self, sample):
        image, target, dataset = _unpack(sample)
        width, height = image_size(image)
        if not torch.is_tensor(image):
            image = TF.pil_to_tensor(image)
        if image.dtype != torch.uint8:
            raise ValueError("RotatedConvertToTensor expects uint8 image data before scaling")
        image = image.float().div_(255.0)
        boxes = regularize_rboxes(target["boxes"])
        if self.normalize_boxes:
            if self.box_coordinate_mode == "isotropic":
                width = height = max(width, height)
                target["box_normalization_size"] = torch.tensor([width, height])
            factor = boxes.new_tensor([width, height, width, height, ANGLE_PERIOD])
            boxes = boxes / factor
            boxes = regularize_rboxes(boxes, normalized_angle=True)
        target["boxes"] = boxes
        return image, target, dataset
