"""CODrone/DOTA-style oriented object-detection dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from ._dataset import DetDataset
from ...core import register
from ...rtv4.rotated_box_ops import corners_to_rboxes


CODRONE_CLASSES = (
    "car", "people", "motor", "truck", "traffic-sign", "traffic-light",
    "boat", "bus", "bicycle", "tricycle", "ship", "bridge",
)

_CLASS_ALIASES = {
    "traffic_sign": "traffic-sign",
    "traffic sign": "traffic-sign",
    "traffic_light": "traffic-light",
    "traffic light": "traffic-light",
}


@register()
class CODroneDetection(DetDataset):
    """Read one CODrone split with DOTA quadrilateral annotations.

    Images without an annotation text file are deliberately retained as
    empty images.  This is required for faithful validation/test inference
    and also handles the two unlabelled images in the tiny ``*_t`` splits.
    """

    __inject__ = ["transforms"]

    def __init__(
        self,
        root: str,
        transforms=None,
        image_dir: str = "images",
        ann_dir: str = "annfile",
        classes=CODRONE_CLASSES,
        filter_empty_gt: bool = False,
    ):
        self.root = Path(root).expanduser().resolve()
        self.image_folder = self.root / image_dir
        self.ann_folder = self.root / ann_dir
        if not self.image_folder.is_dir():
            raise FileNotFoundError(f"CODrone image directory does not exist: {self.image_folder}")
        if not self.ann_folder.is_dir():
            raise FileNotFoundError(f"CODrone annotation directory does not exist: {self.ann_folder}")

        self.transforms = transforms
        self.classes = tuple(classes)
        self.class_to_label = {name: index for index, name in enumerate(self.classes)}
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in self.image_folder.iterdir() if p.suffix.lower() in suffixes)
        if filter_empty_gt:
            images = [p for p in images if (self.ann_folder / f"{p.stem}.txt").is_file()]
        if not images:
            raise RuntimeError(f"No images found in {self.image_folder}")

        self.images = images
        self.ids = list(range(len(images)))
        self.image_ids = [path.stem for path in images]
        self._annotation_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    def _parse_annotation(self, index: int) -> Dict[str, torch.Tensor]:
        if index in self._annotation_cache:
            return {key: value.clone() for key, value in self._annotation_cache[index].items()}

        annotation_path = self.ann_folder / f"{self.images[index].stem}.txt"
        polygons: List[List[float]] = []
        ignored_polygons: List[List[float]] = []
        labels: List[int] = []
        difficulties: List[int] = []
        if annotation_path.is_file():
            with annotation_path.open("r", encoding="utf-8-sig") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    fields = raw_line.strip().split()
                    if not fields:
                        continue
                    if len(fields) < 9:
                        raise ValueError(
                            f"Malformed DOTA annotation {annotation_path}:{line_number}: {raw_line.rstrip()}")
                    try:
                        polygon = [float(value) for value in fields[:8]]
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid coordinates in {annotation_path}:{line_number}") from exc
                    class_name = _CLASS_ALIASES.get(fields[8], fields[8])
                    # CODrone contains a handful of explicit ignore-region
                    # quadrilaterals. They are not one of the 12 categories
                    # and must not become negative training examples.
                    if class_name.lower() in {"ignored", "ignore"}:
                        ignored_polygons.append(polygon)
                        continue
                    if class_name not in self.class_to_label:
                        raise ValueError(
                            f"Unknown CODrone class {class_name!r} in {annotation_path}:{line_number}")
                    difficulty = int(fields[9]) if len(fields) > 9 else 0
                    polygons.append(polygon)
                    labels.append(self.class_to_label[class_name])
                    difficulties.append(difficulty)

        corners = torch.tensor(polygons, dtype=torch.float32).reshape(-1, 4, 2)
        boxes = corners_to_rboxes(corners)
        ignored_corners = torch.tensor(ignored_polygons, dtype=torch.float32).reshape(-1, 4, 2)
        parsed = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "difficulty": torch.tensor(difficulties, dtype=torch.int64),
            "ignore_boxes": corners_to_rboxes(ignored_corners),
        }
        self._annotation_cache[index] = {key: value.clone() for key, value in parsed.items()}
        return parsed

    def load_item(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        image = Image.open(self.images[index]).convert("RGB")
        width, height = image.size
        parsed = self._parse_annotation(index)
        boxes = parsed["boxes"]
        target = {
            "boxes": boxes,
            "labels": parsed["labels"],
            "difficulty": parsed["difficulty"],
            "area": boxes[:, 2] * boxes[:, 3],
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "idx": torch.tensor([index], dtype=torch.int64),
            "orig_size": torch.tensor([width, height], dtype=torch.int64),
            "size": torch.tensor([width, height], dtype=torch.int64),
            "scale_factor": torch.ones(2, dtype=torch.float32),
            "padding": torch.zeros(4, dtype=torch.int64),
        }
        return image, target

    def get_ground_truth(self, image_id: int):
        """Return untransformed pixel-space GT used by the evaluator."""
        index = int(image_id)
        parsed = self._parse_annotation(index)
        return {
            **parsed,
            "image_name": self.images[index].stem,
            "image_path": str(self.images[index]),
        }

    @property
    def categories(self):
        return [{"id": i, "name": name} for i, name in enumerate(self.classes)]

    @property
    def category2name(self):
        return dict(enumerate(self.classes))

    @property
    def category2label(self):
        return {i: i for i in range(len(self.classes))}

    @property
    def label2category(self):
        return {i: i for i in range(len(self.classes))}

    def extra_repr(self):
        return f" root: {self.root}\n classes: {self.classes}\n transforms: {self.transforms}"
