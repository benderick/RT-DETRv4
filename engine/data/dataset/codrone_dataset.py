"""CODrone/DOTA-style oriented object-detection dataset."""

from __future__ import annotations

import json
import re
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
        self.image_dir_name = str(image_dir)
        self.ann_dir_name = str(ann_dir)
        self.filter_empty_gt = bool(filter_empty_gt)
        self.image_folder = self.root / image_dir
        self.ann_folder = self.root / ann_dir
        self.metadata_folder = self.root / "metadata"
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
        self._tiling_metadata_cache: Dict[int, Dict[str, object]] = {}

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
        source_object_indices: List[int] = []
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
                    source_object_index = len(polygons) + len(ignored_polygons)
                    # CODrone contains a handful of explicit ignore-region
                    # quadrilaterals. They are not one of the 12 categories
                    # and must not become negative training examples.
                    if class_name.lower() in {"ignored", "ignore"}:
                        if not self._valid_polygon(polygon):
                            continue
                        ignored_polygons.append(polygon)
                        continue
                    if class_name not in self.class_to_label:
                        raise ValueError(
                            f"Unknown CODrone class {class_name!r} in {annotation_path}:{line_number}")
                    if not self._valid_polygon(polygon):
                        continue
                    difficulty = int(fields[9]) if len(fields) > 9 else 0
                    polygons.append(polygon)
                    labels.append(self.class_to_label[class_name])
                    difficulties.append(difficulty)
                    source_object_indices.append(source_object_index)

        corners = torch.tensor(polygons, dtype=torch.float32).reshape(-1, 4, 2)
        boxes = corners_to_rboxes(corners)
        ignored_corners = torch.tensor(ignored_polygons, dtype=torch.float32).reshape(-1, 4, 2)
        parsed = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "difficulty": torch.tensor(difficulties, dtype=torch.int64),
            "source_object_index": torch.tensor(source_object_indices, dtype=torch.int64),
            "ignore_boxes": corners_to_rboxes(ignored_corners),
        }
        self._annotation_cache[index] = {key: value.clone() for key, value in parsed.items()}
        return parsed

    @staticmethod
    def _valid_polygon(polygon):
        """Return whether a DOTA quadrilateral has finite, positive area."""
        values = torch.as_tensor(polygon, dtype=torch.float64).reshape(4, 2)
        if not torch.isfinite(values).all():
            return False
        x, y = values[:, 0], values[:, 1]
        area = (x @ torch.roll(y, -1) - y @ torch.roll(x, -1)).abs() * 0.5
        return bool(area > torch.finfo(torch.float64).eps)

    def _parse_tiling_metadata(self, index: int, instance_count: int) -> Dict[str, object]:
        if index in self._tiling_metadata_cache:
            return {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in self._tiling_metadata_cache[index].items()
            }
        metadata_path = self.metadata_folder / f"{self.images[index].stem}.json"
        if not metadata_path.is_file():
            return {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        regular_objects = [item for item in payload.get("objects", []) if not item["ignored"]]
        regular_objects.sort(key=lambda item: item["gt_index"])
        expected_indices = list(range(instance_count))
        actual_indices = [item["gt_index"] for item in regular_objects]
        if actual_indices != expected_indices:
            raise ValueError(
                f"Tile metadata/annotation mismatch in {metadata_path}: "
                f"expected GT indices {expected_indices}, got {actual_indices}")

        metadata = {
            "partition_id": payload["partition_id"],
            "tile_id": payload["tile_id"],
            "source_image_id": payload["source_image_id"],
            "source_image_size": torch.tensor(payload["source_size"], dtype=torch.int64),
            "tile_origin": torch.tensor(payload["tile_origin"], dtype=torch.int64),
            "tile_size": torch.tensor(payload["tile_size"], dtype=torch.int64),
            "tile_overlap": torch.tensor(payload["tile_overlap"], dtype=torch.int64),
            "tile_step": torch.tensor(payload["tile_step"], dtype=torch.int64),
            "source_object_index": torch.tensor(
                [item["source_object_index"] for item in regular_objects], dtype=torch.int64),
            "visible_ratio": torch.tensor(
                [item["visible_ratio"] for item in regular_objects], dtype=torch.float32),
            "source_tile_count": torch.tensor(
                [item["source_tile_count"] for item in regular_objects], dtype=torch.int64),
            "center_boundary_distance_px": torch.tensor(
                [item["center_boundary_distance_px"] for item in regular_objects],
                dtype=torch.float32),
            "boundary_distance_px": torch.tensor(
                [item["boundary_distance_px"] for item in regular_objects], dtype=torch.float32),
            "boundary_distance_object_scale": torch.tensor(
                [item["boundary_distance_object_scale"] for item in regular_objects],
                dtype=torch.float32),
        }
        self._tiling_metadata_cache[index] = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in metadata.items()
        }
        return metadata

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
            "source_object_index": parsed["source_object_index"],
            "area": boxes[:, 2] * boxes[:, 3],
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "idx": torch.tensor([index], dtype=torch.int64),
            "orig_size": torch.tensor([width, height], dtype=torch.int64),
            "size": torch.tensor([width, height], dtype=torch.int64),
            "scale_factor": torch.ones(2, dtype=torch.float32),
            "padding": torch.zeros(4, dtype=torch.int64),
        }
        target.update(self._parse_tiling_metadata(index, len(boxes)))
        return image, target

    def get_partition_metadata(self, index: int) -> Dict[str, object]:
        """Return tile/source identity without decoding the image.

        The merged original-image evaluator calls this for every tile, so the
        metadata path must stay independent of PIL image decoding.
        """

        parsed = self._parse_annotation(index)
        metadata = self._parse_tiling_metadata(index, len(parsed["boxes"]))
        if not metadata:
            return {
                "partition_id": "full_image",
                "tile_id": None,
                "source_image_id": self.image_ids[index],
                "source_image_size": None,
                "tile_origin": torch.zeros(2, dtype=torch.int64),
                "tile_size": None,
                "tile_overlap": None,
                "tile_step": None,
            }
        return metadata

    def get_ground_truth(self, image_id: int):
        """Return untransformed pixel-space GT used by the evaluator."""
        index = int(image_id)
        parsed = self._parse_annotation(index)
        return {
            **parsed,
            "image_name": self.images[index].stem,
            "image_path": str(self.images[index]),
        }

    def build_source_dataset(self, root):
        """Build the untiled counterpart required by merged DOTA evaluation.

        This factory is deliberately owned by the dataset adapter.  The
        generic evaluator must never import or guess a concrete dataset
        class, directory layout, or category vocabulary.
        """

        return type(self)(
            root,
            image_dir=self.image_dir_name,
            ann_dir=self.ann_dir_name,
            classes=self.classes,
            filter_empty_gt=False,
        )

    def get_image_metadata(self, image_id: int):
        """Decode CODrone acquisition factors from its stable image name."""

        name = self.image_ids[int(image_id)].lower()
        illumination = re.search(r"(?:^|_)(day|night)(?:_|$)", name)
        altitude = re.search(r"_(\d+(?:\.\d+)?)m(?:_|$)", name)
        view = re.search(r"_(\d+(?:\.\d+)?)c(?:_|$)", name)
        frame = re.search(r"_frame_(\d+)(?:_|$)", name)
        return {
            "illumination": illumination.group(1) if illumination else None,
            "altitude_m": float(altitude.group(1)) if altitude else None,
            "view_angle_deg": float(view.group(1)) if view else None,
            "frame_index": int(frame.group(1)) if frame else None,
        }

    def get_dataset_provenance(self):
        """Return adapter identity and optional tiling-manifest provenance."""

        result = {
            "adapter": type(self).__name__,
            "root": str(self.root),
            "image_count": len(self),
            "classes": self.classes,
        }
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            result.update({
                "partition_id": manifest.get("partition_id"),
                "source_split": manifest.get("source_split"),
                "source_inventory_sha256": manifest.get("source_inventory_sha256"),
                "protocol": manifest.get("protocol"),
            })
        return result

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
