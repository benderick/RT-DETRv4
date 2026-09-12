"""Reusable dataset adapter for DOTA-style oriented-box annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from PIL import Image

from ._dataset import DetDataset
from ...rtv4.rotated_box_ops import corners_to_rboxes


class DotaOBBDetection(DetDataset):
    """Read images paired with one DOTA quadrilateral text file per image.

    Dataset-specific subclasses own their vocabulary and filename metadata;
    this base class owns only the common DOTA parsing and target contract.
    Images without an annotation file are retained as empty images unless
    ``filter_empty_gt`` is requested explicitly.
    """

    __inject__ = ["transforms"]
    image_suffixes = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

    def __init__(
        self,
        root: str,
        transforms=None,
        *,
        image_dir: str = "images",
        ann_dir: str = "annfile",
        classes: Sequence[str],
        class_aliases: Mapping[str, str] | None = None,
        filter_empty_gt: bool = False,
        dataset_name: str = "DOTA-style OBB",
    ):
        self.root = Path(root).expanduser().resolve()
        self.image_dir_name = str(image_dir)
        self.ann_dir_name = str(ann_dir)
        self.filter_empty_gt = bool(filter_empty_gt)
        self.dataset_name = str(dataset_name)
        self.image_folder = self.root / self.image_dir_name
        self.ann_folder = self.root / self.ann_dir_name
        self.metadata_folder = self.root / "metadata"
        if not self.image_folder.is_dir():
            raise FileNotFoundError(
                f"{self.dataset_name} image directory does not exist: {self.image_folder}")
        if not self.ann_folder.is_dir():
            raise FileNotFoundError(
                f"{self.dataset_name} annotation directory does not exist: {self.ann_folder}")

        self.transforms = transforms
        self.classes = tuple(str(name) for name in classes)
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("DOTA classes must be non-empty and unique")
        self.class_to_label = {name: index for index, name in enumerate(self.classes)}
        self.class_aliases = dict(class_aliases or {})
        images = self._find_images()
        if filter_empty_gt:
            images = [
                path for path in images
                if (self.ann_folder / f"{path.stem}.txt").is_file()
            ]
        if not images:
            raise RuntimeError(f"No images found in {self.image_folder}")

        self.images = images
        self.ids = list(range(len(images)))
        self.image_ids = [path.stem for path in images]
        self._annotation_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._tiling_metadata_cache: Dict[int, Dict[str, object]] = {}
        self._inventory_sha256: str | None = None

    def __len__(self):
        return len(self.images)

    def _find_images(self):
        return sorted(path for path in self.image_folder.iterdir()
                      if path.is_file() and path.suffix.lower() in self.image_suffixes)

    def _load_image(self, index):
        return Image.open(self.images[index]).convert("RGB")

    def __getitem__(self, index):
        image, target = self.load_item(index)
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)
        return image, target

    @staticmethod
    def _valid_polygon(polygon) -> bool:
        """Return whether a DOTA quadrilateral is finite with positive area."""

        values = torch.as_tensor(polygon, dtype=torch.float64).reshape(4, 2)
        if not torch.isfinite(values).all():
            return False
        x, y = values[:, 0], values[:, 1]
        area = (x @ torch.roll(y, -1) - y @ torch.roll(x, -1)).abs() * 0.5
        return bool(area > torch.finfo(torch.float64).eps)

    def _parse_annotation(self, index: int) -> Dict[str, torch.Tensor]:
        if index in self._annotation_cache:
            return {
                key: value.clone()
                for key, value in self._annotation_cache[index].items()
            }

        annotation_path = self.ann_folder / f"{self.images[index].stem}.txt"
        polygons: List[List[float]] = []
        ignored_polygons: List[List[float]] = []
        labels: List[int] = []
        difficulties: List[int] = []
        source_object_indices: List[int] = []
        object_index = 0
        if annotation_path.is_file():
            with annotation_path.open("r", encoding="utf-8-sig") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    fields = raw_line.strip().split()
                    if not fields:
                        continue
                    current_object_index = object_index
                    object_index += 1
                    if len(fields) < 9:
                        raise ValueError(
                            f"Malformed DOTA annotation {annotation_path}:{line_number}: "
                            f"{raw_line.rstrip()}")
                    try:
                        polygon = [float(value) for value in fields[:8]]
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid coordinates in {annotation_path}:{line_number}") from exc
                    class_name = self.class_aliases.get(fields[8], fields[8])
                    if class_name.lower() in {"ignored", "ignore"}:
                        if self._valid_polygon(polygon):
                            ignored_polygons.append(polygon)
                        continue
                    if class_name not in self.class_to_label:
                        raise ValueError(
                            f"Unknown {self.dataset_name} class {class_name!r} in "
                            f"{annotation_path}:{line_number}")
                    if not self._valid_polygon(polygon):
                        continue
                    try:
                        difficulty = int(fields[9]) if len(fields) > 9 else 0
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid difficulty in {annotation_path}:{line_number}") from exc
                    if difficulty < 0:
                        raise ValueError(
                            f"Negative difficulty in {annotation_path}:{line_number}")
                    polygons.append(polygon)
                    labels.append(self.class_to_label[class_name])
                    difficulties.append(difficulty)
                    source_object_indices.append(current_object_index)

        corners = torch.tensor(polygons, dtype=torch.float32).reshape(-1, 4, 2)
        ignored_corners = torch.tensor(
            ignored_polygons, dtype=torch.float32).reshape(-1, 4, 2)
        parsed = {
            "boxes": corners_to_rboxes(corners),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "difficulty": torch.tensor(difficulties, dtype=torch.int64),
            "source_object_index": torch.tensor(
                source_object_indices, dtype=torch.int64),
            "ignore_boxes": corners_to_rboxes(ignored_corners),
        }
        self._annotation_cache[index] = {
            key: value.clone() for key, value in parsed.items()
        }
        return parsed

    def _parse_tiling_metadata(
        self, index: int, instance_count: int
    ) -> Dict[str, object]:
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
        regular_objects = [
            item for item in payload.get("objects", []) if not item["ignored"]
        ]
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
            "source_image_size": torch.tensor(
                payload["source_size"], dtype=torch.int64),
            "tile_origin": torch.tensor(payload["tile_origin"], dtype=torch.int64),
            "tile_size": torch.tensor(payload["tile_size"], dtype=torch.int64),
            "tile_overlap": torch.tensor(
                payload["tile_overlap"], dtype=torch.int64),
            "tile_step": torch.tensor(payload["tile_step"], dtype=torch.int64),
            "source_object_index": torch.tensor(
                [item["source_object_index"] for item in regular_objects],
                dtype=torch.int64),
            "visible_ratio": torch.tensor(
                [item["visible_ratio"] for item in regular_objects],
                dtype=torch.float32),
            "source_tile_count": torch.tensor(
                [item["source_tile_count"] for item in regular_objects],
                dtype=torch.int64),
            "center_boundary_distance_px": torch.tensor(
                [item["center_boundary_distance_px"] for item in regular_objects],
                dtype=torch.float32),
            "boundary_distance_px": torch.tensor(
                [item["boundary_distance_px"] for item in regular_objects],
                dtype=torch.float32),
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
        image = self._load_image(index)
        width, height = (image.shape[-1], image.shape[-2]) if torch.is_tensor(image) else image.size
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
        index = int(image_id)
        parsed = self._parse_annotation(index)
        return {
            **parsed,
            "image_name": self.images[index].stem,
            "image_path": str(self.images[index]),
        }

    def build_source_dataset(self, root):
        return type(self)(
            root,
            image_dir=self.image_dir_name,
            ann_dir=self.ann_dir_name,
            classes=self.classes,
            filter_empty_gt=False,
        )

    def get_image_metadata(self, image_id: int):
        return {}

    def _compute_inventory_sha256(self) -> str:
        if self._inventory_sha256 is not None:
            return self._inventory_sha256
        digest = hashlib.sha256()
        for image in self.images:
            annotation = self.ann_folder / f"{image.stem}.txt"
            for path in (image, annotation):
                relative = path.relative_to(self.root).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                if path.is_file():
                    stat = path.stat()
                    digest.update(str(stat.st_size).encode("ascii"))
                    digest.update(b"\0")
                    if path == annotation:
                        digest.update(path.read_bytes())
                digest.update(b"\n")
        self._inventory_sha256 = digest.hexdigest()
        return self._inventory_sha256

    def get_dataset_provenance(self):
        result = {
            "adapter": type(self).__name__,
            "dataset_name": self.dataset_name,
            "root": str(self.root),
            "image_count": len(self),
            "classes": self.classes,
            "inventory_sha256": self._compute_inventory_sha256(),
        }
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            result.update({
                "partition_id": manifest.get("partition_id"),
                "source_split": manifest.get("source_split"),
                "source_inventory_sha256": manifest.get(
                    "source_inventory_sha256"),
                "protocol": manifest.get("protocol"),
            })
        conversion_path = self.root / "conversion_manifest.json"
        if conversion_path.is_file():
            with conversion_path.open("r", encoding="utf-8") as handle:
                conversion = json.load(handle)
            result["conversion"] = {
                "schema_version": conversion.get("schema_version"),
                "source_format": conversion.get("source_format"),
                "target_format": conversion.get("target_format"),
                "source_inventory_sha256": conversion.get(
                    "source_inventory_sha256"),
                "output_inventory_sha256": conversion.get(
                    "output_inventory_sha256"),
            }
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
        return (
            f" root: {self.root}\n classes: {self.classes}\n "
            f"transforms: {self.transforms}"
        )
