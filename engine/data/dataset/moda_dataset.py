"""MODA eight-band CWH arrays with explicit official annotation semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dota_obb_dataset import DotaOBBDetection
from ...core import register


# Author code IDs; paper table display order is different.
MODA_CLASSES = ("car", "van", "truck", "bus", "tricycle", "bike", "awning-bike", "pedestrian")


def load_moda_image(path):
    """Return uint8 CHW; MODA stores CWH, not CHW or HWC."""
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3 or array.shape[0] != 8 or array.dtype != np.uint8:
        raise ValueError(f"Expected MODA uint8 [8,W,H], got {array.shape}/{array.dtype}: {path}")
    if min(array.shape[1:]) <= 0:
        raise ValueError(f"Empty MODA image: {path}")
    return torch.from_numpy(array.transpose(0, 2, 1).copy())


def annotation_inventory(root):
    """Hash complete source label bytes independently of a selected partition."""
    digest = hashlib.sha256()
    for path in sorted((Path(root) / "labels").glob("*.txt")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


@register()
class MODADetection(DotaOBBDetection):
    image_suffixes = frozenset({".npy"})

    def __init__(self, root, transforms=None, split_file=None, expected_labels=None,
                 filter_empty_gt=False):
        if filter_empty_gt:
            raise ValueError("MODA official protocol retains empty images")
        self.split_file = Path(split_file).expanduser().resolve() if split_file else None
        self.expected_labels = expected_labels
        self.partition_manifest = None
        super().__init__(root, transforms, ann_dir="labels", classes=MODA_CLASSES,
                         dataset_name="MODA")

    def _find_images(self):
        labels = {p.stem for p in self.ann_folder.glob("*.txt")}
        available = {p.stem: p for p in super()._find_images()}
        if self.expected_labels is not None and len(labels) != self.expected_labels:
            raise ValueError(f"MODA expects {self.expected_labels} source labels, found {len(labels)}")
        selected = sorted(labels)
        if self.split_file:
            self.partition_manifest = json.loads(self.split_file.read_text())
            manifest = self.partition_manifest
            if manifest.get("schema_version") != "moda-split-v1":
                raise ValueError("Expected moda-split-v1 manifest")
            if manifest.get("source_split") != self.root.name:
                raise ValueError("MODA manifest source_split does not match dataset root")
            selected = manifest.get("image_ids")
            if not isinstance(selected, list) or not selected or any(
                not isinstance(s, str) or Path(s).name != s or s in {".", ".."} for s in selected
            ) or len(set(selected)) != len(selected):
                raise ValueError("MODA image_ids must be a nonempty unique list of filename stems")
            expected_hash = manifest.get("source_annotation_sha256")
            if not expected_hash or expected_hash != annotation_inventory(self.root):
                raise ValueError("MODA source annotation inventory differs from split manifest")
        missing_labels = sorted(set(selected) - labels)
        missing_images = sorted(set(selected) - available.keys())
        orphan_images = sorted(available.keys() - labels)
        if missing_labels or missing_images or orphan_images:
            raise FileNotFoundError(
                f"Incomplete MODA partition: {len(missing_images)} missing images, "
                f"{len(missing_labels)} missing labels, {len(orphan_images)} orphan images. "
                f"Examples: {(missing_images or missing_labels or orphan_images)[:3]}. "
                "Complete the dataset or select an explicit, hashed debug split manifest.")
        return [available[stem] for stem in selected]

    def _load_image(self, index):
        return load_moda_image(self.images[index])

    def _parse_annotation(self, index):
        parsed = super()._parse_annotation(index)
        source = parsed["difficulty"]
        if not ((source >= 0) & (source <= 2)).all():
            raise ValueError("Unknown MODA source difficulty; expected 0, 1 or 2")
        # Official MODA default threshold=100 retains all three source values.
        parsed["source_difficulty"] = source
        parsed["difficulty"] = torch.zeros_like(source)
        return parsed

    def load_item(self, index):
        image, target = super().load_item(index)
        target["source_difficulty"] = self._parse_annotation(index)["source_difficulty"]
        target["valid_mask"] = torch.ones(image.shape[-2:], dtype=torch.bool)
        return image, target

    def get_dataset_provenance(self):
        result = super().get_dataset_provenance()
        result.update(input_layout="CWH->CHW", input_channels=8,
                      annotation_protocol="official_all_difficulty_0_1_2",
                      source_annotation_sha256=annotation_inventory(self.root))
        if self.split_file:
            result.update(split_file=str(self.split_file),
                          split_sha256=hashlib.sha256(self.split_file.read_bytes()).hexdigest(),
                          partition=self.partition_manifest.get("partition"),
                          split_protocol=self.partition_manifest.get("protocol"))
        return result
