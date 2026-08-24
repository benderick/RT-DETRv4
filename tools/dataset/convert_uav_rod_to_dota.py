#!/usr/bin/env python3
"""Convert UAV-ROD rotated VOC XML annotations to DOTA text files.

UAV-ROD stores ``(cx, cy, w, h, angle)`` where the angle is in radians and
increases clockwise in image coordinates.  The XML width axis is rotated by
that angle.  DOTA stores the four consecutive rectangle corners followed by
``class difficulty``.  The conversion preserves geometry and VOC difficulty;
it intentionally drops directed car-heading identity because ordinary OBB
detection is pi-periodic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


SCHEMA_VERSION = "uav-rod-voc-to-dota-v1"
DEFAULT_CLASSES = ("car",)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class RotatedVOCObject:
    class_name: str
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    difficult: int
    truncated: int


@dataclass(frozen=True)
class RotatedVOCAnnotation:
    filename: str
    width: int
    height: int
    depth: int
    objects: tuple[RotatedVOCObject, ...]


def _required_text(parent: ET.Element, path: str, xml_path: Path) -> str:
    element = parent.find(path)
    if element is None or element.text is None or not element.text.strip():
        raise ValueError(f"Missing XML field {path!r} in {xml_path}")
    return element.text.strip()


def _finite_float(parent: ET.Element, path: str, xml_path: Path) -> float:
    try:
        value = float(_required_text(parent, path, xml_path))
    except ValueError as exc:
        raise ValueError(f"Invalid float field {path!r} in {xml_path}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite field {path!r} in {xml_path}")
    return value


def parse_rotated_voc(xml_path: Path) -> RotatedVOCAnnotation:
    """Parse and strictly validate one UAV-ROD rotated VOC XML file."""

    xml_path = Path(xml_path)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML annotation: {xml_path}") from exc

    filename = _required_text(root, "filename", xml_path)
    width = int(_required_text(root, "size/width", xml_path))
    height = int(_required_text(root, "size/height", xml_path))
    depth = int(_required_text(root, "size/depth", xml_path))
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    objects = []
    for object_index, element in enumerate(root.findall("object")):
        class_name = _required_text(element, "name", xml_path)
        box = element.find("robndbox")
        if box is None:
            raise ValueError(
                f"Object {object_index} has no robndbox in {xml_path}")
        cx = _finite_float(box, "cx", xml_path)
        cy = _finite_float(box, "cy", xml_path)
        box_width = _finite_float(box, "w", xml_path)
        box_height = _finite_float(box, "h", xml_path)
        angle = _finite_float(box, "angle", xml_path)
        if box_width <= 0 or box_height <= 0:
            raise ValueError(
                f"Object {object_index} has non-positive size in {xml_path}")
        difficult = int(element.findtext("difficult", default="0"))
        truncated = int(element.findtext("truncated", default="0"))
        if difficult < 0 or truncated < 0:
            raise ValueError(
                f"Object {object_index} has negative flags in {xml_path}")
        objects.append(RotatedVOCObject(
            class_name=class_name,
            cx=cx,
            cy=cy,
            width=box_width,
            height=box_height,
            angle=angle,
            difficult=difficult,
            truncated=truncated,
        ))
    return RotatedVOCAnnotation(
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        objects=tuple(objects),
    )


def rotated_box_to_dota_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle: float,
) -> tuple[float, ...]:
    """Return four consecutive corners for UAV-ROD's image-axis convention."""

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for local_x, local_y in (
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ):
        corners.extend((
            cx + cos_a * local_x - sin_a * local_y,
            cy + sin_a * local_x + cos_a * local_y,
        ))
    return tuple(corners)


def _format_coordinate(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def annotation_to_dota_lines(
    annotation: RotatedVOCAnnotation,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> list[str]:
    allowed = set(classes)
    lines = []
    for object_index, instance in enumerate(annotation.objects):
        if instance.class_name not in allowed:
            raise ValueError(
                f"Unknown UAV-ROD class {instance.class_name!r} at object "
                f"{object_index} in {annotation.filename}")
        corners = rotated_box_to_dota_corners(
            instance.cx,
            instance.cy,
            instance.width,
            instance.height,
            instance.angle,
        )
        fields = [_format_coordinate(value) for value in corners]
        fields.extend((instance.class_name, str(instance.difficult)))
        lines.append(" ".join(fields))
    return lines


def _inventory_digest(
    images: Iterable[Path], annotations: Iterable[Path], root: Path
) -> str:
    digest = hashlib.sha256()
    for path in sorted((*images, *annotations)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.suffix.lower() == ".xml":
            digest.update(path.read_bytes())
        else:
            stat = path.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b":")
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _output_digest(labels: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(labels):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _split_inventory(split_root: Path):
    image_dir = split_root / "images"
    xml_dir = split_root / "annotations"
    if not image_dir.is_dir() or not xml_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and annotations/ under {split_root}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    xml_files = sorted(xml_dir.glob("*.xml"))
    image_by_stem = {path.stem: path for path in images}
    xml_by_stem = {path.stem: path for path in xml_files}
    if len(image_by_stem) != len(images):
        raise ValueError(f"Duplicate image stems in {image_dir}")
    missing_xml = sorted(set(image_by_stem) - set(xml_by_stem))
    orphan_xml = sorted(set(xml_by_stem) - set(image_by_stem))
    if missing_xml or orphan_xml:
        raise ValueError(
            f"Image/XML inventory mismatch in {split_root}: "
            f"missing_xml={missing_xml[:10]}, orphan_xml={orphan_xml[:10]}")
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return images, xml_files, image_by_stem, xml_by_stem


def convert_split(
    split_root: Path,
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    validate_only: bool = False,
) -> dict:
    """Convert or validate one official UAV-ROD split."""

    split_root = Path(split_root).expanduser().resolve()
    images, xml_files, image_by_stem, xml_by_stem = _split_inventory(split_root)
    output_dir = split_root / "annfile"
    manifest_path = split_root / "conversion_manifest.json"
    if not validate_only and (output_dir.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite an existing conversion under {split_root}; "
            "remove it explicitly only after preserving any user changes")
    if validate_only and not output_dir.is_dir():
        raise FileNotFoundError(f"Converted DOTA labels do not exist: {output_dir}")

    temporary_dir = None
    write_dir = output_dir
    if not validate_only:
        temporary_dir = Path(tempfile.mkdtemp(
            prefix=".annfile.tmp-", dir=split_root))
        write_dir = temporary_dir

    class_counts: Counter[str] = Counter()
    difficult_count = 0
    truncated_count = 0
    outside_image_count = 0
    empty_image_count = 0
    dimension_counts: Counter[str] = Counter()
    try:
        for stem in sorted(image_by_stem):
            image_path = image_by_stem[stem]
            xml_path = xml_by_stem[stem]
            annotation = parse_rotated_voc(xml_path)
            if Path(annotation.filename).stem != stem:
                raise ValueError(
                    f"XML filename mismatch in {xml_path}: {annotation.filename!r}")
            with Image.open(image_path) as image:
                actual_size = image.size
            if actual_size != (annotation.width, annotation.height):
                raise ValueError(
                    f"Image/XML size mismatch for {image_path}: image={actual_size}, "
                    f"xml={(annotation.width, annotation.height)}")
            dimension_counts[f"{annotation.width}x{annotation.height}"] += 1
            lines = annotation_to_dota_lines(annotation, classes)
            if not lines:
                empty_image_count += 1
            for instance in annotation.objects:
                class_counts[instance.class_name] += 1
                difficult_count += int(instance.difficult != 0)
                truncated_count += int(instance.truncated != 0)
                corners = rotated_box_to_dota_corners(
                    instance.cx,
                    instance.cy,
                    instance.width,
                    instance.height,
                    instance.angle,
                )
                xs, ys = corners[0::2], corners[1::2]
                outside_image_count += int(
                    min(xs) < 0 or min(ys) < 0
                    or max(xs) > annotation.width
                    or max(ys) > annotation.height
                )
            expected = "\n".join(lines) + ("\n" if lines else "")
            destination = write_dir / f"{stem}.txt"
            if validate_only:
                if not destination.is_file():
                    raise FileNotFoundError(destination)
                actual = destination.read_text(encoding="utf-8")
                if actual != expected:
                    raise ValueError(
                        f"Converted label differs from source XML: {destination}")
            else:
                destination.write_text(expected, encoding="utf-8")

        source_digest = _inventory_digest(images, xml_files, split_root)
        labels = sorted(write_dir.glob("*.txt"))
        if len(labels) != len(images):
            raise RuntimeError(
                f"Expected {len(images)} converted labels, found {len(labels)}")
        output_digest = _output_digest(labels, write_dir)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "split": split_root.name,
            "source_format": "VOC XML robndbox(cx,cy,w,h,angle_radians_clockwise)",
            "target_format": "DOTA quadrilateral class difficulty",
            "heading_policy": (
                "Directed 2pi car heading is intentionally reduced to the "
                "ordinary pi-periodic OBB geometry used by this detector."
            ),
            "angle_axis": "XML w-axis; positive is clockwise in image coordinates",
            "corner_order": "four consecutive corners from local (-w/2,-h/2)",
            "images": len(images),
            "annotations": len(xml_files),
            "objects": sum(class_counts.values()),
            "classes": list(classes),
            "class_counts": dict(sorted(class_counts.items())),
            "difficult_objects": difficult_count,
            "truncated_objects": truncated_count,
            "objects_with_any_corner_outside_image": outside_image_count,
            "empty_images": empty_image_count,
            "image_dimensions": dict(sorted(dimension_counts.items())),
            "source_inventory_sha256": source_digest,
            "output_inventory_sha256": output_digest,
        }
        if not validate_only:
            os.replace(write_dir, output_dir)
            temporary_dir = None
            temporary_manifest = split_root / ".conversion_manifest.json.tmp"
            temporary_manifest.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, manifest_path)
        elif manifest_path.is_file():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in (
                "schema_version", "images", "objects", "class_counts",
                "source_inventory_sha256", "output_inventory_sha256",
            ):
                if recorded.get(key) != summary.get(key):
                    raise ValueError(
                        f"Conversion manifest mismatch for {key!r} in {manifest_path}")
        return summary
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert UAV-ROD rotated VOC annotations to DOTA labels")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../data/UAV-ROD"),
        help="UAV-ROD root containing official train/ and test/ splits",
    )
    parser.add_argument(
        "--splits", nargs="+", default=("train", "test"),
        help="Official split directory names to convert",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Recompute every label and verify an existing conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    for split in args.splits:
        summary = convert_split(
            args.dataset_root / split,
            validate_only=args.validate_only,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "event": "validation_finished" if args.validate_only else "conversion_finished",
        "splits": [summary["split"] for summary in summaries],
        "images": sum(summary["images"] for summary in summaries),
        "objects": sum(summary["objects"] for summary in summaries),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
