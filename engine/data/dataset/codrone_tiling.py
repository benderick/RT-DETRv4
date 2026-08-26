"""Reproducible DOTA-style tiling for the original CODrone splits.

The implementation follows the published CODrone split configuration and the
official DOTA-style splitter.  In particular, ``gap`` means overlap and the
actual step is ``window_size - gap``.  The final window along an axis is
snapped to the image boundary, which can make its overlap larger than
``gap``.

Besides materialising image patches and DOTA annotations, this module
writes structured evidence for every source image, tile, and source object.
Those records preserve preprocessing and boundary evidence without having to
reconstruct tiling decisions after an experiment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .codrone_dataset import CODRONE_CLASSES, _CLASS_ALIASES


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CODroneTilingProtocol:
    """Published CODrone single-scale tiling protocol.

    ``padding_value`` is applied through OpenCV, hence its channel order is
    BGR.  This exactly follows the official splitter implementation.  The
    corresponding RGB value after decoding a saved patch is (124, 116, 104).
    """

    window_size: int = 1180
    gap: int = 200
    image_rate_threshold: float = 0.6
    iof_threshold: float = 0.7
    padding_value: Tuple[int, int, int] = (104, 116, 124)
    image_extension: str = ".jpg"
    image_quality: int = 95

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.gap < 0 or self.gap >= self.window_size:
            raise ValueError("gap must satisfy 0 <= gap < window_size")
        if not 0.0 <= self.image_rate_threshold <= 1.0:
            raise ValueError("image_rate_threshold must be in [0, 1]")
        if not 0.0 <= self.iof_threshold <= 1.0:
            raise ValueError("iof_threshold must be in [0, 1]")
        if len(self.padding_value) != 3 or any(not 0 <= value <= 255 for value in self.padding_value):
            raise ValueError("padding_value must contain three uint8 channel values")
        if self.image_extension.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            raise ValueError(f"Unsupported output extension: {self.image_extension}")
        if not 1 <= self.image_quality <= 100:
            raise ValueError("image_quality must be in [1, 100]")

    @property
    def step(self) -> int:
        return self.window_size - self.gap

    def to_manifest(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.update({
            "step": self.step,
            "gap_semantics": "overlap; step = window_size - gap",
            "last_window_policy": "snap_to_image_boundary",
            "image_rate_comparison": "> threshold (official-code semantics)",
            "iof_comparison": ">= threshold",
            "iof_denominator": "source_polygon_area",
            "truncated_definition": "iof < 1.0",
            "truncated_difficulty": 2,
            "polygon_policy": "translate_without_clipping",
            "padding_channel_order": "OpenCV BGR",
            "padding_value_rgb_after_decode": list(reversed(self.padding_value)),
            "image_encoding": (
                f"JPEG quality {self.image_quality}"
                if self.image_extension.lower() in {".jpg", ".jpeg"}
                else "lossless PNG" if self.image_extension.lower() == ".png"
                else self.image_extension.lower().lstrip(".")
            ),
        })
        payload["padding_value"] = list(self.padding_value)
        return payload


@dataclass(frozen=True)
class TileWindow:
    x_start: int
    y_start: int
    x_stop: int
    y_stop: int
    effective_image_ratio: float
    valid_width: int
    valid_height: int

    @property
    def width(self) -> int:
        return self.x_stop - self.x_start

    @property
    def height(self) -> int:
        return self.y_stop - self.y_start

    @property
    def padding(self) -> Tuple[int, int, int, int]:
        return (0, 0, self.width - self.valid_width, self.height - self.valid_height)

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x_start, self.y_start, self.x_stop, self.y_stop


@dataclass(frozen=True)
class DOTAObject:
    source_index: int
    polygon: Tuple[float, ...]
    class_name: str
    difficulty: int
    ignored: bool


def _axis_starts(length: int, size: int, step: int) -> List[int]:
    if length <= size:
        return [0]
    count = int(math.ceil((length - size) / step + 1))
    starts = [step * index for index in range(count)]
    if len(starts) > 1 and starts[-1] + size > length:
        starts[-1] = length - size
    return starts


def generate_sliding_windows(
    width: int,
    height: int,
    protocol: CODroneTilingProtocol = CODroneTilingProtocol(),
) -> List[TileWindow]:
    """Generate windows with the same ordering and filtering as DOTA devkit.

    OpenCV/Python slicing keeps all starts non-negative.  If no candidate has
    image ratio strictly above the configured threshold, every candidate
    within 0.01 of the maximum ratio is retained, matching the official
    splitter's small-image fallback.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    x_starts = _axis_starts(width, protocol.window_size, protocol.step)
    y_starts = _axis_starts(height, protocol.window_size, protocol.step)
    candidates = []
    for x_start in x_starts:
        for y_start in y_starts:
            x_stop = x_start + protocol.window_size
            y_stop = y_start + protocol.window_size
            valid_width = max(0, min(x_stop, width) - max(x_start, 0))
            valid_height = max(0, min(y_stop, height) - max(y_start, 0))
            ratio = (valid_width * valid_height) / float(protocol.window_size ** 2)
            candidates.append(TileWindow(
                x_start=x_start,
                y_start=y_start,
                x_stop=x_stop,
                y_stop=y_stop,
                effective_image_ratio=ratio,
                valid_width=valid_width,
                valid_height=valid_height,
            ))

    keep = [window.effective_image_ratio > protocol.image_rate_threshold for window in candidates]
    if not any(keep):
        maximum = max(window.effective_image_ratio for window in candidates)
        keep = [abs(window.effective_image_ratio - maximum) < 0.01 for window in candidates]
    return [window for window, retained in zip(candidates, keep) if retained]


def polygon_area(polygon: Sequence[float]) -> float:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _clip_edge(points: List[np.ndarray], inside, intersection) -> List[np.ndarray]:
    if not points:
        return []
    output = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous, previous_inside = current, current_inside
    return output


def _line_intersection(first: np.ndarray, second: np.ndarray, axis: int, value: float) -> np.ndarray:
    delta = second - first
    if abs(float(delta[axis])) < np.finfo(np.float64).eps:
        result = first.copy()
        result[axis] = value
        return result
    fraction = (value - first[axis]) / delta[axis]
    return first + fraction * delta


def clip_polygon_to_window(polygon: Sequence[float], window: TileWindow) -> np.ndarray:
    """Clip a convex CODrone quadrilateral to an axis-aligned window."""

    points = [point.copy() for point in np.asarray(polygon, dtype=np.float64).reshape(-1, 2)]
    points = _clip_edge(
        points, lambda point: point[0] >= window.x_start,
        lambda first, second: _line_intersection(first, second, 0, window.x_start))
    points = _clip_edge(
        points, lambda point: point[0] <= window.x_stop,
        lambda first, second: _line_intersection(first, second, 0, window.x_stop))
    points = _clip_edge(
        points, lambda point: point[1] >= window.y_start,
        lambda first, second: _line_intersection(first, second, 1, window.y_start))
    points = _clip_edge(
        points, lambda point: point[1] <= window.y_stop,
        lambda first, second: _line_intersection(first, second, 1, window.y_stop))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def polygon_window_iof(polygon: Sequence[float], window: TileWindow) -> float:
    area = polygon_area(polygon)
    if area <= np.finfo(np.float64).eps:
        return 0.0
    clipped = clip_polygon_to_window(polygon, window)
    intersection = polygon_area(clipped.reshape(-1)) if len(clipped) else 0.0
    return min(1.0, max(0.0, intersection / area))


def boundary_distances(polygon: Sequence[float], window: TileWindow) -> Tuple[float, float]:
    """Return center and signed polygon-support distance to a tile boundary."""

    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    center = points.mean(axis=0)
    center_distance = min(
        center[0] - window.x_start,
        window.x_stop - center[0],
        center[1] - window.y_start,
        window.y_stop - center[1],
    )
    support_distance = min(
        points[:, 0].min() - window.x_start,
        window.x_stop - points[:, 0].max(),
        points[:, 1].min() - window.y_start,
        window.y_stop - points[:, 1].max(),
    )
    return float(center_distance), float(support_distance)


def parse_dota_annotations(
    annotation_path: Path,
    *,
    strict: bool = True,
    invalid_records: Optional[List[Dict[str, object]]] = None,
) -> List[DOTAObject]:
    """Parse DOTA rows and optionally tolerate invalid zero-area objects.

    The public parser remains strict by default so annotation audits fail
    loudly.  The materializer uses ``strict=False``: a zero-area polygon has
    no valid OBB geometry and is excluded from training/evaluation, while its
    complete source row is written to the invalid-annotation audit stream.
    """
    objects = []
    if not annotation_path.is_file():
        return objects
    known_classes = set(CODRONE_CLASSES)
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) < 9:
                raise ValueError(f"Malformed DOTA annotation {annotation_path}:{line_number}")
            try:
                polygon = tuple(float(value) for value in fields[:8])
            except ValueError as error:
                raise ValueError(
                    f"Invalid coordinates in {annotation_path}:{line_number}") from error
            class_name = _CLASS_ALIASES.get(fields[8], fields[8])
            ignored = class_name.lower() in {"ignored", "ignore"}
            if not ignored and class_name not in known_classes:
                raise ValueError(
                    f"Unknown CODrone class {class_name!r} in {annotation_path}:{line_number}")
            difficulty = int(fields[9]) if len(fields) > 9 else 0
            area = polygon_area(polygon)
            if not np.isfinite(polygon).all() or area <= np.finfo(np.float64).eps:
                reason = "nonfinite_polygon" if not np.isfinite(polygon).all() else "degenerate_polygon"
                record = {
                    "annotation_file": str(annotation_path),
                    "line_number": line_number,
                    "class_name": "ignored" if ignored else class_name,
                    "difficulty": difficulty,
                    "polygon": list(polygon),
                    "area_px": float(area) if np.isfinite(area) else None,
                    "reason": reason,
                }
                if invalid_records is not None:
                    invalid_records.append(record)
                if strict:
                    raise ValueError(
                        f"{reason.replace('_', ' ').capitalize()} in "
                        f"{annotation_path}:{line_number}")
                continue
            objects.append(DOTAObject(
                source_index=len(objects),
                polygon=polygon,
                class_name="ignored" if ignored else class_name,
                difficulty=difficulty,
                ignored=ignored,
            ))
    return objects


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump_atomic(path: Path, payload: Dict[str, object]):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _format_coordinate(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _partition_id(protocol: CODroneTilingProtocol) -> str:
    iof = _format_coordinate(protocol.iof_threshold).replace(".", "p")
    return f"codrone_dota_w{protocol.window_size}_g{protocol.gap}_iof{iof}"


def _write_tile_annotation(path: Path, assignments: Sequence[Dict[str, object]]):
    with path.open("w", encoding="utf-8") as handle:
        for assignment in assignments:
            difficulty = 2 if assignment["truncated"] else assignment["source_difficulty"]
            fields = [
                *(_format_coordinate(value) for value in assignment["polygon_local"]),
                assignment["class_name"],
                str(int(difficulty)),
            ]
            handle.write(" ".join(fields) + "\n")


def _write_patch(path: Path, image: np.ndarray, protocol: CODroneTilingProtocol):
    parameters = []
    if protocol.image_extension.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, protocol.image_quality]
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"Failed to write patch: {path}")


def _render_preview(
    image_path: Path,
    output_path: Path,
    tile_record: Dict[str, object],
    assignments: Sequence[Dict[str, object]],
):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read generated patch for preview: {image_path}")
    valid_width, valid_height = tile_record["valid_size"]
    if valid_width < image.shape[1] or valid_height < image.shape[0]:
        cv2.rectangle(image, (0, 0), (max(valid_width - 1, 0), max(valid_height - 1, 0)),
                      (255, 255, 0), 2)
    for assignment in assignments:
        polygon = np.asarray(assignment["polygon_local"], dtype=np.float64).reshape(4, 2)
        polygon = np.rint(polygon).astype(np.int32)
        if assignment["ignored"]:
            color = (255, 0, 255)
        elif assignment["truncated"]:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
        anchor = tuple(polygon[0].tolist())
        label = f'{assignment["class_name"]} iof={assignment["iof"]:.2f}'
        cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    header = (
        f'{tile_record["source_image_id"]} xy=({tile_record["window"][0]},'
        f'{tile_record["window"][1]}) objects={len(assignments)}')
    cv2.putText(image, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


def _split_one_image(task: Dict[str, object]) -> Dict[str, object]:
    image_path = Path(task["image_path"])
    annotation_path = Path(task["annotation_path"])
    output_dir = Path(task["output_dir"])
    protocol = CODroneTilingProtocol(**task["protocol"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode source image: {image_path}")
    height, width = image.shape[:2]
    windows = generate_sliding_windows(width, height, protocol)
    invalid_annotations = []
    objects = parse_dota_annotations(
        annotation_path, strict=False, invalid_records=invalid_annotations)
    tile_records = []
    object_records = [{
        "schema_version": SCHEMA_VERSION,
        "split": task["split"],
        "source_image_id": image_path.stem,
        "source_object_index": obj.source_index,
        "object_uid": f"{image_path.stem}:{obj.source_index}",
        "class_name": obj.class_name,
        "ignored": obj.ignored,
        "source_difficulty": obj.difficulty,
        "polygon_global": list(obj.polygon),
        "area_px": polygon_area(obj.polygon),
        "positive_window_assignments": [],
        "retained_tile_ids": [],
        "max_iof": 0.0,
    } for obj in objects]
    preview_candidates = []
    tile_payloads = []

    for tile_index, window in enumerate(windows):
        tile_id = (
            f"{image_path.stem}__{window.width}__"
            f"{window.x_start}___{window.y_start}")
        tile_image_name = tile_id + protocol.image_extension
        tile_annotation_name = tile_id + ".txt"
        assignments = []
        for obj, object_record in zip(objects, object_records):
            iof = polygon_window_iof(obj.polygon, window)
            object_record["max_iof"] = max(object_record["max_iof"], iof)
            if iof <= 0.0:
                continue
            center_distance, support_distance = boundary_distances(obj.polygon, window)
            retained = iof >= protocol.iof_threshold
            truncated = iof < 1.0
            local_polygon = np.asarray(obj.polygon, dtype=np.float64) - np.asarray(
                [window.x_start, window.y_start] * 4, dtype=np.float64)
            diagnostic_assignment = {
                "tile_id": tile_id,
                "tile_index": tile_index,
                "iof": float(iof),
                "retained": retained,
                "truncated": truncated,
                "center_to_boundary_px": center_distance,
                "support_to_boundary_px": support_distance,
            }
            object_record["positive_window_assignments"].append(diagnostic_assignment)
            if not retained:
                continue
            object_record["retained_tile_ids"].append(tile_id)
            assignments.append({
                **diagnostic_assignment,
                "source_object_index": obj.source_index,
                "object_uid": object_record["object_uid"],
                "class_name": obj.class_name,
                "ignored": obj.ignored,
                "source_difficulty": obj.difficulty,
                "polygon_global": list(obj.polygon),
                "polygon_local": local_polygon.tolist(),
            })

        patch = image[window.y_start:window.y_stop, window.x_start:window.x_stop]
        if patch.shape[0] != window.height or patch.shape[1] != window.width:
            padded = np.empty((window.height, window.width, image.shape[2]), dtype=np.uint8)
            padded[...] = np.asarray(protocol.padding_value, dtype=np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        image_output = output_dir / "images" / tile_image_name
        annotation_output = output_dir / "annfile" / tile_annotation_name
        _write_patch(image_output, patch, protocol)
        _write_tile_annotation(annotation_output, assignments)
        tile_record = {
            "schema_version": SCHEMA_VERSION,
            "split": task["split"],
            "tile_index": tile_index,
            "tile_id": tile_id,
            "source_image_id": image_path.stem,
            "source_image_name": image_path.name,
            "source_size": [width, height],
            "window": list(window.xyxy),
            "window_size": [window.width, window.height],
            "valid_size": [window.valid_width, window.valid_height],
            "effective_image_ratio": window.effective_image_ratio,
            "padding_ltrb": list(window.padding),
            "image_file": f"images/{tile_image_name}",
            "annotation_file": f"annfile/{tile_annotation_name}",
            "metadata_file": f"metadata/{tile_id}.json",
            "retained_object_count": len(assignments),
            "retained_regular_count": sum(not item["ignored"] for item in assignments),
            "retained_ignore_count": sum(item["ignored"] for item in assignments),
            "truncated_count": sum(item["truncated"] for item in assignments),
            "retained_object_uids": [item["object_uid"] for item in assignments],
        }
        tile_records.append(tile_record)
        tile_payloads.append((tile_record, assignments))
        preview_candidates.append((
            sum(item["truncated"] for item in assignments),
            len(assignments),
            image_output,
            tile_record,
            assignments,
        ))

    for object_record in object_records:
        object_record["retained_assignment_count"] = len(object_record["retained_tile_ids"])
        object_record["dropped_by_iof"] = not object_record["retained_tile_ids"]

    partition_id = _partition_id(protocol)
    for tile_record, assignments in tile_payloads:
        regular_gt_index = 0
        metadata_objects = []
        for annotation_index, assignment in enumerate(assignments):
            object_record = object_records[assignment["source_object_index"]]
            area = float(object_record["area_px"])
            metadata_object = {
                "tile_annotation_index": annotation_index,
                "gt_index": None if assignment["ignored"] else regular_gt_index,
                "source_object_index": assignment["source_object_index"],
                "source_object_uid": assignment["object_uid"],
                "class_name": assignment["class_name"],
                "ignored": assignment["ignored"],
                "visible_ratio": assignment["iof"],
                "truncated": assignment["truncated"],
                "source_tile_count": object_record["retained_assignment_count"],
                "center_boundary_distance_px": assignment["center_to_boundary_px"],
                "boundary_distance_px": assignment["support_to_boundary_px"],
                "boundary_distance_object_scale": (
                    assignment["support_to_boundary_px"] / max(math.sqrt(area), 1e-7)),
            }
            metadata_objects.append(metadata_object)
            if not assignment["ignored"]:
                regular_gt_index += 1
        _json_dump_atomic(output_dir / tile_record["metadata_file"], {
            "schema_version": SCHEMA_VERSION,
            "partition_id": partition_id,
            "tile_id": tile_record["tile_id"],
            "source_image_id": image_path.stem,
            "source_image_name": image_path.name,
            "source_size": [width, height],
            "tile_origin": [tile_record["window"][0], tile_record["window"][1]],
            "tile_size": tile_record["window_size"],
            "tile_overlap": [protocol.gap, protocol.gap],
            "tile_step": [protocol.step, protocol.step],
            "objects": metadata_objects,
        })

    preview_file = None
    if task["preview"] and preview_candidates:
        candidate = max(preview_candidates, key=lambda item: (item[0], item[1], -item[3]["tile_index"]))
        preview_file = f'{image_path.stem}__preview.jpg'
        _render_preview(candidate[2], output_dir / "diagnostics" / "visualizations" / preview_file,
                        candidate[3], candidate[4])

    source_record = {
        "schema_version": SCHEMA_VERSION,
        "split": task["split"],
        "source_image_id": image_path.stem,
        "source_image_name": image_path.name,
        "source_size": [width, height],
        "source_image_sha256": _sha256_file(image_path),
        "source_annotation_name": annotation_path.name,
        "source_annotation_exists": annotation_path.is_file(),
        "source_annotation_sha256": _sha256_file(annotation_path) if annotation_path.is_file() else None,
        "window_count": len(windows),
        "object_count": sum(not obj.ignored for obj in objects),
        "ignore_region_count": sum(obj.ignored for obj in objects),
        "invalid_annotation_count": len(invalid_annotations),
        "invalid_annotation_lines": [item["line_number"] for item in invalid_annotations],
        "retained_assignment_count": sum(record["retained_assignment_count"] for record in object_records),
        "dropped_object_count": sum(record["dropped_by_iof"] for record in object_records),
        "preview_file": f"diagnostics/visualizations/{preview_file}" if preview_file else None,
    }
    return {
        "source": source_record,
        "tiles": tile_records,
        "objects": object_records,
        "invalid_annotations": invalid_annotations,
    }


def _write_jsonl_record(handle, record: Dict[str, object]):
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _update_inventory_hash(digest, record: Dict[str, object]):
    fields = (
        record["source_image_name"],
        record["source_image_sha256"],
        record["source_annotation_name"],
        record["source_annotation_sha256"],
    )
    digest.update(json.dumps(fields, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")


def _iter_split_results(tasks: Sequence[Dict[str, object]], nproc: int) -> Iterator[Dict[str, object]]:
    """Yield results in source-image order without retaining the full split in RAM."""

    if nproc == 1:
        for task in tasks:
            yield _split_one_image(task)
        return
    with ProcessPoolExecutor(max_workers=nproc) as executor:
        yield from executor.map(_split_one_image, tasks)


def split_codrone_split(
    source_split: Path,
    output_split: Path,
    protocol: CODroneTilingProtocol = CODroneTilingProtocol(),
    *,
    split_name: Optional[str] = None,
    nproc: int = 1,
    preview_samples: int = 8,
    preview_seed: int = 0,
    max_images: Optional[int] = None,
    overwrite: bool = False,
    provenance: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Materialise one CODrone split and return its audit summary."""

    source_split = Path(source_split).expanduser().resolve()
    output_split = Path(output_split).expanduser().resolve()
    image_dir, annotation_dir = source_split / "images", source_split / "annfile"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Expected CODrone images/ and annfile/ below {source_split}")
    if (
        output_split == source_split
        or output_split in source_split.parents
        or source_split in output_split.parents
    ):
        raise ValueError("Source and output directories must not replace or contain each other")
    if nproc <= 0:
        raise ValueError("nproc must be positive")
    if output_split.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_split}; pass overwrite=True explicitly")
        shutil.rmtree(output_split)

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        images = images[:max_images]
    if not images:
        raise RuntimeError(f"No source images found in {image_dir}")
    split_name = split_name or source_split.name
    output_split.mkdir(parents=True)
    for relative in (
        "images", "annfile", "metadata", "diagnostics", "diagnostics/visualizations",
    ):
        (output_split / relative).mkdir()

    preview_count = min(max(0, preview_samples), len(images))
    preview_indices = set(random.Random(preview_seed).sample(range(len(images)), preview_count))
    protocol_kwargs = asdict(protocol)
    protocol_kwargs["padding_value"] = tuple(protocol.padding_value)
    tasks = [{
        "image_path": str(image_path),
        "annotation_path": str(annotation_dir / f"{image_path.stem}.txt"),
        "output_dir": str(output_split),
        "split": split_name,
        "protocol": protocol_kwargs,
        "preview": index in preview_indices,
    } for index, image_path in enumerate(images)]

    started_at = time.time()
    in_progress = {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "source_split": str(source_split),
        "output_split": str(output_split),
        "split": split_name,
        "selected_source_images": len(images),
        "protocol": protocol.to_manifest(),
        "provenance": provenance or {},
    }
    _json_dump_atomic(output_split / "manifest.json", in_progress)

    diagnostic_dir = output_split / "diagnostics"
    source_count = 0
    tile_count = 0
    empty_tile_count = 0
    padded_tile_count = 0
    regular_object_count = 0
    ignore_region_count = 0
    retained_assignment_count = 0
    retained_regular_assignment_count = 0
    retained_ignore_assignment_count = 0
    truncated_assignment_count = 0
    truncated_regular_assignment_count = 0
    truncated_ignore_assignment_count = 0
    dropped_regular_count = 0
    duplicated_regular_count = 0
    invalid_annotation_count = 0
    minimum_image_ratio = math.inf
    maximum_image_ratio = -math.inf
    source_class_counts = Counter()
    assignment_class_counts = Counter()
    inventory_digest = hashlib.sha256()

    with (
        gzip.open(diagnostic_dir / "images.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as image_log,
        gzip.open(diagnostic_dir / "tiles.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as tile_log,
        gzip.open(diagnostic_dir / "objects.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as object_log,
        gzip.open(diagnostic_dir / "invalid_annotations.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as invalid_log,
    ):
        for result in _iter_split_results(tasks, nproc):
            source_record = result["source"]
            _write_jsonl_record(image_log, source_record)
            _update_inventory_hash(inventory_digest, source_record)
            source_count += 1

            for invalid_record in result["invalid_annotations"]:
                _write_jsonl_record(invalid_log, {
                    "split": split_name,
                    "source_image_id": source_record["source_image_id"],
                    **invalid_record,
                })
                invalid_annotation_count += 1

            for tile_record in result["tiles"]:
                _write_jsonl_record(tile_log, tile_record)
                tile_count += 1
                empty_tile_count += tile_record["retained_regular_count"] == 0
                padded_tile_count += any(tile_record["padding_ltrb"])
                minimum_image_ratio = min(minimum_image_ratio, tile_record["effective_image_ratio"])
                maximum_image_ratio = max(maximum_image_ratio, tile_record["effective_image_ratio"])

            for object_record in result["objects"]:
                _write_jsonl_record(object_log, object_record)
                retained_count = object_record["retained_assignment_count"]
                retained_assignment_count += retained_count
                object_truncated_count = sum(
                    assignment["retained"] and assignment["truncated"]
                    for assignment in object_record["positive_window_assignments"]
                )
                truncated_assignment_count += object_truncated_count
                if object_record["ignored"]:
                    ignore_region_count += 1
                    retained_ignore_assignment_count += retained_count
                    truncated_ignore_assignment_count += object_truncated_count
                    continue
                regular_object_count += 1
                retained_regular_assignment_count += retained_count
                truncated_regular_assignment_count += object_truncated_count
                source_class_counts[object_record["class_name"]] += 1
                assignment_class_counts[object_record["class_name"]] += retained_count
                dropped_regular_count += object_record["dropped_by_iof"]
                duplicated_regular_count += retained_count > 1

    elapsed = time.time() - started_at
    summary = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "source_images": source_count,
        "tiles": tile_count,
        "empty_tiles": empty_tile_count,
        "padded_tiles": padded_tile_count,
        "regular_source_objects": regular_object_count,
        "ignore_source_regions": ignore_region_count,
        "invalid_annotation_count": invalid_annotation_count,
        "retained_assignments": retained_assignment_count,
        "retained_regular_assignments": retained_regular_assignment_count,
        "retained_ignore_assignments": retained_ignore_assignment_count,
        "truncated_retained_assignments": truncated_assignment_count,
        "truncated_regular_assignments": truncated_regular_assignment_count,
        "truncated_ignore_assignments": truncated_ignore_assignment_count,
        "dropped_regular_objects": dropped_regular_count,
        "duplicated_regular_objects": duplicated_regular_count,
        "minimum_effective_image_ratio": minimum_image_ratio,
        "maximum_effective_image_ratio": maximum_image_ratio,
        "source_class_counts": dict(sorted(source_class_counts.items())),
        "retained_assignment_class_counts": dict(sorted(assignment_class_counts.items())),
        "source_inventory_sha256": inventory_digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "diagnostic_files": {
            "images": "diagnostics/images.jsonl.gz",
            "tiles": "diagnostics/tiles.jsonl.gz",
            "objects": "diagnostics/objects.jsonl.gz",
            "invalid_annotations": "diagnostics/invalid_annotations.jsonl.gz",
            "visualizations": "diagnostics/visualizations/",
            "tile_metadata": "metadata/",
        },
    }
    _json_dump_atomic(diagnostic_dir / "summary.json", summary)
    manifest = {
        **in_progress,
        "status": "complete",
        "source_inventory_sha256": summary["source_inventory_sha256"],
        "summary_file": "diagnostics/summary.json",
        "elapsed_seconds": elapsed,
        "completed_unix_time": time.time(),
    }
    _json_dump_atomic(output_split / "manifest.json", manifest)
    (output_split / "_SUCCESS").write_text(
        summary["source_inventory_sha256"] + "\n", encoding="utf-8")
    return summary


__all__ = [
    "CODroneTilingProtocol",
    "DOTAObject",
    "TileWindow",
    "boundary_distances",
    "clip_polygon_to_window",
    "generate_sliding_windows",
    "parse_dota_annotations",
    "polygon_area",
    "polygon_window_iof",
    "split_codrone_split",
]
