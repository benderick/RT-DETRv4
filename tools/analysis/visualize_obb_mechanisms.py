#!/usr/bin/env python3
"""Render mechanism-near OBB failure visualizations from experiment logs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "gt": (30, 220, 80),
    "query": (255, 70, 55),
    "final": (50, 205, 255),
    "suppressor": (225, 70, 255),
    "center": (255, 220, 40),
}


def _root(path):
    path = path.expanduser().resolve()
    return path / "diagnostics" if (path / "diagnostics").is_dir() else path


def _epoch(root, epoch):
    base = root / "eval"
    if epoch != "latest":
        return base / (epoch if epoch.startswith("epoch_") else f"epoch_{int(epoch):04d}")
    candidates = sorted(base.glob("epoch_*"))
    return candidates[-1] if candidates else base / "standalone"


def _records(directory, stem):
    for path in sorted(directory.glob(f"{stem}.rank*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _corners(box):
    cx, cy, width, height, angle = box
    cosine, sine = math.cos(angle), math.sin(angle)
    result = []
    for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                 (width / 2, height / 2), (-width / 2, height / 2)):
        result.append((cx + cosine * x - sine * y, cy + sine * x + cosine * y))
    return result


def _draw_box(draw, box, color, width=3):
    if box is None:
        return
    points = _corners(box)
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def _draw_axis(draw, box, color, width=3):
    if box is None:
        return
    cx, cy, length, _, angle = box
    dx, dy = 0.5 * length * math.cos(angle), 0.5 * length * math.sin(angle)
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=color, width=width)


def _priority(record):
    raw_iou = float(record.get("rotated_iou") or 0.0)
    final_iou = float(record.get("final_rotated_iou") or 0.0)
    angle = float(record.get("angle_error_deg") or 0.0)
    center = float(record.get("center_error_gt_diagonal") or 0.0)
    # NMS mistakes with a good raw query are especially direct evidence;
    # otherwise prioritize large decomposed localization errors.
    bonus = 5.0 if record.get("failure_stage") == "rotated_nms" else 0.0
    return bonus + 3.0 * raw_iou * (1.0 - final_iou) + angle / 45.0 + min(center, 3.0)


def _caption_panel(image, lines, height=92):
    panel = Image.new("RGB", (image.width, image.height + height), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    y = image.height + 8
    for line in lines:
        draw.text((8, y), line, fill=(20, 20, 20), font=font)
        y += 16
    return panel


def _crop_box(image, boxes):
    points = [point for box in boxes if box is not None for point in _corners(box)]
    if not points:
        return (0, 0, image.width, image.height)
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    margin = max(80.0, 1.5 * max(width, height))
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    half_width = max(160.0, 0.5 * width + margin)
    half_height = max(160.0, 0.5 * height + margin)
    left, top = max(0, int(cx - half_width)), max(0, int(cy - half_height))
    right, bottom = min(image.width, int(cx + half_width)), min(image.height, int(cy + half_height))
    return left, top, right, bottom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--output", type=Path, default=Path("obb_mechanism_cases"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--stage", action="append", help="restrict to one or more failure_stage values")
    args = parser.parse_args()
    directory = _epoch(_root(args.run), args.epoch)
    candidates = [record for record in _records(directory, "matches")
                  if record.get("query_index") is not None and
                  (not args.stage or record.get("failure_stage") in args.stage)]
    candidates.sort(key=_priority, reverse=True)
    selected = candidates[:args.count]
    selected_keys = {(record["image_id"], record.get("pre_nms_candidate_index"))
                     for record in selected if record.get("pre_nms_candidate_index") is not None}
    nms_by_key = {}
    parent_keys = set()
    for record in _records(directory, "nms"):
        key = (record["image_id"], record["pre_nms_index"])
        if key in selected_keys:
            nms_by_key[key] = record
            parent = record.get("suppressed_by_pre_nms_index")
            if parent is not None:
                parent_keys.add((record["image_id"], parent))
        elif key in parent_keys:
            nms_by_key[key] = record
    # Parent rows can precede their suppressed child in score order; make one
    # second streaming pass for any parent not captured above.
    missing = parent_keys - nms_by_key.keys()
    if missing:
        for record in _records(directory, "nms"):
            key = (record["image_id"], record["pre_nms_index"])
            if key in missing:
                nms_by_key[key] = record

    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for case_index, record in enumerate(selected):
        image_path = record.get("image_path")
        if not image_path or not Path(image_path).is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        gt_box, query_box = record.get("gt_box"), record.get("query_box")
        _draw_box(draw, gt_box, COLORS["gt"], 4)
        _draw_axis(draw, gt_box, COLORS["gt"], 3)
        _draw_box(draw, query_box, COLORS["query"], 4)
        _draw_axis(draw, query_box, COLORS["query"], 3)
        _draw_box(draw, record.get("final_box"), COLORS["final"], 3)
        if gt_box and query_box:
            draw.line([(gt_box[0], gt_box[1]), (query_box[0], query_box[1])],
                      fill=COLORS["center"], width=3)
        candidate_key = (record["image_id"], record.get("pre_nms_candidate_index"))
        candidate = nms_by_key.get(candidate_key)
        suppressor = None
        if candidate and candidate.get("suppressed_by_pre_nms_index") is not None:
            suppressor = nms_by_key.get((
                record["image_id"], candidate["suppressed_by_pre_nms_index"]))
            if suppressor:
                _draw_box(draw, suppressor.get("box"), COLORS["suppressor"], 3)
        crop = _crop_box(image, [gt_box, query_box, record.get("final_box"),
                                 suppressor.get("box") if suppressor else None])
        image = image.crop(crop)
        panel = _caption_panel(image, [
            f"stage={record.get('failure_stage')} class={record.get('class_name')} "
            f"image={record.get('image_name')} gt={record.get('gt_index')}",
            f"raw rIoU={record.get('rotated_iou'):.3f} final rIoU={record.get('final_rotated_iou'):.3f} "
            f"center={record.get('center_error_px'):.2f}px angle={record.get('angle_error_deg'):.2f}deg",
            "green=GT red=matched query cyan=final yellow=center error magenta=NMS suppressor",
        ])
        output = args.output / f"case_{case_index:03d}_{record['failure_stage']}.png"
        panel.save(output)
        index.append({"output": str(output.resolve()), "record": record,
                      "nms_candidate": candidate, "nms_suppressor": suppressor})
    with (args.output / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(index)} mechanism visualizations to {args.output.resolve()}")


if __name__ == "__main__":
    main()
