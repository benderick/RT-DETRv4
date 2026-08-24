#!/usr/bin/env python3
"""Visualize the exact cross-tile duplicates removed by global rotated NMS."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "gt": (30, 220, 80),
    "suppressor_gt": (30, 205, 235),
    "suppressed": (255, 145, 30),
    "suppressor": (225, 70, 255),
    "candidate_tile": (255, 190, 50),
    "suppressor_tile": (180, 80, 255),
}


def _diagnostic_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "diagnostics" if (path / "diagnostics").is_dir() else path


def _epoch_directory(root: Path, epoch: str) -> Path:
    base = root / "eval"
    if epoch != "latest":
        return base / (epoch if epoch.startswith("epoch_") else f"epoch_{int(epoch):04d}")
    candidates = sorted(base.glob("epoch_*"))
    return candidates[-1] if candidates else base / "standalone"


def _records(directory: Path):
    for path in sorted(directory.glob("merge_candidates.rank*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _corners(box):
    cx, cy, width, height, angle = box
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        (cx + cosine * x - sine * y, cy + sine * x + cosine * y)
        for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                     (width / 2, height / 2), (-width / 2, height / 2))
    ]


def _draw_box(draw, box, color, width=4):
    if box is None:
        return
    points = _corners(box)
    draw.line(points + [points[0]], fill=color, width=width, joint="curve")


def _draw_tile(draw, record, color):
    origin, size = record.get("tile_origin"), record.get("tile_size")
    if origin is None or size is None:
        return
    x, y = origin
    draw.rectangle((x, y, x + size[0], y + size[1]), outline=color, width=3)


def _priority(record):
    boundary = abs(float(record.get("candidate_support_boundary_distance_px") or 0.0))
    candidate_gt = record.get("best_source_gt_box")
    suppressor_gt = record.get("suppressor_best_source_gt_box")
    separation = 0.0
    if candidate_gt is not None and suppressor_gt is not None:
        center_distance = math.hypot(
            candidate_gt[0] - suppressor_gt[0], candidate_gt[1] - suppressor_gt[1])
        object_scale = max(math.sqrt(candidate_gt[2] * candidate_gt[3]), 1e-7)
        separation = center_distance / object_scale
    return (
        20.0 * bool(record.get("different_best_gt_suppression"))
        + 10.0 * bool(record.get("cross_tile_suppression"))
        + 5.0 * min(separation, 2.0)
        + 3.0 * float(record.get("best_source_gt_iou") or 0.0)
        + 2.0 * float(record.get("suppression_iou") or 0.0)
        + 1.0 / (1.0 + boundary)
    )


def _crop(image, boxes):
    points = [point for box in boxes if box is not None for point in _corners(box)]
    if not points:
        return (0, 0, image.width, image.height)
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 40.0)
    margin = max(120.0, 2.0 * span)
    return (
        max(0, int(min(xs) - margin)),
        max(0, int(min(ys) - margin)),
        min(image.width, int(max(xs) + margin)),
        min(image.height, int(max(ys) + margin)),
    )


def _caption(image, lines):
    if image.width < 720:
        scale = 720.0 / max(image.width, 1)
        image = image.resize((720, max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    height = 104
    panel = Image.new("RGB", (image.width, image.height + height), "white")
    panel.paste(image, (0, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for index, line in enumerate(lines):
        draw.text((8, image.height + 7 + index * 17), line,
                  fill=(20, 20, 20), font=font)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--mode", choices=("collision", "duplicate", "all"), default="collision",
        help="collision=different GTs suppressed; duplicate=same GT copies; all=any NMS")
    parser.add_argument("--output", type=Path, default=Path("tile_merge_mechanisms"))
    args = parser.parse_args()

    directory = _epoch_directory(_diagnostic_root(args.run), args.epoch)
    records = list(_records(directory))
    by_key = {
        (record["source_image_id"], record["global_candidate_index"]): record
        for record in records
    }
    candidates = [
        record for record in records
        if record.get("status") == "global_nms_overlap"
        and record.get("suppressed_by_global_candidate_index") is not None
        and (args.mode == "all"
             or args.mode == "collision" and record.get("different_best_gt_suppression")
             or args.mode == "duplicate" and record.get("same_best_gt_suppression"))
    ]
    candidates.sort(key=_priority, reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)

    index = []
    for case_index, candidate in enumerate(candidates[:max(0, args.count)]):
        suppressor = by_key.get((
            candidate["source_image_id"],
            candidate["suppressed_by_global_candidate_index"],
        ))
        image_path = candidate.get("source_image_path")
        if suppressor is None or not image_path or not Path(image_path).is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        _draw_tile(draw, candidate, COLORS["candidate_tile"])
        _draw_tile(draw, suppressor, COLORS["suppressor_tile"])
        _draw_box(draw, candidate.get("best_source_gt_box"), COLORS["gt"], 5)
        _draw_box(draw, suppressor.get("best_source_gt_box"), COLORS["suppressor_gt"], 5)
        _draw_box(draw, candidate.get("global_box"), COLORS["suppressed"], 4)
        _draw_box(draw, suppressor.get("global_box"), COLORS["suppressor"], 4)
        crop = _crop(image, [
            candidate.get("best_source_gt_box"), candidate.get("global_box"),
            suppressor.get("best_source_gt_box"), suppressor.get("global_box"),
        ])
        panel = _caption(image.crop(crop), [
            f"source={candidate['source_image_id']} class={candidate.get('class_name')}",
            f"candidate object={candidate.get('source_object_uid')} tile={candidate.get('tile_id')}",
            f"suppressor object={candidate.get('suppressor_source_object_uid')} tile={suppressor.get('tile_id')}",
            f"boundary={float(candidate.get('candidate_support_boundary_distance_px') or 0):.1f}px  "
            f"candidate/GT rIoU={float(candidate.get('best_source_gt_iou') or 0):.3f}  "
            f"NMS rIoU={float(candidate.get('suppression_iou') or 0):.3f}",
            f"mechanism={'different-object collision' if candidate.get('different_best_gt_suppression') else 'same-object duplicate'}; "
            "green/cyan=GTs orange=suppressed magenta=retained rectangles=tiles",
        ])
        output = args.output / f"merge_{case_index:03d}_{candidate['source_image_id']}.png"
        panel.save(output)
        index.append({
            "output": str(output.resolve()),
            "candidate": candidate,
            "suppressor": suppressor,
        })

    with (args.output / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(index)} tile-merge mechanism visualizations to {args.output.resolve()}")


if __name__ == "__main__":
    main()
