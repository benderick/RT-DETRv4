#!/usr/bin/env python3
"""Create paired, object-aligned mechanism comparisons for two OBB runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from visualize_obb_mechanisms import (
    COLORS, _crop_box, _draw_axis, _draw_box, _epoch, _records, _root,
)


def _key(record):
    return record.get("image_name"), record.get("gt_index")


def _number(record, field, default=0.0):
    value = record.get(field)
    return float(value) if value is not None else default


def _improvement(baseline, method, criterion):
    center = _number(baseline, "center_error_gt_diagonal") - \
        _number(method, "center_error_gt_diagonal")
    angle = (_number(baseline, "angle_error_deg") -
             _number(method, "angle_error_deg")) / 45.0
    raw_iou = _number(method, "rotated_iou") - _number(baseline, "rotated_iou")
    final_iou = _number(method, "final_rotated_iou") - _number(baseline, "final_rotated_iou")
    if criterion == "center":
        return center
    if criterion == "angle":
        return angle
    if criterion == "riou":
        return raw_iou
    return center + angle + 2.0 * raw_iou + final_iou


def _render(image_path, record, crop, label):
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
    image = image.crop(crop)
    caption_height = 76
    panel = Image.new("RGB", (image.width, image.height + caption_height), "white")
    panel.paste(image, (0, 0))
    caption = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    lines = [
        f"{label}: stage={record.get('failure_stage')}",
        f"center={_number(record, 'center_error_px'):.2f}px "
        f"({ _number(record, 'center_error_gt_diagonal'):.3f} GT diag), "
        f"angle={_number(record, 'angle_error_deg'):.2f}deg",
        f"raw rIoU={_number(record, 'rotated_iou'):.3f}, "
        f"final rIoU={_number(record, 'final_rotated_iou'):.3f}",
    ]
    for index, line in enumerate(lines):
        caption.text((8, image.height + 7 + 17 * index), line, fill=(20, 20, 20), font=font)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("method", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--method-label", default="method")
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--baseline-epoch")
    parser.add_argument("--method-epoch")
    parser.add_argument("--criterion", choices=("combined", "center", "angle", "riou"),
                        default="combined")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("obb_paired_mechanisms"))
    args = parser.parse_args()

    baseline_dir = _epoch(_root(args.baseline), args.baseline_epoch or args.epoch)
    method_dir = _epoch(_root(args.method), args.method_epoch or args.epoch)
    baseline = {_key(record): record for record in _records(baseline_dir, "matches")
                if record.get("query_index") is not None}
    method = {_key(record): record for record in _records(method_dir, "matches")
              if record.get("query_index") is not None}
    pairs = [(baseline[key], method[key]) for key in baseline.keys() & method.keys()]
    pairs.sort(key=lambda pair: _improvement(pair[0], pair[1], args.criterion), reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)

    index = []
    for pair_index, (base_record, method_record) in enumerate(pairs[:args.count]):
        image_path = method_record.get("image_path") or base_record.get("image_path")
        if not image_path or not Path(image_path).is_file():
            continue
        original = Image.open(image_path)
        crop = _crop_box(original, [
            base_record.get("gt_box"), base_record.get("query_box"),
            base_record.get("final_box"), method_record.get("query_box"),
            method_record.get("final_box"),
        ])
        left = _render(image_path, base_record, crop, args.baseline_label)
        right = _render(image_path, method_record, crop, args.method_label)
        gap, header_height = 12, 42
        canvas = Image.new("RGB", (
            left.width + right.width + gap,
            max(left.height, right.height) + header_height), "white")
        canvas.paste(left, (0, header_height))
        canvas.paste(right, (left.width + gap, header_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8),
                  f"{base_record.get('image_name')} / GT {base_record.get('gt_index')} / "
                  f"{base_record.get('class_name')} / improvement={_improvement(base_record, method_record, args.criterion):.3f}",
                  fill=(20, 20, 20), font=ImageFont.load_default())
        draw.text((8, 24), "green=GT red=matched query cyan=final yellow=centre discrepancy",
                  fill=(20, 20, 20), font=ImageFont.load_default())
        output = args.output / f"paired_{pair_index:03d}_{base_record.get('image_name')}.png"
        canvas.save(output)
        index.append({
            "output": str(output.resolve()),
            "improvement": _improvement(base_record, method_record, args.criterion),
            "baseline": base_record, "method": method_record,
        })
    with (args.output / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(index)} paired mechanism comparisons to {args.output.resolve()}")


if __name__ == "__main__":
    main()
