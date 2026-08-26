#!/usr/bin/env python3
"""Render reproducible, mechanism-near O² evidence from training logs.

The tool never chooses examples by appearance.  It tracks a ground-truth
identity across diagnostic epochs and applies declared numerical selection
rules.  Every rendered case and its selection statistics are written to the
machine-readable evidence index.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PLOT_CACHE = Path(tempfile.gettempdir()) / f"o2_plot_cache_{os.getuid()}"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from engine.rtv4.obb.methods.o2.adr import adr_target_residual
from engine.rtv4.rotated_box_ops import rbox_to_corners
from tools.analysis.summarize_obb_diagnostics import (
    _diagnostics_root,
    _epoch_dir,
    _mean,
    _records,
)


COLORS = {
    "gt": "#00bcd4",
    "pre": "#d4a017",
    "first": "#e67e22",
    "final": "#d62728",
    "reference": "#777777",
    "epsilon": "#8e44ad",
    "eta": "#2ca02c",
}


def _safe_name(value):
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    )


def _stages(record):
    return list(record.get("stages") or ())


def _decoder_stages(record):
    return [stage for stage in _stages(record)
            if stage.get("stage") == "decoder_layer"]


def _box_corners(box):
    tensor = torch.as_tensor(box, dtype=torch.float32).reshape(1, 5)
    return rbox_to_corners(tensor, normalized_angle=False)[0].numpy()


def _draw_box(axis, box, color, label, linewidth=2.0, linestyle="-"):
    corners = _box_corners(box)
    closed = np.concatenate((corners, corners[:1]), axis=0)
    axis.plot(
        closed[:, 0], closed[:, 1], color=color, linewidth=linewidth,
        linestyle=linestyle, label=label,
    )
    return corners


def _crop_limits(boxes, image_size):
    corners = np.concatenate([_box_corners(box) for box in boxes], axis=0)
    low, high = corners.min(axis=0), corners.max(axis=0)
    span = max(float((high - low).max()), 12.0)
    margin = span * 1.25
    center = (low + high) / 2
    width, height = image_size
    x0, x1 = max(0.0, center[0] - margin), min(float(width), center[0] + margin)
    y0, y1 = max(0.0, center[1] - margin), min(float(height), center[1] + margin)
    return x0, x1, y0, y1


def _open_image(record):
    path = record.get("image_path")
    if not path or not Path(path).is_file():
        return None
    return Image.open(path).convert("RGB")


def _fractional_bin(value, codebook):
    return float(np.interp(
        np.clip(value, codebook[0], codebook[-1]),
        codebook,
        np.arange(len(codebook), dtype=float),
    ))


def _target_adr_residual(record, stage):
    anchor = stage.get("initial_anchor_box")
    target = record.get("gt_box")
    if anchor is None or target is None:
        return None
    anchor = torch.as_tensor(anchor, dtype=torch.float64).reshape(1, 5)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(1, 5)
    return adr_target_residual(
        anchor, target, normalized_angle=False)[0].numpy()


def _adr_distribution_figure(record, path):
    layers = [stage for stage in _decoder_stages(record)
              if stage.get("fine_grained_distributions")]
    codebook = np.asarray(record.get("distribution_codebook"), dtype=float)
    if not layers or codebook.ndim != 1 or not len(codebook):
        return False
    first, last = layers[0], layers[-1]
    names = list(first["fine_grained_distributions"])
    if len(names) != 6:
        return False
    target = _target_adr_residual(record, first)
    bins = np.arange(len(codebook))

    figure, axes = plt.subplots(2, 6, figsize=(21, 6.8), squeeze=False)
    for component_index, name in enumerate(names):
        for stage, color, label in (
            (first, COLORS["pre"], first["display_name"]),
            (last, COLORS["final"], last["display_name"]),
        ):
            component = stage["fine_grained_distributions"][name]
            probability = np.asarray(component["probabilities"], dtype=float)
            weighted = np.asarray(
                component.get("weighted_values", probability * codebook), dtype=float)
            axes[0, component_index].plot(
                bins, probability, color=color, linewidth=1.6, label=label)
            axes[1, component_index].plot(
                bins, weighted, color=color, linewidth=1.6,
                label=(f"{label}: Σ={component['expected_residual']:+.3f}"))
        if target is not None:
            target_bin = _fractional_bin(target[component_index], codebook)
            for row in range(2):
                axes[row, component_index].axvline(
                    target_bin, color=COLORS["gt"], linewidth=1.2,
                    linestyle="--", alpha=0.9,
                    label="GT residual" if component_index == 0 and row == 0 else None)
        axes[0, component_index].set_title(name.replace("_", " "))
        axes[0, component_index].set_ylabel("P(n)" if component_index == 0 else "")
        axes[1, component_index].set_ylabel(
            "A(n)P(n)" if component_index == 0 else "")
        axes[1, component_index].set_xlabel("bin index n")
        for row in range(2):
            axes[row, component_index].grid(alpha=0.18)
            axes[row, component_index].set_xlim(0, len(codebook) - 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    weighted_handles, weighted_labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        handles + weighted_handles, labels + weighted_labels,
        loc="upper center", bbox_to_anchor=(.5, .91),
        ncol=5, frameon=False, fontsize=8,
    )
    lqe_first = (first.get("location_quality_estimator") or {}).get(
        "target_class_logit_delta")
    lqe_last = (last.get("location_quality_estimator") or {}).get(
        "target_class_logit_delta")
    lqe_text = ""
    if lqe_first is not None and lqe_last is not None:
        lqe_text = f" | separate LQE Δlogit: {lqe_first:+.3f} → {lqe_last:+.3f}"
    figure.suptitle(
        "ADR probability refinement: unweighted P(n) and fixed-codebook "
        f"weighted A(n)P(n){lqe_text}\n"
        f"{record.get('image_name')} / GT {record.get('matched_gt_index')}",
        y=.99,
    )
    figure.tight_layout(rect=(0, 0, 1, .84))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def _gliding_geometry(box):
    corners = _box_corners(box)
    x, y = corners[:, 0], corners[:, 1]
    x1, x2, y1, y2 = x.min(), x.max(), y.min(), y.max()
    top_candidates = np.flatnonzero(np.isclose(y, y1, rtol=0, atol=1e-5))
    right_candidates = np.flatnonzero(np.isclose(x, x2, rtol=0, atol=1e-5))
    top_index = int(top_candidates[np.argmax(x[top_candidates])]) \
        if len(top_candidates) else int(np.argmin(y))
    right_index = int(right_candidates[np.argmax(y[right_candidates])]) \
        if len(right_candidates) else int(np.argmax(x))
    return {
        "corners": corners,
        "external": np.asarray(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]),
        "epsilon": np.asarray([corners[top_index], [x2, y1]]),
        "eta": np.asarray([corners[right_index], [x2, y2]]),
    }


def _adr_geometry_figure(record, path):
    image = _open_image(record)
    stages = _stages(record)
    layers = _decoder_stages(record)
    if image is None or not stages or not layers or record.get("gt_box") is None:
        return False
    chosen = [stages[0], layers[0]]
    if layers[-1] is not layers[0]:
        chosen.append(layers[-1])
    boxes = [record["gt_box"], *(stage["box"] for stage in chosen)]
    limits = _crop_limits(boxes, image.size)
    figure, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen), 5.2), squeeze=False)
    for axis, stage in zip(axes.flat, chosen):
        axis.imshow(image)
        _draw_box(axis, record["gt_box"], COLORS["gt"], "GT", linewidth=2.5)
        _draw_box(axis, stage["box"], COLORS["final"], "predicted OBB", linewidth=2.2)
        geometry = _gliding_geometry(stage["box"])
        axis.plot(
            geometry["external"][:, 0], geometry["external"][:, 1],
            color=COLORS["pre"], linestyle="--", linewidth=1.8,
            label="external HBox",
        )
        axis.plot(
            geometry["epsilon"][:, 0], geometry["epsilon"][:, 1],
            color=COLORS["epsilon"], linewidth=2.5, marker="o",
            label="epsilon",
        )
        axis.plot(
            geometry["eta"][:, 0], geometry["eta"][:, 1],
            color=COLORS["eta"], linewidth=2.5, marker="o",
            label="eta",
        )
        if stage.get("input_reference_box") is not None:
            _draw_box(
                axis, stage["input_reference_box"], COLORS["reference"],
                "attention input ref", linewidth=1.2, linestyle=":",
            )
        x0, x1, y0, y1 = limits
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal")
        axis.set_title(
            f"{stage['display_name']}\nIoU={stage.get('rotated_iou', float('nan')):.3f}, "
            f"angle err={stage.get('angle_error_deg', float('nan')):.2f}°")
        axis.set_axis_off()
    axes[0, 0].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "O² six-parameter geometry: external HBox + epsilon/eta → OBB\n"
        f"{record.get('image_name')} / GT {record.get('matched_gt_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def _attention_figure(record, path):
    layers = [stage for stage in _decoder_stages(record)
              if stage.get("rotated_cross_attention")]
    if not layers:
        return False
    chosen = [layers[0]] if len(layers) == 1 else [layers[0], layers[-1]]
    figure, axes = plt.subplots(
        1, len(chosen), figsize=(6 * len(chosen), 5.2), squeeze=False)
    for axis, layer in zip(axes.flat, chosen):
        attention = layer["rotated_cross_attention"]
        before = np.asarray(attention["unrotated_offsets"], dtype=float).reshape(-1, 2)
        after = np.asarray(attention["rotated_offsets"], dtype=float).reshape(-1, 2)
        weights = np.asarray(attention["attention_weights"], dtype=float).reshape(-1)
        sizes = 12 + 220 * weights / max(float(weights.max()), 1e-12)
        axis.scatter(
            before[:, 0], before[:, 1], s=sizes, facecolors="none",
            edgecolors=COLORS["reference"], alpha=.75, label="local offsets")
        axis.scatter(
            after[:, 0], after[:, 1], s=sizes, c=weights, cmap="viridis",
            alpha=.85, label="rotated offsets")
        for source, destination in zip(before, after):
            axis.plot(
                (source[0], destination[0]), (source[1], destination[1]),
                color="#bbbbbb", linewidth=.35, alpha=.4)
        axis.scatter([0], [0], marker="x", s=70, color=COLORS["final"])
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(layer["display_name"])
        axis.set_xlabel("normalized x offset")
        axis.set_ylabel("normalized y offset")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Rotation-aware cross-attention: {record.get('image_name')} / "
        f"GT {record.get('matched_gt_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _chamfer_figure(record, path):
    candidates = record.get("hungarian_candidate_costs") or []
    weights = record.get("hungarian_weights") or {}
    if not candidates:
        return False
    terms = [name for name in ("class", "bbox", "angle", "kld", "chamfer")
             if name in weights and weights[name] != 0]
    labels = [f"rank {row['rank']}\nq{row['query_index']}" for row in candidates]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, [row["total"] for row in candidates], color="#3569a8")
    axes[0].set_title("Top candidate total Hungarian cost")
    axes[0].set_ylabel("lower is preferred")
    axes[0].grid(axis="y", alpha=.2)
    x = np.arange(len(candidates))
    width = .8 / max(len(terms), 1)
    for term_index, term in enumerate(terms):
        axes[1].bar(
            x - .4 + width / 2 + term_index * width,
            [row.get(term, 0.) * weights[term] for row in candidates],
            width=width, label=f"{term} × {weights[term]:g}")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Weighted costs for the same GT")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        f"CDC matching: {record.get('image_name')} / GT {record.get('gt_index')} / "
        f"assigned q{record.get('query_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _ocd_instability_figure(root, path):
    rows = defaultdict(lambda: defaultdict(list))
    for record in _records(root / "train", "steps"):
        matching = record.get("main_hungarian_matches") or {}
        value = (matching.get("assignment_instability") or {}).get("instability")
        if value is None:
            continue
        mode = (record.get("denoising") or {}).get("mode") or "unknown"
        rows[mode][int(record.get("epoch", -1))].append(value)
    if not rows:
        return False
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for mode, epochs in sorted(rows.items()):
        x = sorted(epochs)
        axis.plot(x, [_mean(epochs[epoch]) for epoch in x], marker="o", label=mode)
    axis.set_xlabel("epoch")
    axis.set_ylabel("fraction of GT changing matched query")
    axis.set_ylim(bottom=0)
    axis.set_title("OCD assignment instability")
    axis.grid(alpha=.2)
    axis.legend(title="noise mode")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _epoch_number(directory):
    return int(directory.name.split("_")[-1])


def _object_tracks(root):
    tracks = defaultdict(dict)
    for directory in sorted((root / "eval").glob("epoch_*")):
        epoch = _epoch_number(directory)
        for record in _records(directory, "queries"):
            gt_index = record.get("matched_gt_index")
            if gt_index is None or not _stages(record):
                continue
            key = (int(record["image_id"]), int(gt_index))
            tracks[key][epoch] = record
    return tracks


def _track_statistics(key, records):
    ordered = [records[epoch] for epoch in sorted(records)]
    first, latest = ordered[0], ordered[-1]
    first_final, latest_final = _stages(first)[-1], _stages(latest)[-1]
    latest_pre = _stages(latest)[0]
    return {
        "image_id": key[0],
        "gt_index": key[1],
        "epochs": sorted(records),
        "first_final_iou": first_final.get("rotated_iou"),
        "latest_final_iou": latest_final.get("rotated_iou"),
        "learning_gain": (
            latest_final.get("rotated_iou", 0.) - first_final.get("rotated_iou", 0.)),
        "latest_refinement_gain": (
            latest_final.get("rotated_iou", 0.) - latest_pre.get("rotated_iou", 0.)),
        "latest_image_path": latest.get("image_path"),
    }


def _parse_object(value):
    try:
        image_id, gt_index = value.split(":", 1)
        return int(image_id), int(gt_index)
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("object must be IMAGE_ID:GT_INDEX") from error


def _select_tracks(tracks, count, requested):
    eligible = []
    for key, records in tracks.items():
        statistics = _track_statistics(key, records)
        if len(records) >= 2 and statistics["latest_image_path"] and \
                Path(statistics["latest_image_path"]).is_file():
            eligible.append((key, records, statistics))
    if requested:
        selected = []
        lookup = {item[0]: item for item in eligible}
        for key in requested:
            if key not in lookup:
                raise ValueError(
                    f"Requested object {key[0]}:{key[1]} lacks at least two "
                    "detailed epochs or a readable image")
            selected.append(("explicit", *lookup[key]))
        return selected[:count]
    if not eligible or count <= 0:
        return []

    chosen = []
    used = set()

    def choose(rule, key_function, reverse):
        candidates = [item for item in eligible if item[0] not in used]
        if not candidates:
            return
        item = sorted(candidates, key=key_function, reverse=reverse)[0]
        used.add(item[0])
        chosen.append((rule, *item))

    choose("largest_cross_epoch_iou_gain", lambda item: item[2]["learning_gain"], True)
    choose("largest_latest_pre_to_final_iou_gain",
           lambda item: item[2]["latest_refinement_gain"], True)
    choose("largest_over_refinement_or_smallest_gain",
           lambda item: item[2]["latest_refinement_gain"], False)
    remaining = [item for item in eligible if item[0] not in used]
    if remaining and len(chosen) < count:
        median = float(np.median([
            item[2]["latest_final_iou"] for item in remaining]))
        choose("latest_final_iou_nearest_median",
               lambda item: abs(item[2]["latest_final_iou"] - median), False)
    for item in sorted(eligible, key=lambda value: value[0]):
        if len(chosen) >= count:
            break
        if item[0] not in used:
            used.add(item[0])
            chosen.append(("stable_identity_fill", *item))
    return chosen[:count]


def _sample_epochs(records, limit):
    epochs = sorted(records)
    if len(epochs) <= limit:
        return epochs
    indices = np.linspace(0, len(epochs) - 1, limit).round().astype(int)
    return [epochs[index] for index in sorted(set(indices.tolist()))]


def _epoch_progression_figure(records, path, max_epochs):
    epochs = _sample_epochs(records, max_epochs)
    selected = [records[epoch] for epoch in epochs]
    image = _open_image(selected[-1])
    if image is None:
        return False
    boxes = [selected[-1]["gt_box"], *(_stages(record)[-1]["box"] for record in selected)]
    limits = _crop_limits(boxes, image.size)
    figure, axes = plt.subplots(
        1, len(selected), figsize=(4.2 * len(selected), 4.4), squeeze=False)
    for epoch, record, axis in zip(epochs, selected, axes.flat):
        final = _stages(record)[-1]
        axis.imshow(image)
        _draw_box(axis, record["gt_box"], COLORS["gt"], "GT", linewidth=2.5)
        _draw_box(axis, final["box"], COLORS["final"], "prediction", linewidth=2.2)
        x0, x1, y0, y1 = limits
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal")
        axis.set_title(
            f"epoch {epoch + 1}\nIoU={final.get('rotated_iou', float('nan')):.3f}, "
            f"score={final.get('target_class_score', float('nan')):.3f}")
        axis.set_axis_off()
    axes[0, 0].legend(loc="upper right", fontsize=8)
    latest = selected[-1]
    figure.suptitle(
        "Fixed GT across training epochs (query identity may change by Hungarian matching)\n"
        f"{latest.get('image_name')} / GT {latest.get('matched_gt_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def _decoder_progression_figure(record, path):
    image = _open_image(record)
    stages = _stages(record)
    if image is None or not stages or record.get("gt_box") is None:
        return False
    boxes = [record["gt_box"], *(stage["box"] for stage in stages)]
    limits = _crop_limits(boxes, image.size)
    figure, axes = plt.subplots(
        1, len(stages), figsize=(4.1 * len(stages), 4.4), squeeze=False)
    for axis, stage in zip(axes.flat, stages):
        axis.imshow(image)
        _draw_box(axis, record["gt_box"], COLORS["gt"], "GT", linewidth=2.5)
        _draw_box(axis, stage["box"], COLORS["final"], "prediction", linewidth=2.2)
        if stage.get("input_reference_box") is not None:
            _draw_box(
                axis, stage["input_reference_box"], COLORS["reference"],
                "layer input ref", linewidth=1.2, linestyle=":")
        x0, x1, y0, y1 = limits
        axis.set_xlim(x0, x1)
        axis.set_ylim(y1, y0)
        axis.set_aspect("equal")
        delta = (stage.get("transition_from_previous") or {}).get(
            "rotated_iou_delta")
        delta_text = "" if delta is None else f", Δ={delta:+.3f}"
        axis.set_title(
            f"{stage['display_name']}\nIoU={stage.get('rotated_iou', float('nan')):.3f}"
            f"{delta_text}")
        axis.set_axis_off()
    axes[0, 0].legend(loc="upper right", fontsize=7)
    figure.suptitle(
        "Same matched query through pre-box and decoder refinement stages\n"
        f"{record.get('image_name')} / GT {record.get('matched_gt_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def _stage_metric_history(root, path):
    rows = []
    for directory in sorted((root / "eval").glob("epoch_*")):
        metric_path = directory / "refinement_stages.json"
        if not metric_path.is_file():
            continue
        data = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append((_epoch_number(directory), data))
    if not rows:
        return False
    stage_order = rows[-1][1]["stage_order"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), squeeze=False)
    metrics = (
        ("AP50_DOTA07", "DOTA-07 AP50"),
        ("AP75_DOTA07", "DOTA-07 AP75"),
        ("mAP50_75_DOTA07", "mean(AP50, AP75)"),
    )
    for axis, (metric, title) in zip(axes.flat, metrics):
        for stage in stage_order:
            x, y = [], []
            for epoch, data in rows:
                value = data["stages"][stage]["metrics"].get(metric)
                if value is not None:
                    x.append(epoch + 1)
                    y.append(value)
            axis.plot(x, y, marker="o", markersize=3, linewidth=1.4,
                      label=stage.replace("_", " "))
        axis.set_xlabel("epoch")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Full-validation refinement-stage metrics across training")
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-epochs-per-object", type=int, default=6)
    parser.add_argument("--object", action="append", type=_parse_object, default=[])
    parser.add_argument("--output", type=Path, default=Path("o2_mechanisms"))
    args = parser.parse_args()
    root = _diagnostics_root(args.run)
    latest_directory = _epoch_dir(root, args.epoch)
    args.output.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": "o2-paper-evidence-v1",
        "run": str(args.run.resolve()),
        "selection_policy": {
            "appearance_used": False,
            "identity": "(image_id, gt_index)",
            "rules_in_order": [
                "largest_cross_epoch_iou_gain",
                "largest_latest_pre_to_final_iou_gain",
                "largest_over_refinement_or_smallest_gain",
                "latest_final_iou_nearest_median",
            ],
        },
        "semantics": {
            "unweighted": "softmax probability P(n)",
            "weighted": "fixed analytic codebook product A(n)P(n)",
            "integral": "sum_n A(n)P(n)",
            "lqe": "separate MLP correction of classification logits from top-k distribution statistics",
        },
        "selected_objects": [],
        "aggregate": {},
        "chamfer": [],
        "ocd": [],
    }

    tracks = _object_tracks(root)
    selected = _select_tracks(tracks, max(args.count, 0), args.object)
    for rule, key, records, statistics in selected:
        latest_epoch = max(records)
        latest = records[latest_epoch]
        stem = f"image{key[0]}_gt{key[1]}"
        files = {}
        for kind, renderer, suffix in (
            ("epoch_progression",
             lambda record, output: _epoch_progression_figure(
                 records, output, max(args.max_epochs_per_object, 2)),
             "epoch_progression"),
            ("decoder_progression", _decoder_progression_figure, "decoder_progression"),
            ("adr_geometry", _adr_geometry_figure, "adr_geometry"),
            ("adr_distributions", _adr_distribution_figure, "adr_distributions"),
            ("rotated_cross_attention", _attention_figure, "rotated_attention"),
        ):
            output = args.output / f"{suffix}_{stem}.png"
            if renderer(latest, output):
                files[kind] = str(output.resolve())
                pdf = output.with_suffix(".pdf")
                if pdf.is_file():
                    files[f"{kind}_pdf"] = str(pdf.resolve())
        index["selected_objects"].append({
            "selection_rule": rule,
            **statistics,
            "latest_query_index": latest.get("query_index"),
            "files": files,
        })

    match_records = [record for record in _records(latest_directory, "matches")
                     if record.get("hungarian_candidate_costs")]
    for record in match_records[:max(args.count, 0)]:
        output = args.output / (
            f"chamfer_{_safe_name(record.get('image_name'))}_gt{record.get('gt_index')}.png")
        if _chamfer_figure(record, output):
            index["chamfer"].append(str(output.resolve()))

    instability_path = args.output / "ocd_assignment_instability.png"
    if _ocd_instability_figure(root, instability_path):
        index["ocd"].append(str(instability_path.resolve()))
    stage_path = args.output / "refinement_stage_metrics_across_epochs.png"
    if _stage_metric_history(root, stage_path):
        index["aggregate"]["refinement_stage_metrics"] = str(stage_path.resolve())
        index["aggregate"]["refinement_stage_metrics_pdf"] = str(
            stage_path.with_suffix(".pdf").resolve())

    with (args.output / "evidence_index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote O² paper evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
