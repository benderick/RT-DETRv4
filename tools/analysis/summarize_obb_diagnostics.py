#!/usr/bin/env python3
"""Summarize one or more structured OBB experiment logs.

This tool works directly from the experiment directory.  It never requires a
researcher-maintained spreadsheet or a second ad-hoc inference pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ERROR_FIELDS = (
    "center_error_px", "center_error_gt_diagonal", "angle_error_deg",
    "width_relative_error", "height_relative_error", "corner_chamfer_px",
    "rotated_iou", "final_rotated_iou",
)


def _diagnostics_root(path: Path):
    path = path.expanduser().resolve()
    if (path / "diagnostics").is_dir():
        return path / "diagnostics"
    if path.name == "diagnostics" and path.is_dir():
        return path
    raise FileNotFoundError(f"No diagnostics directory under {path}")


def _epoch_dir(root: Path, epoch: str):
    base = root / "eval"
    if epoch != "latest":
        name = epoch if epoch.startswith("epoch_") else f"epoch_{int(epoch):04d}"
        candidate = base / name
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = sorted(path for path in base.glob("epoch_*") if path.is_dir())
    if not candidates:
        standalone = base / "standalone"
        if standalone.is_dir():
            return standalone
        raise FileNotFoundError(f"No evaluation diagnostics in {base}")
    return candidates[-1]


def _records(directory: Path, stem: str):
    for path in sorted(directory.glob(f"{stem}.rank*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) / len(values) if values else None


def _quantile(values, fraction):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def _summarize_run(run_path: Path, epoch: str, nms_sample_limit=20000):
    root = _diagnostics_root(run_path)
    directory = _epoch_dir(root, epoch)
    metrics_path = directory / "metrics.json"
    evaluator_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) \
        if metrics_path.is_file() else {}
    matches = []
    metrics = defaultdict(list)
    failure_stages = Counter()
    failure_stages_all = Counter()
    evaluation_outcomes = Counter()
    difficult_count = 0
    class_rows = defaultdict(lambda: defaultdict(list))
    areas = []
    for record in _records(directory, "matches"):
        evaluation_outcomes[record.get("evaluation_outcome", "unknown")] += 1
        if record.get("query_index") is None:
            failure_stages_all[record.get("failure_stage", "unknown")] += 1
            if record.get("difficulty"):
                difficult_count += 1
            else:
                failure_stages[record.get("failure_stage", "unknown")] += 1
            continue
        matches.append(record)
        failure_stages_all[record.get("failure_stage", "unknown")] += 1
        if record.get("difficulty"):
            difficult_count += 1
        else:
            failure_stages[record.get("failure_stage", "unknown")] += 1
        areas.append(record.get("area_px2"))
        for field in ERROR_FIELDS:
            value = record.get(field)
            if value is not None:
                metrics[field].append(value)
                class_rows[record.get("class_name", str(record.get("label")))][field].append(value)

    area_q1, area_q2 = _quantile(areas, 1 / 3), _quantile(areas, 2 / 3)
    size_rows = defaultdict(lambda: defaultdict(list))
    boundary_rows = defaultdict(lambda: defaultdict(list))
    for record in matches:
        area = record.get("area_px2")
        if area is None or area_q1 is None:
            size = "unknown"
        elif area <= area_q1:
            size = "small"
        elif area <= area_q2:
            size = "medium"
        else:
            size = "large"
        for field in ERROR_FIELDS:
            value = record.get(field)
            if value is not None:
                size_rows[size][field].append(value)
        boundary = record.get("boundary_distance_object_scale")
        if boundary is None:
            boundary_group = "unknown"
        elif boundary <= 0:
            boundary_group = "crossing"
        elif boundary <= 0.5:
            boundary_group = "near_0_0.5"
        elif boundary <= 2.0:
            boundary_group = "near_0.5_2"
        else:
            boundary_group = "interior_gt_2"
        for field in ERROR_FIELDS:
            value = record.get(field)
            if value is not None:
                boundary_rows[boundary_group][field].append(value)

    rng = random.Random(3407)
    nms_counts = Counter()
    nms_sample = []
    nms_seen = 0
    for record in _records(directory, "nms"):
        nms_counts[record["status"]] += 1
        if record["status"] == "kept":
            continue
        nms_seen += 1
        compact = {
            "status": record["status"], "score": record.get("score"),
            "suppression_iou": record.get("suppression_iou"),
        }
        if len(nms_sample) < nms_sample_limit:
            nms_sample.append(compact)
        else:
            replace = rng.randrange(nms_seen)
            if replace < nms_sample_limit:
                nms_sample[replace] = compact

    queries = list(_records(directory, "queries"))
    layer_values = defaultdict(lambda: defaultdict(list))
    layer_names = {}
    for query in queries:
        if query.get("matched_gt_index") is None:
            continue
        for stage in query.get("stages", []):
            position = int(stage.get("stage_index", -1)) + 1
            layer_names[position] = stage.get("display_name", str(position))
            for field in ("center_error_px", "angle_error_deg", "rotated_iou", "target_class_score"):
                value = stage.get(field)
                if value is not None:
                    layer_values[position][field].append(value)

    training_mechanisms = defaultdict(lambda: {
        "assignment_instability": [], "ocd_modes": Counter(),
        "distribution_entropy": [], "distribution_peak_probability": [],
        "distribution_variance": [],
        "amp_scale": [], "amp_skipped_steps": 0,
        "gradient_l2_norm": [], "gradient_nonfinite_count": [],
    })
    for record in _records(root / "train", "steps"):
        epoch_index = int(record.get("epoch", -1))
        row = training_mechanisms[epoch_index]
        matching = record.get("main_hungarian_matches") or {}
        instability = (matching.get("assignment_instability") or {}).get("instability")
        if instability is not None:
            row["assignment_instability"].append(instability)
        denoising = record.get("denoising") or {}
        if denoising.get("mode"):
            row["ocd_modes"][denoising["mode"]] += 1
        components = (record.get("fine_grained_distributions") or {}).get("components", {})
        for component in components.values():
            for source, target in (
                ("entropy", "distribution_entropy"),
                ("peak_probability", "distribution_peak_probability"),
                ("variance", "distribution_variance"),
            ):
                value = (component.get(source) or {}).get("mean")
                if value is not None:
                    row[target].append(value)
        amp = record.get("amp") or {}
        if amp.get("scale_after") is not None:
            row["amp_scale"].append(amp["scale_after"])
        row["amp_skipped_steps"] += int(bool(amp.get("optimizer_step_skipped")))
        gradients = record.get("gradients_before_clip") or {}
        if gradients.get("total_l2_norm") is not None:
            row["gradient_l2_norm"].append(gradients["total_l2_norm"])
        nonfinite = sum(
            int((gradients.get(group) or {}).get("nonfinite_count") or 0)
            for group in ("backbone", "encoder", "decoder")
        )
        row["gradient_nonfinite_count"].append(nonfinite)

    training_summary = {
        str(epoch_index): {
            "sampled_step_count": max(
                len(row["assignment_instability"]),
                sum(row["ocd_modes"].values())),
            "assignment_instability_mean": _mean(row["assignment_instability"]),
            "ocd_mode_counts": dict(row["ocd_modes"]),
            "distribution_entropy_mean": _mean(row["distribution_entropy"]),
            "distribution_peak_probability_mean": _mean(
                row["distribution_peak_probability"]),
            "distribution_variance_mean": _mean(row["distribution_variance"]),
            "amp_scale_min": min(row["amp_scale"]) if row["amp_scale"] else None,
            "amp_scale_final_sample": row["amp_scale"][-1] if row["amp_scale"] else None,
            "amp_skipped_sampled_steps": row["amp_skipped_steps"],
            "gradient_l2_norm_median": _quantile(row["gradient_l2_norm"], 0.5),
            "gradient_nonfinite_count_max": max(
                row["gradient_nonfinite_count"], default=0),
        }
        for epoch_index, row in sorted(training_mechanisms.items())
    }

    merge_counts = Counter()
    merge_rows = defaultdict(lambda: defaultdict(list))
    for record in _records(directory, "merge_candidates"):
        status = record.get("status", "unknown")
        merge_counts[status] += 1
        if status != "global_nms_overlap":
            mechanism = status
        elif record.get("different_best_gt_suppression"):
            mechanism = "different_gt_collision"
        elif record.get("same_best_gt_suppression"):
            mechanism = "same_gt_duplicate"
        else:
            mechanism = "unmatched_gt_suppression"
        merge_rows[mechanism]["boundary_px"].append(
            record.get("candidate_support_boundary_distance_px"))
        merge_rows[mechanism]["suppression_iou"].append(record.get("suppression_iou"))
        merge_rows[mechanism]["best_gt_iou"].append(record.get("best_source_gt_iou"))
        merge_rows[mechanism]["score"].append(record.get("score"))

    summary = {
        "run": str(run_path.resolve()),
        "epoch_directory": str(directory),
        "evaluator": evaluator_metrics,
        "matched_object_count": len(matches),
        "difficult_object_count": difficult_count,
        "failure_stage_counts": dict(failure_stages),
        "failure_stage_counts_including_difficult": dict(failure_stages_all),
        "evaluation_outcome_counts": dict(evaluation_outcomes),
        "official_tp50_rate": (
            evaluation_outcomes["tp50"] + evaluation_outcomes["tp75"]
        ) / max(sum(value for key, value in evaluation_outcomes.items()
                    if key != "ignored_difficult"), 1),
        "official_tp75_rate": evaluation_outcomes["tp75"] /
            max(sum(value for key, value in evaluation_outcomes.items()
                    if key != "ignored_difficult"), 1),
        "failure_stage_rates": {
            key: value / max(sum(failure_stages.values()), 1)
            for key, value in failure_stages.items()
        },
        "metrics": {
            field: {
                "mean": _mean(values), "median": _quantile(values, 0.5),
                "p90": _quantile(values, 0.9), "p95": _quantile(values, 0.95),
            }
            for field, values in metrics.items()
        },
        "area_tercile_boundaries_px2": [area_q1, area_q2],
        "nms_counts": dict(nms_counts),
        "nms_statistics": {
            status: {
                "mean_score": _mean([row.get("score") for row in nms_sample
                                      if row["status"] == status]),
                "mean_suppression_iou": _mean([
                    row.get("suppression_iou") for row in nms_sample
                    if row["status"] == status]),
            }
            for status in nms_counts
        },
        "refinement_stages": {
            layer_names.get(layer, str(layer)): {
                field: {"mean": _mean(values), "median": _quantile(values, 0.5)}
                         for field, values in fields.items()}
            for layer, fields in sorted(layer_values.items())
        },
        "training_mechanisms_by_epoch": training_summary,
        "tile_merge": {
            "status_counts": dict(merge_counts),
            "mechanisms": {
                mechanism: {
                    "count": max((len(values) for values in fields.values()), default=0),
                    "mean_boundary_px": _mean(fields["boundary_px"]),
                    "median_boundary_px": _quantile(fields["boundary_px"], 0.5),
                    "mean_suppression_iou": _mean(fields["suppression_iou"]),
                    "mean_best_gt_iou": _mean(fields["best_gt_iou"]),
                    "mean_score": _mean(fields["score"]),
                }
                for mechanism, fields in sorted(merge_rows.items())
            },
        },
    }
    return {
        "root": root, "directory": directory, "summary": summary,
        "matches": matches, "metrics": metrics, "class_rows": class_rows,
        "size_rows": size_rows, "nms_sample": nms_sample,
        "boundary_rows": boundary_rows, "layer_values": layer_values,
        "layer_names": layer_names,
        "merge_rows": merge_rows, "training_mechanisms": training_mechanisms,
    }


def _write_group_csv(path, runs, group_key):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run", "group", "count"] + [
            f"{field}_{stat}" for field in ERROR_FIELDS for stat in ("mean", "median", "p90")]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, run in runs:
            for group, fields in sorted(run[group_key].items()):
                count = max((len(values) for values in fields.values()), default=0)
                row = {"run": label, "group": group, "count": count}
                for field in ERROR_FIELDS:
                    values = fields.get(field, [])
                    row[f"{field}_mean"] = _mean(values)
                    row[f"{field}_median"] = _quantile(values, 0.5)
                    row[f"{field}_p90"] = _quantile(values, 0.9)
                writer.writerow(row)


def _write_failure_cases(path, runs, limit=200):
    rows = []
    for label, run in runs:
        for record in run["matches"]:
            severity = (
                (1.0 - float(record.get("final_rotated_iou") or 0.0)) * 4.0 +
                min(float(record.get("center_error_gt_diagonal") or 0.0), 5.0) +
                min(float(record.get("angle_error_deg") or 0.0) / 45.0, 2.0))
            rows.append((severity, label, record))
    rows.sort(key=lambda item: item[0], reverse=True)
    fields = [
        "run", "severity", "failure_stage", "image_id", "image_name", "image_path",
        "gt_index", "class_name", "query_index", "center_error_px",
        "center_error_gt_diagonal", "angle_error_deg", "rotated_iou",
        "final_rotated_iou", "candidate_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for severity, label, record in rows[:limit]:
            writer.writerow({key: label if key == "run" else severity if key == "severity" else record.get(key)
                             for key in fields})


def _write_merge_csv(path, runs):
    fields = [
        "run", "mechanism", "count", "mean_boundary_px", "median_boundary_px",
        "mean_suppression_iou", "mean_best_gt_iou", "mean_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, run in runs:
            mechanisms = run["summary"]["tile_merge"]["mechanisms"]
            for mechanism, statistics in sorted(mechanisms.items()):
                writer.writerow({"run": label, "mechanism": mechanism, **statistics})


def _write_training_mechanism_csv(path, runs):
    fields = [
        "run", "epoch", "assignment_instability_mean", "ocd_mode_counts",
        "distribution_entropy_mean", "distribution_peak_probability_mean",
        "distribution_variance_mean", "amp_scale_min", "amp_scale_final_sample",
        "amp_skipped_sampled_steps", "gradient_l2_norm_median",
        "gradient_nonfinite_count_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, run in runs:
            for epoch, row in run["summary"]["training_mechanisms_by_epoch"].items():
                writer.writerow({"run": label, "epoch": epoch, **{
                    key: json.dumps(value, ensure_ascii=False)
                    if key == "ocd_mode_counts" else value
                    for key, value in row.items() if key in fields
                }})


def _plot(output, runs):
    plot_cache = Path(tempfile.gettempdir()) / f"codrone_plot_cache_{os.getuid()}"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache / "xdg"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_fields = (
        ("center_error_gt_diagonal", "Centre error / GT diagonal"),
        ("angle_error_deg", "Periodic angle error (deg)"),
        ("rotated_iou", "Raw matched-query rotated IoU"),
        ("final_rotated_iou", "Final same-class rotated IoU"),
    )
    for axis, (field, title) in zip(axes.flat, plot_fields):
        for label, run in runs:
            values = run["metrics"].get(field, [])
            if values:
                axis.hist(values, bins=50, density=True, histtype="step", linewidth=1.7, label=label)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("OBB error components (mechanism-level comparison)")
    figure.tight_layout()
    figure.savefig(output / "object_error_components.png", dpi=180)
    plt.close(figure)

    stages = sorted(set().union(*(
        run["summary"]["failure_stage_counts"].keys() for _, run in runs)))
    figure, axis = plt.subplots(figsize=(max(9, len(stages) * 1.1), 4.8))
    width = 0.8 / max(len(runs), 1)
    positions = list(range(len(stages)))
    for run_index, (label, run) in enumerate(runs):
        rates = run["summary"]["failure_stage_rates"]
        axis.bar([position - 0.4 + width / 2 + run_index * width for position in positions],
                 [rates.get(stage, 0.0) for stage in stages], width=width, label=label)
    axis.set_xticks(positions, stages, rotation=25, ha="right")
    axis.set_ylabel("fraction of GT objects")
    axis.set_title("Where each GT is lost in the prediction pipeline")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "failure_stage_decomposition.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for label, run in runs:
        layers = sorted(run["layer_values"])
        if not layers:
            continue
        for axis, field, title in zip(
            axes,
            ("center_error_px", "angle_error_deg", "rotated_iou"),
            ("Centre error (px)", "Angle error (deg)", "Rotated IoU"),
        ):
            axis.plot(layers, [_quantile(run["layer_values"][layer][field], 0.5)
                               for layer in layers], marker="o", label=label)
            axis.set_title(title)
            axis.set_xlabel("refinement stage")
            axis.grid(alpha=0.2)
            axis.set_xticks(
                layers, [run["layer_names"].get(layer, str(layer)) for layer in layers],
                rotation=20, ha="right")
    axes[0].legend()
    figure.suptitle("Matched-query refinement from pre-box through decoder layers")
    figure.tight_layout()
    figure.savefig(output / "decoder_refinement.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, run in runs:
        suppressed = [row for row in run["nms_sample"] if row["status"] == "nms_overlap"]
        overlaps = [row["suppression_iou"] for row in suppressed
                    if row.get("suppression_iou") is not None]
        scores = [row["score"] for row in suppressed if row.get("score") is not None]
        if overlaps:
            axes[0].hist(overlaps, bins=40, density=True, histtype="step",
                         linewidth=1.7, label=label)
        if scores:
            axes[1].hist(scores, bins=40, density=True, histtype="step",
                         linewidth=1.7, label=label)
    axes[0].set_title("Overlap with the NMS suppressor")
    axes[0].set_xlabel("rotated IoU")
    axes[1].set_title("Scores of NMS-deleted candidates")
    axes[1].set_xlabel("score")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("Rotated-NMS mechanism, not final-box appearance")
    figure.tight_layout()
    figure.savefig(output / "rotated_nms_mechanism.png", dpi=180)
    plt.close(figure)

    boundary_order = ("crossing", "near_0_0.5", "near_0.5_2", "interior_gt_2")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for run_index, (label, run) in enumerate(runs):
        for axis, field, title in zip(
            axes,
            ("center_error_gt_diagonal", "angle_error_deg", "rotated_iou"),
            ("Centre / GT diagonal", "Angle error (deg)", "Raw rotated IoU"),
        ):
            axis.plot(range(len(boundary_order)), [
                _mean(run["boundary_rows"].get(group, {}).get(field, []))
                for group in boundary_order
            ], marker="o", label=label)
            axis.set_title(title)
            axis.set_xticks(range(len(boundary_order)), boundary_order, rotation=25, ha="right")
            axis.grid(alpha=0.2)
    axes[0].legend()
    figure.suptitle("OBB error versus signed boundary distance / sqrt(GT area)")
    figure.tight_layout()
    figure.savefig(output / "boundary_error_mechanism.png", dpi=180)
    plt.close(figure)

    if any(run["merge_rows"] for _, run in runs):
        mechanisms = ("same_gt_duplicate", "different_gt_collision", "unmatched_gt_suppression")
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        width = 0.8 / max(len(runs), 1)
        positions = list(range(len(mechanisms)))
        for run_index, (label, run) in enumerate(runs):
            axes[0].bar(
                [position - 0.4 + width / 2 + run_index * width for position in positions],
                [len(run["merge_rows"].get(mechanism, {}).get("suppression_iou", []))
                 for mechanism in mechanisms],
                width=width, label=label)
            for mechanism, linestyle in zip(mechanisms, ("-", "--", ":")):
                boundary = run["merge_rows"].get(mechanism, {}).get("boundary_px", [])
                overlap = run["merge_rows"].get(mechanism, {}).get("suppression_iou", [])
                boundary = [value for value in boundary if value is not None]
                overlap = [value for value in overlap if value is not None]
                if boundary:
                    axes[1].hist(boundary, bins=40, density=True, histtype="step",
                                 linestyle=linestyle, label=f"{label}:{mechanism}")
                if overlap:
                    axes[2].hist(overlap, bins=40, density=True, histtype="step",
                                 linestyle=linestyle, label=f"{label}:{mechanism}")
        axes[0].set_xticks(positions, mechanisms, rotation=20, ha="right")
        axes[0].set_ylabel("global-NMS candidate count")
        axes[1].set_xlabel("signed candidate-to-tile-boundary distance (px)")
        axes[2].set_xlabel("candidate/suppressor rotated IoU")
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend()
        axes[1].legend(fontsize=7)
        figure.suptitle("Global tile-merge NMS: required de-duplication versus different-GT collision")
        figure.tight_layout()
        figure.savefig(output / "tile_merge_nms_mechanism.png", dpi=180)
        plt.close(figure)

    if any(run["training_mechanisms"] for _, run in runs):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8.2))
        specifications = (
            ("assignment_instability", "Assignment instability"),
            ("distribution_entropy", "Distribution entropy"),
            ("distribution_peak_probability", "Peak probability"),
            ("distribution_variance", "Distribution variance"),
            ("gradient_l2_norm", "Gradient L2 norm"),
            ("amp_scale", "AMP GradScaler scale"),
        )
        for label, run in runs:
            epochs = sorted(run["training_mechanisms"])
            for axis, (field, title) in zip(axes.flat, specifications):
                values = [_mean(run["training_mechanisms"][epoch][field]) for epoch in epochs]
                if any(value is not None for value in values):
                    axis.plot(epochs, values, marker="o", label=label)
                axis.set_title(title)
                axis.set_xlabel("epoch")
                axis.grid(alpha=0.2)
                if field == "amp_scale":
                    axis.set_yscale("log")
        axes.flat[0].legend()
        figure.suptitle("O^2 training mechanisms from sampled structured logs")
        figure.tight_layout()
        figure.savefig(output / "o2_training_mechanisms.png", dpi=180)
        plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path, help="experiment or diagnostics directories")
    parser.add_argument("--labels", nargs="*", help="labels in the same order as runs")
    parser.add_argument("--epoch", default="latest", help="latest, an integer, or epoch_XXXX")
    parser.add_argument("--output", type=Path, default=Path("obb_analysis"))
    args = parser.parse_args()
    labels = args.labels or [path.name for path in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--labels must contain one label per run")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [(label, _summarize_run(path, args.epoch))
            for label, path in zip(labels, args.runs)]
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({label: run["summary"] for label, run in runs}, handle,
                  ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    _write_group_csv(args.output / "metrics_by_class.csv", runs, "class_rows")
    _write_group_csv(args.output / "metrics_by_size.csv", runs, "size_rows")
    _write_group_csv(args.output / "metrics_by_boundary.csv", runs, "boundary_rows")
    _write_failure_cases(args.output / "failure_cases.csv", runs)
    _write_merge_csv(args.output / "tile_merge_mechanisms.csv", runs)
    _write_training_mechanism_csv(args.output / "o2_training_mechanisms.csv", runs)
    plotted = _plot(args.output, runs)
    print(f"Wrote OBB analysis to {args.output.resolve()} (plots={plotted})")


if __name__ == "__main__":
    main()
