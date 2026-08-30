"""Generic read-only comparison of two completed OBB training runs.

The functions in this module deliberately consume standard framework logs
instead of loading a checkpoint.  This keeps model selection, AP replay,
paired-object diagnostics, stability, and efficiency checks reproducible from
the evidence emitted by completed runs.  Method-specific schedule audits and
verdicts remain in their isolated research directories.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from engine.evaluation.obb.dota import DotaOBBEvaluator

IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05, dtype=np.float64)


def aligned_hbox_iou(first, second) -> float:
    """Return aligned HBox IoU for two ``cx,cy,w,h,*`` arrays."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != (5,) or second.shape != (5,):
        raise ValueError("aligned_hbox_iou expects two five-value boxes")
    first_wh = np.maximum(first[2:4], 0.0)
    second_wh = np.maximum(second[2:4], 0.0)
    first_min, first_max = first[:2] - first_wh / 2, first[:2] + first_wh / 2
    second_min, second_max = (
        second[:2] - second_wh / 2, second[:2] + second_wh / 2)
    intersection_wh = np.maximum(
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
        0.0,
    )
    intersection = float(np.prod(intersection_wh))
    union = float(np.prod(first_wh) + np.prod(second_wh) - intersection)
    return intersection / union if union > 0 else float("nan")


def read_jsonl_gz(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def load_metric_curve(run_root: str | Path) -> list[dict]:
    run_root = Path(run_root)
    paths = sorted((run_root / "diagnostics" / "eval").glob("epoch_*/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"No epoch metrics under {run_root}")
    curve = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        curve.append({"epoch": int(value["epoch"]), **value["metrics"]})
    expected = list(range(len(curve)))
    observed = [record["epoch"] for record in curve]
    if observed != expected:
        raise ValueError(f"Incomplete/non-contiguous epoch metrics: {observed}")
    return curve


def select_epoch(curve: list[dict], metric: str) -> dict:
    if not curve or any(metric not in record for record in curve):
        raise ValueError(f"Cannot select metric {metric!r} from curve")
    # An earlier epoch wins exact ties, making selection independent of path
    # enumeration and matching conventional first-best checkpoint semantics.
    return max(curve, key=lambda record: (float(record[metric]), -record["epoch"]))


@dataclass
class _LoggedDataset:
    classes: list[str]
    image_ids: list[str]
    ground_truth: list[dict]

    def __len__(self):
        return len(self.image_ids)

    def get_ground_truth(self, image_id: int) -> dict:
        return self.ground_truth[int(image_id)]


def _stream_paths(directory: Path, name: str) -> list[Path]:
    paths = sorted(directory.glob(f"{name}.rank*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"Missing {name}.rank*.jsonl.gz in {directory}")
    return paths


def _read_stream(directory: Path, name: str):
    for path in _stream_paths(directory, name):
        yield from read_jsonl_gz(path)


def replay_threshold_ap(diagnostics: str | Path) -> dict:
    """Replay exact macro AP at every IoU threshold from emitted detections.

    This builds the same generic ``DotaOBBEvaluator`` used during validation.
    It is not an alternative evaluator and it does not touch checkpoints.
    """

    diagnostics = Path(diagnostics)
    context_path = diagnostics / "context.json"
    metrics_path = diagnostics / "metrics.json"
    if not context_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Incomplete diagnostic directory: {diagnostics}")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    classes = list(context.get("classes", []))
    if not classes:
        raise ValueError("Diagnostic context does not declare classes")

    image_records = {int(record["image_id"]): record
                     for record in _read_stream(diagnostics, "images")}
    image_ids = sorted(image_records)
    if image_ids != list(range(len(image_ids))):
        raise ValueError("Logged evaluator replay requires contiguous image ids")
    by_image = {image_id: [] for image_id in image_ids}
    for record in _read_stream(diagnostics, "ground_truth"):
        by_image[int(record["image_id"])].append(record)
    ground_truth = []
    for image_id in image_ids:
        records = sorted(by_image[image_id], key=lambda item: int(item["gt_index"]))
        ground_truth.append({
            "boxes": torch.as_tensor(
                [item["box"] for item in records], dtype=torch.float32).reshape(-1, 5),
            "labels": torch.as_tensor(
                [item["label"] for item in records], dtype=torch.long),
            "difficulty": torch.as_tensor(
                [bool(item.get("difficulty", 0)) for item in records], dtype=torch.bool),
            "ignore_boxes": torch.empty((0, 5), dtype=torch.float32),
        })
    dataset = _LoggedDataset(
        classes=classes,
        image_ids=[str(image_records[index]["image_name"]) for index in image_ids],
        ground_truth=ground_truth,
    )
    detections = defaultdict(list)
    for record in _read_stream(diagnostics, "detections"):
        detections[int(record["image_id"])].append(record)
    predictions = {}
    for image_id in image_ids:
        records = sorted(
            detections[image_id], key=lambda item: int(item["detection_index"]))
        predictions[image_id] = {
            "boxes": torch.as_tensor(
                [item["box"] for item in records], dtype=torch.float32).reshape(-1, 5),
            "scores": torch.as_tensor(
                [item["score"] for item in records], dtype=torch.float32),
            "labels": torch.as_tensor(
                [item["label"] for item in records], dtype=torch.long),
        }
    evaluator = DotaOBBEvaluator(
        dataset, iou_thresholds=IOU_THRESHOLDS, use_07_metric=True)
    evaluator.update(predictions)
    per_class = np.stack([
        evaluator._class_aps(index, IOU_THRESHOLDS)  # exact evaluator primitive
        for index in range(len(classes))
    ])
    with np.errstate(invalid="ignore"):
        macro = np.nanmean(per_class, axis=0)
    official = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    replay_map = float(np.nanmean(macro))
    discrepancy = replay_map - float(official["mAP50_95"])
    if abs(discrepancy) > 2e-4:
        raise RuntimeError(
            "Per-threshold replay disagrees with logged evaluator: "
            f"replay={replay_map:.9f}, logged={official['mAP50_95']:.9f}")
    return {
        "thresholds": IOU_THRESHOLDS.tolist(),
        "ap": macro.tolist(),
        "mAP50_95": replay_map,
        "logged_mAP50_95": float(official["mAP50_95"]),
        "replay_minus_logged": float(discrepancy),
        "class_count": len(classes),
        "image_count": len(image_ids),
        "prediction_count": int(sum(len(value["scores"]) for value in predictions.values())),
    }


def load_matches(diagnostics: str | Path) -> dict[str, dict]:
    diagnostics = Path(diagnostics)
    result = {}
    for record in _read_stream(diagnostics, "matches"):
        key = str(record.get("source_object_uid") or
                  f"{record['image_name']}:{record['gt_index']}")
        if key in result:
            raise ValueError(f"Duplicate matched object identity: {key}")
        result[key] = record
    if not result:
        raise ValueError(f"No match records in {diagnostics}")
    return result


def _correlation(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() == 0 or second.std() == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = .5 * (start + 1 + end)
        start = end
    return ranks


def _field_summary(first, second, *, higher_is_better: bool) -> dict:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    delta = second - first
    improved = delta > 0 if higher_is_better else delta < 0
    return {
        "stable_mean": float(first.mean()),
        "prototype_mean": float(second.mean()),
        "mean_delta": float(delta.mean()),
        "stable_median": float(np.median(first)),
        "prototype_median": float(np.median(second)),
        "median_delta": float(np.median(delta)),
        "improved_fraction": float(improved.mean()),
    }


def compare_matches(
    stable: dict[str, dict],
    prototype: dict[str, dict],
    *,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260829,
) -> dict:
    stable_keys, prototype_keys = set(stable), set(prototype)
    if stable_keys != prototype_keys:
        raise ValueError(
            "Paired diagnostics do not cover identical GT identities: "
            f"stable-only={len(stable_keys - prototype_keys)}, "
            f"prototype-only={len(prototype_keys - stable_keys)}")
    keys = sorted(stable_keys)
    if not all(np.allclose(stable[key]["gt_box"], prototype[key]["gt_box"],
                           rtol=0, atol=0) for key in keys):
        raise ValueError("Paired diagnostics disagree on GT geometry")

    fields = {
        "rotated_iou": True,
        "target_class_score": True,
        "center_error_px": False,
        "center_error_gt_diagonal": False,
        "angle_error_deg": False,
        "width_relative_error": False,
        "height_relative_error": False,
        "corner_chamfer_px": False,
    }
    summaries = {}
    arrays = {}
    for field, higher_is_better in fields.items():
        first = np.asarray([stable[key][field] for key in keys], dtype=np.float64)
        second = np.asarray([prototype[key][field] for key in keys], dtype=np.float64)
        arrays[field] = (first, second)
        summaries[field] = _field_summary(
            first, second, higher_is_better=higher_is_better)

    threshold_counts = []
    stable_riou, prototype_riou = arrays["rotated_iou"]
    for threshold in IOU_THRESHOLDS:
        first = stable_riou >= threshold
        second = prototype_riou >= threshold
        threshold_counts.append({
            "threshold": float(threshold),
            "stable_count": int(first.sum()),
            "prototype_count": int(second.sum()),
            "net_count": int(second.sum() - first.sum()),
            "up_crossings": int((~first & second).sum()),
            "down_crossings": int((first & ~second).sum()),
        })

    score_alignment = {}
    for name, records, riou in (
        ("stable_o2", stable, stable_riou),
        ("prototype", prototype, prototype_riou),
    ):
        scores = np.asarray(
            [records[key]["target_class_score"] for key in keys], dtype=np.float64)
        hbox = np.asarray([
            aligned_hbox_iou(records[key]["query_box"], records[key]["gt_box"])
            for key in keys
        ], dtype=np.float64)
        score_alignment[name] = {
            "score_rotated_iou_pearson": _correlation(scores, riou),
            "score_rotated_iou_spearman": _correlation(
                _average_ranks(scores), _average_ranks(riou)),
            "score_rotated_iou_mae": float(np.abs(scores - riou).mean()),
            "score_hbox_iou_pearson": _correlation(scores, hbox),
            "rotated_minus_hbox_correlation": (
                _correlation(scores, riou) - _correlation(scores, hbox)),
            "score_mean": float(scores.mean()),
            "rotated_iou_mean": float(riou.mean()),
            "hbox_iou_mean": float(hbox.mean()),
            "hbox_rotated_absolute_gap_mean": float(np.abs(hbox - riou).mean()),
        }

    by_video = defaultdict(list)
    delta_riou = prototype_riou - stable_riou
    for index, key in enumerate(keys):
        video_id = str(stable[key].get("video_id") or stable[key]["image_name"])
        by_video[video_id].append(float(delta_riou[index]))
    video_summary = {
        name: {"count": len(values), "mean_delta": float(np.mean(values))}
        for name, values in sorted(by_video.items())
    }
    generator = np.random.default_rng(int(seed))
    video_ids = sorted(by_video)
    bootstrap = np.empty(int(bootstrap_repetitions), dtype=np.float64)
    arrays_by_video = {
        name: np.asarray(by_video[name], dtype=np.float64) for name in video_ids}
    for index in range(len(bootstrap)):
        sampled = generator.choice(video_ids, size=len(video_ids), replace=True)
        bootstrap[index] = np.concatenate(
            [arrays_by_video[name] for name in sampled]).mean()

    calibration = {}
    edges = np.linspace(0.0, 1.0, 21)
    for name, records, riou in (
        ("stable_o2", stable, stable_riou),
        ("prototype", prototype, prototype_riou),
    ):
        scores = np.asarray(
            [records[key]["target_class_score"] for key in keys], dtype=np.float64)
        bins = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = (scores >= lower) & (
                scores <= upper if upper == 1.0 else scores < upper)
            if selected.any():
                bins.append({
                    "lower": float(lower), "upper": float(upper),
                    "count": int(selected.sum()),
                    "score_mean": float(scores[selected].mean()),
                    "rotated_iou_mean": float(riou[selected].mean()),
                })
        calibration[name] = bins

    return {
        "paired_object_count": len(keys),
        "video_count": len(video_ids),
        "field_summaries": summaries,
        "threshold_counts": threshold_counts,
        "score_alignment": score_alignment,
        "score_calibration_bins": calibration,
        "video_clustered_riou_delta": {
            "point": float(delta_riou.mean()),
            "low_95": float(np.quantile(bootstrap, .025)),
            "median": float(np.quantile(bootstrap, .5)),
            "high_95": float(np.quantile(bootstrap, .975)),
            "repetitions": len(bootstrap),
            "cluster_unit": "video_id",
            "per_video": video_summary,
        },
    }


def compare_training_diagnostics(
    stable_root: str | Path,
    prototype_root: str | Path,
) -> dict:
    stable_root, prototype_root = Path(stable_root), Path(prototype_root)
    step_path = Path("diagnostics/train/steps.rank000.jsonl.gz")
    epoch_path = Path("diagnostics/train/epochs.rank000.jsonl.gz")
    stable_steps = read_jsonl_gz(stable_root / step_path)
    prototype_steps = read_jsonl_gz(prototype_root / step_path)
    by_step = {
        "stable_o2": {int(item["global_step"]): item for item in stable_steps},
        "prototype": {int(item["global_step"]): item for item in prototype_steps},
    }
    if set(by_step["stable_o2"]) != set(by_step["prototype"]):
        raise ValueError("Training diagnostic global steps do not align")
    selected_steps = [
        step for step in sorted(by_step["stable_o2"])
        if step >= 100
        and not by_step["stable_o2"][step]["amp"]["optimizer_step_skipped"]
        and not by_step["prototype"][step]["amp"]["optimizer_step_skipped"]
    ]
    if not selected_steps:
        raise ValueError("No paired post-warmup timing records")

    timing_fields = (
        "data_loader_wait", "forward", "criterion", "backward",
        "gradient_inspection_and_clip", "optimizer", "step_total",
        "throughput_images_per_second",
    )
    timings = {}
    for field in timing_fields:
        first = np.asarray([
            by_step["stable_o2"][step]["timing_ms"][field]
            for step in selected_steps
        ], dtype=np.float64)
        second = np.asarray([
            by_step["prototype"][step]["timing_ms"][field]
            for step in selected_steps
        ], dtype=np.float64)
        timings[field] = {
            "stable_median": float(np.median(first)),
            "prototype_median": float(np.median(second)),
            "ratio_of_medians": float(np.median(second) / np.median(first)),
            "median_paired_relative_delta": float(np.median((second - first) / first)),
        }

    memory = {}
    for field in ("allocated_mb", "reserved_mb", "peak_allocated_mb", "peak_reserved_mb"):
        first = np.asarray([
            by_step["stable_o2"][step]["gpu_memory"][field]
            for step in selected_steps
        ], dtype=np.float64)
        second = np.asarray([
            by_step["prototype"][step]["gpu_memory"][field]
            for step in selected_steps
        ], dtype=np.float64)
        memory[field] = {
            "stable_median": float(np.median(first)),
            "prototype_median": float(np.median(second)),
            "ratio_of_medians": float(np.median(second) / np.median(first)),
            "stable_max": float(first.max()),
            "prototype_max": float(second.max()),
            "max_ratio": float(second.max() / first.max()),
        }

    stability = {}
    for name, root, steps in (
        ("stable_o2", stable_root, stable_steps),
        ("prototype", prototype_root, prototype_steps),
    ):
        epochs = read_jsonl_gz(root / epoch_path)
        statistics = [record["statistics"] for record in epochs]
        stability[name] = {
            "epoch_count": len(epochs),
            "logged_step_count": len(steps),
            "amp_skipped_steps": int(sum(
                record.get("amp_skipped_steps", 0) for record in statistics)),
            "minimum_amp_scale": float(min(
                record.get("amp_scale_min", math.inf) for record in statistics)),
            "nonfinite_epoch_losses": int(sum(
                not math.isfinite(float(record["loss"])) for record in statistics)),
            "nonfinite_logged_step_losses": int(sum(
                not math.isfinite(float(record["loss_total"])) for record in steps)),
        }

    expected_schedule = (0.0, 1 / 3, 2 / 3, 1.0)
    observed_schedules, identity_error, quality_record_count = set(), 0.0, 0
    endpoint_role_pairs = set()
    for step in prototype_steps:
        matches = step.get("main_hungarian_matches", {})
        semantics = matches.get("training_semantics", {}).get(
            "localization_quality", {})
        schedule = tuple(semantics.get("decoder_progress", ()))
        observed_schedules.add(schedule)
        for record in matches.get("metric_homotopy", []):
            expected = (
                (1.0 - float(record["progress"])) * float(record["hbox_iou_mean"])
                + float(record["progress"]) * float(record["rotated_iou_mean"])
            )
            identity_error = max(
                identity_error, abs(float(record["homotopy_quality_mean"]) - expected))
            quality_record_count += 1
            if record["loss_suffix"] == "final":
                endpoint_role_pairs.add((record["role"], float(record["progress"])))

    return {
        "paired_timing_record_count": len(selected_steps),
        "timing_ms": timings,
        "gpu_memory": memory,
        "stability": stability,
        "homotopy_audit": {
            "expected_schedule": list(expected_schedule),
            "observed_schedules": [list(value) for value in sorted(observed_schedules)],
            "schedule_exact": observed_schedules == {expected_schedule},
            "quality_record_count": quality_record_count,
            "maximum_mean_interpolation_identity_error": float(identity_error),
            "endpoint_roles": [
                {"role": role, "progress": progress}
                for role, progress in sorted(endpoint_role_pairs)
            ],
        },
    }


def compare_training_runtime(
    stable_root: str | Path,
    prototype_root: str | Path,
) -> dict:
    """Return method-independent timing, memory and stability evidence.

    ``compare_training_diagnostics`` predates this framework-level module and
    additionally preserves the frozen MHR homotopy audit.  New methods should
    use this view and attach their own semantic audit in their owned directory.
    """

    result = compare_training_diagnostics(stable_root, prototype_root)
    result.pop("homotopy_audit", None)
    return result


def load_refinement_stages(run_root: str | Path, epoch: int) -> dict:
    path = (Path(run_root) / "diagnostics" / "eval" /
            f"epoch_{int(epoch):04d}" / "refinement_stages.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_query_trajectories(diagnostics: str | Path) -> dict[str, dict]:
    diagnostics = Path(diagnostics)
    result = {}
    for record in _read_stream(diagnostics, "queries"):
        if record.get("selected_reason") != "hungarian":
            continue
        key = str(record.get("source_object_uid") or
                  f"{record['image_name']}:{record['matched_gt_index']}")
        result[key] = record
    return result


def select_trajectory_case(
    stable: dict[str, dict], prototype: dict[str, dict]
) -> dict:
    common = sorted(set(stable).intersection(prototype))
    if not common:
        raise ValueError("No common logged object trajectories")
    candidates = []
    for key in common:
        first = float(stable[key]["stages"][-1]["rotated_iou"])
        second = float(prototype[key]["stages"][-1]["rotated_iou"])
        candidates.append((second - first, key, first, second))
    crossing = [item for item in candidates if item[2] < .95 <= item[3]]
    selected = max(crossing or candidates)
    return {
        "source_object_uid": selected[1],
        "stable_final_rotated_iou": selected[2],
        "prototype_final_rotated_iou": selected[3],
        "final_delta": selected[0],
        "selection_rule": (
            "maximum final rIoU gain among common logged trajectories that "
            "cross 0.95; fallback to maximum gain if no crossing exists"),
        "stable_o2": stable[selected[1]],
        "prototype": prototype[selected[1]],
    }
