#!/usr/bin/env python3
"""Run tiled D-FINE OBB inference and merge predictions on source images."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
import math
import numbers
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.evaluation.obb import MergedDotaOBBEvaluator  # noqa: E402
from engine.rtv4.obb_visualization import save_obb_visualization  # noqa: E402


def _checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    if "ema" in state:
        state = state["ema"].get("module", state["ema"])
    elif "model" in state:
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return _jsonable(value.item() if value.ndim == 0 else value.tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl_gz(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(
                _jsonable(record), ensure_ascii=False,
                separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _memory(device):
    if device.type != "cuda":
        return {
            "allocated_mb": 0.0, "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0,
        }
    divisor = 1024.0 ** 2
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / divisor,
        "reserved_mb": torch.cuda.memory_reserved(device) / divisor,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / divisor,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / divisor,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tile-root", type=Path,
        help="Override the config validation tile root, e.g. standard_patches/test")
    parser.add_argument("--output", type=Path, default=Path("obb_tile_predictions"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--visualization-score-threshold", type=float, default=0.3)
    parser.add_argument("--visualize-count", type=int, default=12)
    parser.add_argument("--mechanism-count", type=int, default=12)
    parser.add_argument(
        "--compact-merge-log", action="store_true",
        help=("Keep final-box tile provenance and merge totals, but omit the "
              "large per-candidate NMS trace"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; pass --overwrite explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    val_override = {}
    if args.tile_root is not None:
        val_override.setdefault("dataset", {})["root"] = str(args.tile_root)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        val_override["total_batch_size"] = args.batch_size
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("--workers must be non-negative")
        val_override["num_workers"] = args.workers
    config = YAMLConfig(
        args.config,
        HGNetv2={"pretrained": False},
        **({"val_dataloader": val_override} if val_override else {}),
    )
    device = torch.device(args.device)
    model = config.model.to(device).eval()
    missing, unexpected = model.load_state_dict(
        _checkpoint_state(args.checkpoint), strict=False)
    if missing or unexpected:
        print(f"checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    postprocessor = config.postprocessor.to(device).eval()
    if args.score_threshold is not None:
        postprocessor.score_threshold = args.score_threshold
    data_loader = config.val_dataloader
    evaluator = config.evaluator
    if not isinstance(evaluator, MergedDotaOBBEvaluator):
        raise TypeError("Tile inference config must use MergedDotaOBBEvaluator")
    if args.compact_merge_log:
        if args.mechanism_count > 0:
            raise ValueError("--compact-merge-log requires --mechanism-count 0")
        evaluator.record_merge_candidates = False

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    loader_wait_started = wall_started
    batch_records = []
    forward_seconds = postprocess_seconds = data_wait_seconds = 0.0
    tile_count = 0
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(data_loader):
            yielded = time.perf_counter()
            wait_seconds = yielded - loader_wait_started
            data_wait_seconds += wait_seconds
            transfer_started = _synchronize(device)
            images = images.to(device)
            device_targets = [
                {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in target.items()}
                for target in targets
            ]
            forward_started = _synchronize(device)
            outputs = model(images)
            forward_finished = _synchronize(device)
            results = postprocessor(outputs, device_targets)
            postprocess_finished = _synchronize(device)
            forward_seconds += forward_finished - forward_started
            postprocess_seconds += postprocess_finished - forward_finished
            tile_count += len(images)
            evaluator.update({
                int(target["image_id"].reshape(-1)[0]): result
                for target, result in zip(device_targets, results)
            })
            batch_records.append({
                "batch_index": batch_index,
                "batch_size": len(images),
                "tile_ids": [target.get("tile_id") for target in targets],
                "source_image_ids": [target.get("source_image_id") for target in targets],
                "data_loader_wait_ms": wait_seconds * 1000.0,
                "host_to_device_ms": (forward_started - transfer_started) * 1000.0,
                "forward_ms": (forward_finished - forward_started) * 1000.0,
                "local_postprocess_nms_ms": (
                    postprocess_finished - forward_finished) * 1000.0,
                "gpu_memory": _memory(device),
            })
            print(f"batch {batch_index + 1}/{len(data_loader)}: {tile_count} tiles")
            loader_wait_started = time.perf_counter()

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    wall_seconds = time.perf_counter() - wall_started

    diagnostic_dir = output / "diagnostics" / "eval" / "standalone"
    _write_jsonl_gz(
        diagnostic_dir / "merge_candidates.rank000.jsonl.gz",
        evaluator.merge_candidate_records)
    _write_jsonl_gz(
        diagnostic_dir / "merge_images.rank000.jsonl.gz",
        evaluator.merge_image_records)
    _write_jsonl_gz(
        diagnostic_dir / "performance_batches.rank000.jsonl.gz", batch_records)
    _write_json(diagnostic_dir / "merge_summary.json", evaluator.merge_summary)
    _write_json(diagnostic_dir / "metrics.json", {
        "metrics": evaluator.metrics,
        "per_class": evaluator.per_class,
        "per_class_metrics": evaluator.per_class_metrics,
        "stats": evaluator.stats,
        "merge_summary": evaluator.merge_summary,
    })

    final_records = []
    for source_index, prediction in sorted(evaluator.predictions.items()):
        final_records.append({
            "source_image_index": source_index,
            "source_image_id": evaluator.dataset.image_ids[source_index],
            **prediction,
            "provenance": evaluator.final_prediction_metadata.get(source_index, {}),
        })
    _write_jsonl_gz(output / "predictions.jsonl.gz", final_records)
    evaluator.export_dota_results(output / "dota")

    ranked_images = sorted(
        evaluator.merge_image_records,
        key=lambda record: (
            record["global_nms_suppressed_count"], record["candidate_count"]),
        reverse=True,
    )
    for record in ranked_images[:max(0, args.visualize_count)]:
        source_index = record["source_image_index"]
        prediction = evaluator.predictions[source_index]
        image, _ = evaluator.dataset.load_item(source_index)
        save_obb_visualization(
            output / "visualizations" / f'{record["source_image_id"]}.jpg',
            image,
            prediction["boxes"], prediction["labels"], prediction["scores"],
            evaluator.dataset.classes,
            score_threshold=args.visualization_score_threshold,
        )

    performance = {
        "tiles": tile_count,
        "source_images": len(evaluator.dataset),
        "wall_seconds": wall_seconds,
        "tiles_per_second_end_to_end": tile_count / max(wall_seconds, 1e-9),
        "source_images_per_second_end_to_end": (
            len(evaluator.dataset) / max(wall_seconds, 1e-9)),
        "mean_wall_ms_per_source_image": (
            wall_seconds * 1000.0 / max(len(evaluator.dataset), 1)),
        "forward_seconds": forward_seconds,
        "local_postprocess_seconds": postprocess_seconds,
        "data_loader_wait_seconds": data_wait_seconds,
        "global_merge_seconds": evaluator.merge_summary.get("merge_total_ms", 0.0) / 1000.0,
        "gpu_memory": _memory(device),
    }
    _write_json(output / "performance.json", performance)
    _write_json(output / "manifest.json", {
        "command": sys.argv,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tile_root": str(data_loader.dataset.root),
        "source_root": str(evaluator.dataset.root),
        "inference_partition": {
            "storage": "materialized_tiles",
        },
        "tiling_manifest": evaluator.tiling_manifest,
        "local_score_threshold": postprocessor.score_threshold,
        "local_nms_iou_threshold": postprocessor.nms_iou_threshold,
        "global_nms_iou_threshold": evaluator.merge_iou_threshold,
        "merge_candidate_trace_recorded": evaluator.record_merge_candidates,
        "data_loader": {
            "batch_size": data_loader.batch_size,
            "num_workers": data_loader.num_workers,
            "prefetch_factor": (
                data_loader.prefetch_factor if data_loader.num_workers else None),
            "pin_memory": data_loader.pin_memory,
            "persistent_workers": data_loader.persistent_workers,
        },
        "performance": performance,
    })

    if args.mechanism_count > 0:
        subprocess.run([
            sys.executable,
            str(REPO_ROOT / "tools/analysis/visualize_tile_merge_mechanisms.py"),
            str(output), "--count", str(args.mechanism_count),
            "--output", str(output / "merge_mechanisms"),
        ], check=True)
    print(f"Wrote merged original-image inference to {output}")


if __name__ == "__main__":
    main()
