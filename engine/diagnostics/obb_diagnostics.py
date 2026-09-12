"""Machine-readable diagnostics for oriented-object experiments.

The evaluator metric is intentionally not the only experiment artifact.  This
module preserves the intermediate evidence needed to explain a result:
Hungarian assignments, geometry error components, decoder refinement, score
filtering, rotated-NMS decisions, timing, memory, and run provenance.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..misc import dist_utils
from ..rtv4.rotated_box_ops import angle_distance, rbox_to_corners, rotated_iou


SCHEMA_VERSION = "obb-diagnostics-v5"
NMS_STATUS = {0: "kept", 1: "nms_overlap", 2: "max_detections"}


def _finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _jsonable(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.ndim == 0:
            return _finite_float(value) if value.is_floating_point() else int(value)
        return _jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return _finite_float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _append_jsonl_gzip_atomic(path: Path, record):
    """Atomically append one complete gzip member to a JSONL stream.

    Training records are sparse but need to be readable while a run is still
    active.  Keeping a normal ``gzip.open(..., "at")`` handle alive buffers the
    header and payload and leaves the file incomplete until the entire solver
    exits.  Building a complete member and replacing the stream atomically
    keeps every previously published record valid even if the process is
    interrupted while preparing the next one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        _jsonable(record), ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    member = gzip.compress(line.encode("utf-8"), mtime=0)
    temporary = path.with_name(f".{path.name}.rankwriter-{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as destination:
            if path.is_file():
                with path.open("rb") as source:
                    shutil.copyfileobj(source, destination)
            destination.write(member)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        # A failed write must never replace the last complete stream.  The
        # temporary may remain only when replacement itself failed.
        temporary.unlink(missing_ok=True)


class _JsonlWriter:
    def __init__(self, path: Path, append=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = gzip.open(path, "at" if append else "wt", encoding="utf-8")

    def write(self, record):
        self.handle.write(json.dumps(
            _jsonable(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _git_metadata(repository: Path):
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=repository, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _dataset_image_metadata(dataset, image_id: int):
    """Read optional adapter-owned acquisition metadata.

    Filename conventions belong to dataset adapters.  Generic diagnostics
    preserve whatever JSON-compatible fields an adapter provides without
    learning CODrone, DOTA, or future dataset naming schemes.
    """

    provider = getattr(dataset, "get_image_metadata", None)
    return dict(provider(image_id)) if provider is not None else {}


def _restore_target_boxes(target):
    boxes = target["boxes"].detach().clone()
    if not len(boxes):
        return boxes
    size = target.get("box_normalization_size", target["size"]).to(device=boxes.device, dtype=boxes.dtype)
    scale = target.get("scale_factor", torch.ones(2, device=boxes.device)).to(
        device=boxes.device, dtype=boxes.dtype)
    padding = target.get("padding", torch.zeros(4, device=boxes.device)).to(
        device=boxes.device, dtype=boxes.dtype)
    boxes[:, 0] = (boxes[:, 0] * size[0] - padding[0]) / scale[0]
    boxes[:, 1] = (boxes[:, 1] * size[1] - padding[1]) / scale[1]
    length_scale = scale.mean()
    boxes[:, 2] = boxes[:, 2] * size[0] / length_scale
    boxes[:, 3] = boxes[:, 3] * size[1] / length_scale
    boxes[:, 4] *= math.pi
    return boxes


def _aligned_box_errors(predictions, targets):
    """Vectorized error decomposition for aligned original-pixel OBBs."""
    if not len(predictions):
        return {name: predictions.new_empty(0) for name in (
            "center_error_px", "center_error_gt_diagonal", "angle_error_deg",
            "width_relative_error", "height_relative_error", "log_area_ratio",
            "corner_chamfer_px", "rotated_iou")}
    center_error = torch.linalg.vector_norm(predictions[:, :2] - targets[:, :2], dim=1)
    target_diagonal = torch.linalg.vector_norm(targets[:, 2:4], dim=1).clamp_min(1e-7)
    pred_corners = rbox_to_corners(predictions, normalized_angle=False)
    target_corners = rbox_to_corners(targets, normalized_angle=False)
    corner_distances = torch.linalg.vector_norm(
        pred_corners[:, :, None, :] - target_corners[:, None, :, :], dim=-1)
    chamfer = corner_distances.min(dim=2).values.mean(dim=1) + \
        corner_distances.min(dim=1).values.mean(dim=1)
    return {
        "center_error_px": center_error,
        "center_error_gt_diagonal": center_error / target_diagonal,
        "angle_error_deg": angle_distance(
            predictions[:, 4], targets[:, 4], normalized=False) * 180.0 / math.pi,
        "width_relative_error": torch.abs(predictions[:, 2] - targets[:, 2]) /
            targets[:, 2].clamp_min(1e-7),
        "height_relative_error": torch.abs(predictions[:, 3] - targets[:, 3]) /
            targets[:, 3].clamp_min(1e-7),
        "log_area_ratio": torch.log(
            (predictions[:, 2] * predictions[:, 3]).clamp_min(1e-7) /
            (targets[:, 2] * targets[:, 3]).clamp_min(1e-7)),
        "corner_chamfer_px": chamfer,
        "rotated_iou": rotated_iou(
            predictions, targets, aligned=True, model_space=False).clamp(0, 1),
    }


def _box_errors(prediction, target):
    return {name: _finite_float(values[0]) for name, values in
            _aligned_box_errors(prediction[None], target[None]).items()}


def _cpu_dict(value):
    return {
        key: item.detach().cpu() if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def _target_metadata(target, key, default=None, instance_index=None, instance_count=None):
    value = target.get(key, default)
    if instance_index is not None and torch.is_tensor(value) and value.ndim > 0 \
            and instance_count is not None and value.shape[0] == instance_count:
        return value[instance_index]
    return value


def _memory_snapshot(device):
    if device.type != "cuda" or not torch.cuda.is_available():
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


def _model_structure(model):
    """Describe model capacity without serializing parameter values.

    A full name/shape inventory makes later method comparisons possible from
    the run artifacts alone.  In particular, a refinement head can be compared
    with a stable baseline directly from the recorded runs.
    """
    if model is None:
        return None
    model = dist_utils.de_parallel(model)
    parameters = []
    prefix_counts = {}
    total = trainable = parameter_bytes = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        trainable += count if parameter.requires_grad else 0
        parameter_bytes += count * parameter.element_size()
        parameters.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": count,
            "trainable": bool(parameter.requires_grad),
            "dtype": str(parameter.dtype),
        })
        parts = name.split(".")
        # Several prefix depths preserve both subsystem totals and individual
        # refinement-head totals while remaining architecture agnostic.
        for depth in range(1, min(4, len(parts)) + 1):
            prefix = ".".join(parts[:depth])
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + count

    buffers = []
    buffer_total = buffer_bytes = 0
    for name, buffer in model.named_buffers():
        count = int(buffer.numel())
        buffer_total += count
        buffer_bytes += count * buffer.element_size()
        buffers.append({
            "name": name,
            "shape": list(buffer.shape),
            "numel": count,
            "dtype": str(buffer.dtype),
        })
    return {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "nontrainable_parameter_count": total - trainable,
        "parameter_bytes": parameter_bytes,
        "buffer_count": buffer_total,
        "buffer_bytes": buffer_bytes,
        "parameter_count_by_prefix": dict(sorted(prefix_counts.items())),
        "parameters": parameters,
        "buffers": buffers,
    }


class OBBDiagnostics:
    """Own structured logs for one solver run.

    Each distributed rank writes separate gzip JSONL files.  This avoids file
    races and preserves every validation sample; analysis tools glob all ranks.
    """

    def __init__(self, cfg, output_dir, model=None, train_dataset=None):
        self.enabled = bool(getattr(cfg, "diagnostics_enabled", False))
        self.train_interval = max(1, int(getattr(cfg, "diagnostics_train_interval", 50)))
        self.detailed_image_limit = max(0, int(getattr(cfg, "diagnostics_detailed_image_limit", 32)))
        self.detailed_epoch_interval = max(
            1, int(getattr(cfg, "diagnostics_detailed_epoch_interval", 1)))
        self.layerwise_epoch_interval = max(
            0, int(getattr(cfg, "diagnostics_layerwise_epoch_interval", 0)))
        self.total_epochs = int(getattr(cfg, "epoches", 0))
        self.query_topk = max(0, int(getattr(cfg, "diagnostics_query_topk", 50)))
        self.rank = dist_utils.get_rank()
        self.world_size = dist_utils.get_world_size()
        self.root = Path(output_dir) / "diagnostics"
        self._writers = {}
        self._eval_dir = None
        self._eval_epoch = None
        self._eval_batch_index = 0
        self._detailed_seen = 0
        self._detailed_eval_enabled = True
        self._layerwise_eval_enabled = False
        self._counts = {}
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        # A from-scratch run reusing an output directory must not inherit old
        # step records.  Resume runs deliberately append their new segment.
        if not getattr(cfg, "resume", None):
            for stem in ("steps", "epochs"):
                stale = self.root / "train" / f"{stem}.rank{self.rank:03d}.jsonl.gz"
                if stale.is_file():
                    stale.unlink()
        if dist_utils.is_main_process():
            repository = Path(__file__).resolve().parents[2]
            cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            cuda_name = torch.cuda.get_device_name(cuda_device) if cuda_device is not None else None
            config = getattr(cfg, "yaml_cfg", {})
            config_bytes = json.dumps(
                _jsonable(config), sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            decoder_config = {}
            if isinstance(config, dict):
                decoder_config = config.get("RotatedDFINETransformer", {})
            refinement_mode = decoder_config.get("refinement_mode")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "run_id": Path(output_dir).resolve().name,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "refinement_mode": refinement_mode,
                "seed": getattr(cfg, "seed", None),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv,
                "working_directory": os.getcwd(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "pytorch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": cuda_name,
                "world_size": self.world_size,
                "git": _git_metadata(repository),
                "config": config,
                "train_dataset_provenance": (
                    train_dataset.get_dataset_provenance()
                    if callable(getattr(train_dataset, "get_dataset_provenance", None)) else None),
                "coordinate_convention": {
                    "model": ("[cx/S, cy/S, w/S, h/S, theta/pi], S=max(W,H)" if decoder_config.get("box_coordinate_mode") == "isotropic"
                              else "[cx/W, cy/H, w/W, h/H, theta/pi], theta in [0,1), long-edge w>=h"),
                    "diagnostics": "[cx, cy, w, h, theta] in original-image pixels/radians, theta in [0,pi)",
                    "angle_error": "shortest half-turn-periodic unsigned distance in degrees",
                },
                "codebooks": {
                    "aug_flip_code": {"0": "none", "1": "horizontal",
                                      "2": "vertical", "3": "diagonal"},
                    "aug_photometric_order_code":
                        "least-significant to most-significant digit is execution order; "
                        "1=brightness, 2=contrast, 3=saturation, 4=hue; -1=not applied",
                    "nms_status": NMS_STATUS,
                },
                "files": {
                    "model_structure": "model_structure.json",
                    "train_steps": "train/steps.rankNNN.jsonl.gz",
                    "eval": "eval/epoch_XXXX/{images,ground_truth,detections,matches,queries,nms}.rankNNN.jsonl.gz",
                    "refinement_stage_metrics": "eval/epoch_XXXX/refinement_stages.json",
                },
            }
            _write_json(self.root / "manifest.json", manifest)
            if model is not None:
                _write_json(
                    self.root / "model_structure.json", _model_structure(model)
                )

    def should_log_train(self, global_step):
        return self.enabled and int(global_step) % self.train_interval == 0

    def needs_detailed_eval_layers(self):
        return self.enabled and self._detailed_eval_enabled and \
            self._detailed_seen < self.detailed_image_limit

    def needs_layerwise_eval(self):
        """Whether this epoch requires full-validation refinement metrics."""

        return self.enabled and self._layerwise_eval_enabled

    def _writer(self, name, directory=None, append=True):
        directory = self.root if directory is None else Path(directory)
        key = (str(directory), name)
        if key not in self._writers:
            self._writers[key] = _JsonlWriter(
                directory / f"{name}.rank{self.rank:03d}.jsonl.gz", append=append)
        return self._writers[key]

    def _write_durable_train_record(self, name, record):
        path = self.root / "train" / f"{name}.rank{self.rank:03d}.jsonl.gz"
        _append_jsonl_gzip_atomic(path, {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            **record,
        })

    def record_train_step(self, record):
        if not self.enabled:
            return
        self._write_durable_train_record("steps", record)

    def record_train_epoch(self, epoch, statistics):
        if not self.enabled:
            return
        self._write_durable_train_record("epochs", {
            "epoch": int(epoch), "statistics": statistics,
        })

    def start_evaluation(self, epoch, dataset, split="val", model_source=None):
        if not self.enabled:
            return
        self._close_eval_writers()
        label = f"epoch_{int(epoch):04d}" if epoch is not None else "standalone"
        self._eval_dir = self.root / "eval" / label
        self._eval_dir.mkdir(parents=True, exist_ok=True)
        self._eval_epoch = None if epoch is None else int(epoch)
        self._eval_batch_index = 0
        self._detailed_seen = 0
        epoch_number = None if epoch is None else int(epoch) + 1
        self._detailed_eval_enabled = bool(
            epoch is None or int(epoch) == 0 or
            epoch_number % self.detailed_epoch_interval == 0 or
            (self.total_epochs > 0 and epoch_number == self.total_epochs)
        )
        self._layerwise_eval_enabled = bool(
            self.layerwise_epoch_interval > 0 and (
                epoch is None or int(epoch) == 0 or
                epoch_number % self.layerwise_epoch_interval == 0 or
                (self.total_epochs > 0 and epoch_number == self.total_epochs)
            )
        )
        self._counts = {name: 0 for name in (
            "images", "ground_truth", "detections", "matches", "queries", "nms",
            "merge_images", "merge_candidates")}
        if dist_utils.is_main_process():
            provenance_provider = getattr(dataset, "get_dataset_provenance", None)
            _write_json(self._eval_dir / "context.json", {
                "schema_version": SCHEMA_VERSION,
                "epoch": self._eval_epoch,
                "split": split,
                "model_source": model_source,
                "dataset_root": getattr(dataset, "root", None),
                "image_count": len(dataset),
                "classes": getattr(dataset, "classes", None),
                "dataset_provenance": (
                    provenance_provider() if provenance_provider is not None else {}),
                "detailed_image_limit_per_rank": self.detailed_image_limit,
                "detailed_epoch_interval": self.detailed_epoch_interval,
                "detailed_layers_enabled": self._detailed_eval_enabled,
                "layerwise_epoch_interval": self.layerwise_epoch_interval,
                "full_validation_layerwise_metrics_enabled": self._layerwise_eval_enabled,
                "query_topk": self.query_topk,
                "nms_status_codes": NMS_STATUS,
            })

    def _emit(self, name, record):
        # Evaluation files are replaced per invocation instead of appending to
        # stale files from an earlier test-only run of the same epoch.
        self._writer(name, self._eval_dir, append=False).write(record)
        self._counts[name] += 1

    def record_evaluation_batch(
        self, outputs, targets, results, post_diagnostics, matching, dataset,
        timings=None, device=None,
    ):
        if not self.enabled:
            return
        pre_logits = outputs.get("diagnostic_pre_logits")
        pre_boxes = outputs.get("diagnostic_pre_boxes")
        layer_logits = outputs.get("diagnostic_layer_logits")
        layer_raw_logits = outputs.get("diagnostic_layer_class_logits_before_lqe")
        layer_lqe_delta = outputs.get("diagnostic_layer_lqe_logit_delta")
        layer_boxes = outputs.get("diagnostic_layer_boxes")
        layer_anchors = outputs.get("diagnostic_layer_anchors")
        layer_input_refs = outputs.get("diagnostic_layer_input_refs")
        layer_distributions = outputs.get("diagnostic_layer_distributions")
        layer_adr_residuals = outputs.get("diagnostic_layer_adr_residuals")
        layer_adr_values = outputs.get("diagnostic_layer_adr_values")
        layer_adr_orthogonality = outputs.get(
            "diagnostic_layer_adr_raw_orthogonality_error")
        distribution_project = outputs.get("diagnostic_distribution_project")
        distribution_names = outputs.get("diagnostic_distribution_names", ())
        refinement_kind = outputs.get("diagnostic_refinement_kind")
        refinement_mode = outputs.get("diagnostic_refinement_mode")
        sampling_locations = outputs.get("diagnostic_sampling_locations")
        sampling_unrotated = outputs.get("diagnostic_sampling_unrotated_offsets")
        sampling_rotated = outputs.get("diagnostic_sampling_rotated_offsets")
        sampling_weights = outputs.get("diagnostic_sampling_attention_weights")
        sampling_points_per_level = outputs.get("diagnostic_sampling_points_per_level")
        query_extensions = outputs.get("diagnostic_query_extensions")
        if distribution_project is not None:
            distribution_project = distribution_project.detach().float().cpu().reshape(-1)
        for batch_index, (target, result, post) in enumerate(
                zip(targets, results, post_diagnostics)):
            # One batched device transfer avoids thousands of scalar GPU
            # synchronizations while serializing complete validation records.
            target = _cpu_dict(target)
            result = _cpu_dict(result)
            post = _cpu_dict(post)
            image_id = int(target["image_id"].reshape(-1)[0])
            annotation = dataset.get_ground_truth(image_id)
            image_name = annotation.get("image_name", str(image_id))
            image_path = annotation.get("image_path")
            acquisition = _dataset_image_metadata(dataset, image_id)
            partition_context = {
                "partition_id": _target_metadata(target, "partition_id", "full_image"),
                "tile_id": _target_metadata(target, "tile_id"),
                "tile_origin": _target_metadata(target, "tile_origin"),
                "tile_size": _target_metadata(target, "tile_size"),
                "tile_overlap": _target_metadata(target, "tile_overlap"),
                "tile_step": _target_metadata(target, "tile_step"),
                "source_image_id": _target_metadata(target, "source_image_id", image_name),
                "source_image_size": _target_metadata(target, "source_image_size"),
            }
            gt_boxes = _restore_target_boxes(target)
            gt_labels = target["labels"].detach()
            difficulties = target.get(
                "difficulty", torch.zeros(len(gt_boxes), device=gt_boxes.device, dtype=torch.long))
            image_record = {
                "schema_version": SCHEMA_VERSION, "rank": self.rank,
                "epoch": self._eval_epoch, "image_id": image_id,
                "evaluation_batch_index": self._eval_batch_index,
                "batch_item_index": batch_index,
                "image_name": image_name, "image_path": image_path,
                "original_size": target.get("orig_size"),
                "canvas_size": target.get("size"),
                "scale_factor": target.get("scale_factor"),
                "padding": target.get("padding"),
                "gt_count": len(gt_boxes), "final_detection_count": len(result["boxes"]),
                "pre_nms_count": len(post["pre_nms_boxes"]),
                **partition_context,
                **acquisition,
            }
            if timings:
                image_record["batch_performance"] = timings
            if device is not None:
                image_record["gpu_memory"] = _memory_snapshot(device)
            self._emit("images", image_record)

            gt_geometry = []
            gt_identity = []
            for gt_index, (box, label, difficult) in enumerate(
                    zip(gt_boxes, gt_labels, difficulties)):
                width, height = float(box[2]), float(box[3])
                corners = rbox_to_corners(box[None], normalized_angle=False)[0]
                image_width = float(target["orig_size"][0])
                image_height = float(target["orig_size"][1])
                geometry = {
                    "area_px2": width * height,
                    "aspect_ratio": width / max(height, 1e-7),
                    "center_boundary_distance_px": min(
                        float(box[0]), float(box[1]),
                        image_width - float(box[0]), image_height - float(box[1])),
                    # Signed distance from the oriented footprint to the image
                    # boundary; negative means that a corner crosses it.
                    "boundary_distance_px": min(
                        float(corners[:, 0].min()), float(corners[:, 1].min()),
                        image_width - float(corners[:, 0].max()),
                        image_height - float(corners[:, 1].max())),
                }
                geometry["boundary_distance_object_scale"] = \
                    geometry["boundary_distance_px"] / max(math.sqrt(geometry["area_px2"]), 1e-7)
                boundary_override = _target_metadata(
                    target, "boundary_distance_px",
                    instance_index=gt_index, instance_count=len(gt_boxes))
                scale_override = _target_metadata(
                    target, "boundary_distance_object_scale",
                    instance_index=gt_index, instance_count=len(gt_boxes))
                if boundary_override is not None:
                    geometry["boundary_distance_px"] = boundary_override
                    geometry["boundary_distance_object_scale"] = \
                        float(boundary_override) / max(math.sqrt(geometry["area_px2"]), 1e-7)
                if scale_override is not None:
                    geometry["boundary_distance_object_scale"] = scale_override
                visible_ratio = _target_metadata(
                    target, "visible_ratio", 1.0, gt_index, len(gt_boxes))
                source_tile_count = _target_metadata(
                    target, "source_tile_count", 1, gt_index, len(gt_boxes))
                source_object_index = _target_metadata(
                    target, "source_object_index", gt_index, gt_index, len(gt_boxes))
                identity = {
                    **partition_context,
                    "source_object_index": source_object_index,
                    "source_object_uid": (
                        f'{partition_context["source_image_id"]}:{int(source_object_index)}'
                        if source_object_index is not None else None),
                }
                gt_geometry.append(geometry)
                gt_identity.append(identity)
                self._emit("ground_truth", {
                    "schema_version": SCHEMA_VERSION, "rank": self.rank,
                    "epoch": self._eval_epoch, "image_id": image_id,
                    "image_name": image_name, "gt_index": gt_index,
                    "label": int(label),
                    "class_name": dataset.classes[int(label)],
                    "box": box, "difficulty": int(difficult),
                    **geometry,
                    "visible_ratio": visible_ratio,
                    "source_tile_count": source_tile_count,
                    **identity,
                    **acquisition,
                })

            pre_count = len(post["pre_nms_boxes"])
            for pre_index in range(pre_count):
                status_code = int(post["status"][pre_index])
                parent_index = int(post["suppressed_by"][pre_index])
                record = {
                    "schema_version": SCHEMA_VERSION, "rank": self.rank,
                    "epoch": self._eval_epoch, "image_id": image_id,
                    "image_name": image_name, "pre_nms_index": pre_index,
                    "query_index": int(post["pre_nms_query_indices"][pre_index]),
                    "box": post["pre_nms_boxes"][pre_index],
                    "score": post["pre_nms_scores"][pre_index],
                    "label": int(post["pre_nms_labels"][pre_index]),
                    "status": NMS_STATUS[status_code],
                    "suppressed_by_pre_nms_index": parent_index if parent_index >= 0 else None,
                    "suppression_iou": post["suppression_iou"][pre_index],
                    **partition_context,
                }
                self._emit("nms", record)

            final_queries = post["pre_nms_query_indices"][post["keep_indices"]]
            for detection_index in range(len(result["boxes"])):
                self._emit("detections", {
                    "schema_version": SCHEMA_VERSION, "rank": self.rank,
                    "epoch": self._eval_epoch, "image_id": image_id,
                    "image_name": image_name, "detection_index": detection_index,
                    "query_index": int(final_queries[detection_index]),
                    "box": result["boxes"][detection_index],
                    "score": result["scores"][detection_index],
                    "label": int(result["labels"][detection_index]),
                    **partition_context,
                })

            source_indices, target_indices = matching["indices"][batch_index]
            source_indices, target_indices = source_indices.cpu(), target_indices.cpu()
            costs = _cpu_dict(
                matching.get("matched_costs", [{} for _ in targets])[batch_index])
            candidates = _cpu_dict(
                matching.get("candidate_costs", [{} for _ in targets])[batch_index])
            candidate_records = []
            for gt_index in range(len(gt_boxes)):
                rows = []
                query_indices = candidates.get("query_indices", torch.empty((len(gt_boxes), 0)))
                for candidate_rank in range(query_indices.shape[1] if query_indices.ndim == 2 else 0):
                    rows.append({
                        "rank": candidate_rank,
                        "query_index": int(query_indices[gt_index, candidate_rank]),
                        **{
                            name: value[gt_index, candidate_rank]
                            for name, value in candidates.items()
                            if name != "query_indices"
                        },
                    })
                candidate_records.append(rows)
            gt_to_match = {int(gt): position for position, gt in enumerate(target_indices.tolist())}
            matched_queries = set(int(query) for query in source_indices.tolist())
            probabilities = post["probabilities"]

            matched_errors = _aligned_box_errors(
                post["query_boxes"][source_indices], gt_boxes[target_indices])
            final_best_index = torch.full((len(gt_boxes),), -1, dtype=torch.long)
            final_best_iou = torch.zeros(len(gt_boxes), dtype=torch.float32)
            official_detection = {
                0.5: torch.full((len(gt_boxes),), -1, dtype=torch.long),
                0.75: torch.full((len(gt_boxes),), -1, dtype=torch.long),
            }
            ignore_boxes = annotation.get("ignore_boxes", torch.empty((0, 5))).detach().float().cpu()
            for label in gt_labels.unique(sorted=True):
                gt_class_indices = torch.nonzero(gt_labels == label, as_tuple=False).squeeze(1)
                final_class_indices = torch.nonzero(
                    result["labels"] == label, as_tuple=False).squeeze(1)
                if not len(gt_class_indices) or not len(final_class_indices):
                    continue
                overlaps = rotated_iou(
                    result["boxes"][final_class_indices], gt_boxes[gt_class_indices],
                    model_space=False)
                best_iou, best_local = overlaps.max(dim=0)
                final_best_iou[gt_class_indices] = best_iou
                final_best_index[gt_class_indices] = final_class_indices[best_local]
                evaluation_boxes = torch.cat((gt_boxes[gt_class_indices], ignore_boxes), dim=0)
                evaluation_difficult = torch.cat((
                    difficulties[gt_class_indices].bool(),
                    torch.ones(len(ignore_boxes), dtype=torch.bool)), dim=0)
                evaluation_overlaps = rotated_iou(
                    result["boxes"][final_class_indices], evaluation_boxes,
                    model_space=False)
                score_order = torch.argsort(
                    result["scores"][final_class_indices], descending=True)
                for threshold, assigned in official_detection.items():
                    used = torch.zeros(len(evaluation_boxes), dtype=torch.bool)
                    for local_detection in score_order.tolist():
                        overlap, evaluation_gt = evaluation_overlaps[local_detection].max(dim=0)
                        evaluation_gt = int(evaluation_gt)
                        if float(overlap) < threshold or bool(evaluation_difficult[evaluation_gt]):
                            continue
                        if not bool(used[evaluation_gt]):
                            used[evaluation_gt] = True
                            assigned[gt_class_indices[evaluation_gt]] = final_class_indices[local_detection]
            for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                position = gt_to_match.get(gt_index)
                if position is None:
                    self._emit("matches", {
                        "schema_version": SCHEMA_VERSION, "rank": self.rank,
                        "epoch": self._eval_epoch, "image_id": image_id,
                        "image_name": image_name, "image_path": image_path,
                        "gt_index": gt_index,
                        "label": int(gt_label), "query_index": None,
                        "class_name": dataset.classes[int(gt_label)],
                        "gt_box": gt_box,
                        "hungarian_candidate_costs": candidate_records[gt_index],
                        "hungarian_weights": matching.get("weights"),
                        "chamfer_distance": matching.get("chamfer_distance"),
                        "difficulty": int(difficulties[gt_index]),
                        "evaluation_outcome": "ignored_difficult"
                            if bool(difficulties[gt_index]) else "miss",
                        **gt_geometry[gt_index],
                        "visible_ratio": _target_metadata(
                            target, "visible_ratio", 1.0, gt_index, len(gt_boxes)),
                        "source_tile_count": _target_metadata(
                            target, "source_tile_count", 1, gt_index, len(gt_boxes)),
                        **gt_identity[gt_index],
                        "failure_stage": "unmatched_capacity",
                    })
                    continue
                query_index = int(source_indices[position])
                query_box = post["query_boxes"][query_index]
                top_score, top_label = probabilities[query_index].max(dim=0)
                target_score = probabilities[query_index, gt_label]
                candidate_mask = (
                    (post["pre_nms_query_indices"] == query_index) &
                    (post["pre_nms_labels"] == gt_label))
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
                candidate_index = int(candidate_indices[0]) if len(candidate_indices) else None
                if candidate_index is not None:
                    candidate_status = NMS_STATUS[int(post["status"][candidate_index])]
                elif float(target_score) < float(post["score_threshold"]):
                    candidate_status = "below_score_threshold"
                else:
                    candidate_status = "topk_truncated"

                best_index = int(final_best_index[gt_index])
                final_index = best_index if best_index >= 0 else None
                final_iou = float(final_best_iou[gt_index])
                final_score = None
                final_box = None
                if final_index is not None:
                    final_score = float(result["scores"][final_index])
                    final_box = result["boxes"][final_index]
                official_index_50 = int(official_detection[0.5][gt_index])
                official_index_75 = int(official_detection[0.75][gt_index])

                errors = {
                    name: _finite_float(values[position])
                    for name, values in matched_errors.items()
                }
                if errors["rotated_iou"] is not None and errors["rotated_iou"] < 0.5:
                    failure_stage = "raw_localization"
                elif int(top_label) != int(gt_label):
                    failure_stage = "raw_classification"
                elif candidate_status == "below_score_threshold":
                    failure_stage = "score_threshold"
                elif candidate_status == "topk_truncated":
                    failure_stage = "topk_truncation"
                elif candidate_status == "nms_overlap":
                    failure_stage = "rotated_nms"
                elif candidate_status == "max_detections":
                    failure_stage = "max_detections"
                elif final_iou < 0.5:
                    failure_stage = "final_assignment"
                else:
                    failure_stage = "success"
                cost_record = {
                    name: values[position] for name, values in costs.items()
                }
                self._emit("matches", {
                    "schema_version": SCHEMA_VERSION, "rank": self.rank,
                    "epoch": self._eval_epoch, "image_id": image_id,
                    "image_name": image_name, "image_path": image_path,
                    "gt_index": gt_index, "label": int(gt_label),
                    "class_name": dataset.classes[int(gt_label)],
                    "gt_box": gt_box, "query_index": query_index,
                    "query_box": query_box,
                    "target_class_score": target_score, "top_score": top_score,
                    "top_label": int(top_label),
                    "class_correct": int(top_label) == int(gt_label),
                    "hungarian_cost": cost_record,
                    "hungarian_candidate_costs": candidate_records[gt_index],
                    "hungarian_weights": matching.get("weights"),
                    "chamfer_distance": matching.get("chamfer_distance"),
                    **errors,
                    "pre_nms_candidate_index": candidate_index,
                    "candidate_status": candidate_status,
                    "final_detection_index": final_index,
                    "final_box": final_box, "final_score": final_score,
                    "final_rotated_iou": final_iou,
                    "official_detection_index_50": official_index_50
                        if official_index_50 >= 0 else None,
                    "official_detection_index_75": official_index_75
                        if official_index_75 >= 0 else None,
                    "final_tp50": official_index_50 >= 0,
                    "final_tp75": official_index_75 >= 0,
                    "evaluation_outcome": "ignored_difficult" if bool(difficulties[gt_index])
                        else "tp75" if official_index_75 >= 0
                        else "tp50" if official_index_50 >= 0 else "miss",
                    "failure_stage": failure_stage,
                    "difficulty": int(difficulties[gt_index]),
                    **gt_geometry[gt_index],
                    "visible_ratio": _target_metadata(
                        target, "visible_ratio", 1.0, gt_index, len(gt_boxes)),
                    "source_tile_count": _target_metadata(
                        target, "source_tile_count", 1, gt_index, len(gt_boxes)),
                    **gt_identity[gt_index],
                    **acquisition,
                })

            if self._detailed_eval_enabled and \
                    self._detailed_seen < self.detailed_image_limit:
                image_pre_logits = pre_logits[batch_index].detach().float().cpu() \
                    if pre_logits is not None else None
                image_pre_boxes = pre_boxes[batch_index].detach().float().cpu() \
                    if pre_boxes is not None else None
                image_layer_logits = layer_logits[:, batch_index].detach().cpu() \
                    if layer_logits is not None else None
                image_layer_raw_logits = layer_raw_logits[:, batch_index].detach().float().cpu() \
                    if layer_raw_logits is not None else None
                image_layer_lqe_delta = layer_lqe_delta[:, batch_index].detach().float().cpu() \
                    if layer_lqe_delta is not None else None
                image_layer_boxes = layer_boxes[:, batch_index].detach().cpu() \
                    if layer_boxes is not None else None
                image_layer_anchors = layer_anchors[:, batch_index].detach().float().cpu() \
                    if layer_anchors is not None else None
                image_layer_input_refs = layer_input_refs[:, batch_index].detach().float().cpu() \
                    if layer_input_refs is not None else None
                image_layer_distributions = layer_distributions[:, batch_index].detach().float().cpu() \
                    if layer_distributions is not None else None
                image_layer_adr_residuals = layer_adr_residuals[:, batch_index].detach().float().cpu() \
                    if layer_adr_residuals is not None else None
                image_layer_adr_values = layer_adr_values[:, batch_index].detach().float().cpu() \
                    if layer_adr_values is not None else None
                image_layer_adr_orthogonality = \
                    layer_adr_orthogonality[:, batch_index].detach().float().cpu() \
                    if layer_adr_orthogonality is not None else None
                image_sampling_locations = sampling_locations[:, batch_index].detach().float().cpu() \
                    if sampling_locations is not None else None
                image_sampling_unrotated = sampling_unrotated[:, batch_index].detach().float().cpu() \
                    if sampling_unrotated is not None else None
                image_sampling_rotated = sampling_rotated[:, batch_index].detach().float().cpu() \
                    if sampling_rotated is not None else None
                image_sampling_weights = sampling_weights[:, batch_index].detach().float().cpu() \
                    if sampling_weights is not None else None
                extension_schema = None
                extension_metadata = {}
                image_query_extensions = {}
                if isinstance(query_extensions, dict):
                    extension_schema = query_extensions.get("schema_version")
                    for extension_name, extension_value in query_extensions.items():
                        if extension_name == "schema_version":
                            continue
                        if torch.is_tensor(extension_value):
                            expected_shape = (
                                int(layer_boxes.shape[0]), len(targets),
                                int(layer_boxes.shape[2]),
                            ) if layer_boxes is not None else None
                            if extension_value.ndim < 3 or expected_shape is None or \
                                    tuple(extension_value.shape[:3]) != expected_shape:
                                raise ValueError(
                                    "diagnostic query extension tensors must have "
                                    "the same [layer,batch,query,...] prefix as "
                                    "diagnostic_layer_boxes")
                            image_query_extensions[extension_name] = \
                                extension_value[:, batch_index].detach().float().cpu()
                        else:
                            extension_metadata[extension_name] = extension_value
                query_scores = probabilities.max(dim=1).values
                top_query_count = min(self.query_topk, len(query_scores))
                selected = set(torch.topk(query_scores, top_query_count).indices.tolist()) \
                    if top_query_count else set()
                selected.update(matched_queries)
                query_to_gt = {
                    int(query): int(gt) for query, gt in zip(source_indices.tolist(), target_indices.tolist())
                }

                def restore_box(normalized):
                    return _restore_target_boxes({
                        **target, "boxes": normalized[None]
                    })[0]

                for query_index in sorted(selected):
                    gt_index = query_to_gt.get(query_index)
                    stages = []

                    if image_pre_logits is not None and image_pre_boxes is not None:
                        logits = image_pre_logits[query_index]
                        probability = logits.sigmoid()
                        score, label = probability.max(dim=0)
                        stage_box = restore_box(image_pre_boxes[query_index])
                        stage_record = {
                            "stage": "pre_box",
                            "stage_index": -1,
                            "display_name": "Pre-box",
                            "box": stage_box,
                            "top_score": score,
                            "top_label": int(label),
                        }
                        if image_layer_input_refs is not None:
                            input_box = restore_box(
                                image_layer_input_refs[0, query_index])
                            stage_record["input_reference_box"] = input_box
                            if gt_index is not None:
                                stage_record["input_reference_geometry"] = _box_errors(
                                    input_box, gt_boxes[gt_index])
                        if gt_index is not None:
                            stage_record.update(_box_errors(
                                stage_box, gt_boxes[gt_index]))
                            stage_record["target_class_score"] = probability[
                                gt_labels[gt_index]]
                        stages.append(stage_record)

                    if image_layer_logits is not None and image_layer_boxes is not None:
                        for layer_index in range(image_layer_logits.shape[0]):
                            logits = image_layer_logits[layer_index, query_index]
                            layer_probability = logits.sigmoid()
                            score, label = layer_probability.max(dim=0)
                            layer_box = restore_box(
                                image_layer_boxes[layer_index, query_index])
                            layer_record = {
                                "stage": "decoder_layer",
                                "stage_index": layer_index,
                                "display_name": f"Decoder {layer_index}",
                                "box": layer_box,
                                "top_score": score, "top_label": int(label),
                            }
                            if image_layer_input_refs is not None:
                                input_box = restore_box(image_layer_input_refs[
                                    layer_index, query_index])
                                layer_record["input_reference_box"] = input_box
                                if gt_index is not None:
                                    layer_record["input_reference_geometry"] = _box_errors(
                                        input_box, gt_boxes[gt_index])
                            if image_layer_anchors is not None:
                                anchor_box = restore_box(image_layer_anchors[
                                    layer_index, query_index])
                                layer_record["initial_anchor_box"] = anchor_box
                                if gt_index is not None:
                                    layer_record["initial_anchor_geometry"] = _box_errors(
                                        anchor_box, gt_boxes[gt_index])
                            if gt_index is not None:
                                layer_record.update(_box_errors(layer_box, gt_boxes[gt_index]))
                                layer_record["target_class_score"] = layer_probability[gt_labels[gt_index]]
                            if image_layer_raw_logits is not None and \
                                    image_layer_lqe_delta is not None:
                                raw_logits = image_layer_raw_logits[
                                    layer_index, query_index]
                                raw_probability = raw_logits.sigmoid()
                                raw_score, raw_label = raw_probability.max(dim=0)
                                lqe_delta = image_layer_lqe_delta[
                                    layer_index, query_index]
                                lqe_record = {
                                    "top_score_before": raw_score,
                                    "top_label_before": int(raw_label),
                                    "top_score_after": score,
                                    "top_label_after": int(label),
                                    "top_class_logit_delta": lqe_delta[label],
                                }
                                if gt_index is not None:
                                    target_label = gt_labels[gt_index]
                                    lqe_record.update({
                                        "target_score_before": raw_probability[target_label],
                                        "target_score_after": layer_probability[target_label],
                                        "target_class_logit_delta": lqe_delta[target_label],
                                    })
                                layer_record["location_quality_estimator"] = lqe_record
                            if image_layer_distributions is not None and distribution_project is not None:
                                distribution_logits = image_layer_distributions[
                                    layer_index, query_index]
                                bins = len(distribution_project)
                                if bins and distribution_logits.numel() % bins == 0:
                                    components = distribution_logits.numel() // bins
                                    names = list(distribution_names)
                                    if len(names) != components:
                                        names = [f"component_{item}" for item in range(components)]
                                    distribution_logits = distribution_logits.reshape(components, bins)
                                    distribution_probability = distribution_logits.softmax(dim=-1)
                                    component_records = {}
                                    for component_index, name in enumerate(names):
                                        probability_values = distribution_probability[component_index]
                                        expectation = torch.sum(
                                            probability_values * distribution_project)
                                        component_records[name] = {
                                            "logits": distribution_logits[component_index],
                                            "probabilities": probability_values,
                                            # This is the paper's weighted
                                            # distribution A(n)P(n).  The LQE
                                            # MLP above is a separate mechanism.
                                            "weighted_values": (
                                                probability_values * distribution_project),
                                            "entropy": -torch.sum(
                                                probability_values *
                                                probability_values.clamp_min(1e-12).log()),
                                            "peak_bin": int(probability_values.argmax()),
                                            "peak_probability": probability_values.max(),
                                            "expected_residual": expectation,
                                            "variance": torch.sum(
                                                probability_values *
                                                (distribution_project - expectation).square()),
                                        }
                                    layer_record["fine_grained_distributions"] = component_records
                            if image_layer_adr_residuals is not None:
                                layer_record["adr_geometry"] = {
                                    "residuals": image_layer_adr_residuals[
                                        layer_index, query_index],
                                    "raw_six_values_normalized": (
                                        image_layer_adr_values[layer_index, query_index]
                                        if image_layer_adr_values is not None else None),
                                    "raw_orthogonality_error": (
                                        image_layer_adr_orthogonality[
                                            layer_index, query_index]
                                        if image_layer_adr_orthogonality is not None else None),
                                }
                            if image_sampling_locations is not None:
                                layer_record["rotated_cross_attention"] = {
                                    "unrotated_offsets": image_sampling_unrotated[
                                        layer_index, query_index],
                                    "rotated_offsets": image_sampling_rotated[
                                        layer_index, query_index],
                                    "sampling_locations_normalized": image_sampling_locations[
                                        layer_index, query_index],
                                    "attention_weights": image_sampling_weights[
                                        layer_index, query_index],
                                }
                            if image_query_extensions:
                                layer_record["method_diagnostics"] = {
                                    "schema_version": extension_schema,
                                    **extension_metadata,
                                    **{
                                        name: value[layer_index, query_index]
                                        for name, value in image_query_extensions.items()
                                    },
                                }
                            if gt_index is not None and stages:
                                previous = stages[-1]
                                layer_record["transition_from_previous"] = {
                                    "rotated_iou_delta": (
                                        layer_record.get("rotated_iou", 0.0) -
                                        previous.get("rotated_iou", 0.0)),
                                    "center_error_px_delta": (
                                        layer_record.get("center_error_px", 0.0) -
                                        previous.get("center_error_px", 0.0)),
                                    "angle_error_deg_delta": (
                                        layer_record.get("angle_error_deg", 0.0) -
                                        previous.get("angle_error_deg", 0.0)),
                                }
                            stages.append(layer_record)
                    self._emit("queries", {
                        "schema_version": SCHEMA_VERSION, "rank": self.rank,
                        "epoch": self._eval_epoch, "image_id": image_id,
                        "image_name": image_name, "query_index": query_index,
                        "image_path": image_path,
                        "original_size": target.get("orig_size"),
                        "matched_gt_index": gt_index,
                        "gt_box": gt_boxes[gt_index] if gt_index is not None else None,
                        "selected_reason": "hungarian" if gt_index is not None else "top_score",
                        "distribution_kind": (
                            refinement_kind.upper() if refinement_kind is not None
                            else None),
                        "refinement_mode": refinement_mode,
                        "distribution_codebook": distribution_project,
                        "sampling_points_per_level": sampling_points_per_level,
                        "stage_sequence": [stage["display_name"] for stage in stages],
                        "stages": stages,
                        **(gt_identity[gt_index] if gt_index is not None else partition_context),
                    })
                self._detailed_seen += 1
        self._eval_batch_index += 1

    def finish_evaluation(self, evaluator=None, refinement_stages=None):
        if not self.enabled or self._eval_dir is None:
            return
        local_summary = {
            "schema_version": SCHEMA_VERSION, "rank": self.rank,
            "epoch": self._eval_epoch, "record_counts": self._counts,
        }
        _write_json(self._eval_dir / f"summary.rank{self.rank:03d}.json", local_summary)
        if dist_utils.is_main_process():
            _write_json(self._eval_dir / "metrics.json", {
                "schema_version": SCHEMA_VERSION, "epoch": self._eval_epoch,
                "metrics": getattr(evaluator, "metrics", {}),
                "per_class": getattr(evaluator, "per_class", {}),
                "per_class_metrics": getattr(evaluator, "per_class_metrics", {}),
                "stats": getattr(evaluator, "stats", []),
                "merge_summary": getattr(evaluator, "merge_summary", {}),
            })
            if refinement_stages is not None:
                _write_json(self._eval_dir / "refinement_stages.json", {
                    "schema_version": SCHEMA_VERSION,
                    "epoch": self._eval_epoch,
                    **refinement_stages,
                })
        self._close_eval_writers()

    def record_global_merge(self, evaluator):
        """Persist original-image merge/NMS evidence from a tiled evaluator."""

        if not self.enabled or self._eval_dir is None or self.rank != 0:
            return
        summary = getattr(evaluator, "merge_summary", None)
        if not summary:
            return
        for record in getattr(evaluator, "merge_image_records", []):
            self._emit("merge_images", {
                "schema_version": SCHEMA_VERSION,
                "rank": self.rank,
                "epoch": self._eval_epoch,
                **record,
            })
        for record in getattr(evaluator, "merge_candidate_records", []):
            self._emit("merge_candidates", {
                "schema_version": SCHEMA_VERSION,
                "rank": self.rank,
                "epoch": self._eval_epoch,
                **record,
            })
        _write_json(self._eval_dir / "merge_summary.json", {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "epoch": self._eval_epoch,
            **summary,
        })

    def _close_eval_writers(self):
        for key, writer in list(self._writers.items()):
            if self._eval_dir is not None and key[0] == str(self._eval_dir):
                writer.close()
                del self._writers[key]

    def close(self):
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    @staticmethod
    def memory_snapshot(device):
        return _memory_snapshot(device)
