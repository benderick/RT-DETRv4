#!/usr/bin/env python3
"""Run a frozen full-validation audit of O² ADR metric consistency.

This tool performs inference only.  It never changes the checkpoint, model
configuration, post-processing policy, or O² implementation.  Every final
Hungarian query is tracked through pre-box and all decoder layers; the compact
record contains the paper-defined ADR target, distribution statistics, LQE
effect, and geometric errors needed by the pre-registered audit in
``metric_consistency.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.evaluation.obb import DotaOBBEvaluator  # noqa: E402
from engine.rtv4.obb.methods.o2.adr import (  # noqa: E402
    ADR_COMPONENT_NAMES,
    adr_target_residual,
    rbox_to_adr,
)
from engine.rtv4.rotated_box_ops import angle_distance, rotated_iou  # noqa: E402
from tools.research.o2.metric_consistency import (  # noqa: E402
    EDGE_COMPONENTS,
    PROTOCOL,
    SCHEMA_VERSION,
    VERTEX_COMPONENTS,
    is_local_control,
    is_locality_violation,
    render_case,
    render_empirical_mismatch,
    render_theoretical_transition,
    select_cases,
    summarize_records,
)
from tools.research.o2.run_acceptance import (  # noqa: E402
    _accumulate,
    _checkpoint_state,
    _json_hash,
    _prediction_map,
    _sha256,
    _source_snapshot,
    _stage_outputs,
)


MODEL_PIXEL_IOU_TOLERANCE = 1e-3


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


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
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_gzip_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(
                _jsonable(record), ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            ) + "\n")
    os.replace(temporary, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_metadata(dataset, image_id):
    annotation = dataset.get_ground_truth(image_id)
    image_name = annotation.get("image_name")
    if image_name is None and hasattr(dataset, "image_ids"):
        image_name = str(dataset.image_ids[image_id])
    image_path = annotation.get("image_path")
    if image_path is None and hasattr(dataset, "images"):
        image_path = dataset.images[image_id]
    return str(image_name or image_id), str(image_path) if image_path else None


def _distribution_summaries(probabilities, project):
    expectation = (probabilities * project).sum(dim=-1)
    variance = (
        probabilities * (project - expectation[..., None]).square()
    ).sum(dim=-1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    peak = probabilities.max(dim=-1).values
    endpoint = probabilities[..., 0] + probabilities[..., -1]
    return {
        "expected_residual": expectation,
        "variance": variance,
        "entropy": entropy,
        "peak_probability": peak,
        "endpoint_mass": endpoint,
    }


def _case_score(record, name):
    stages = record["stages"]
    if name == "locality_failure":
        return stages[-1]["rotated_iou"] - stages[0]["rotated_iou"]
    if name == "local_control_success":
        return stages[-1]["rotated_iou"] - stages[0]["rotated_iou"]
    if name == "layer0_failure":
        return stages[1]["rotated_iou"] - stages[0]["rotated_iou"]
    raise KeyError(name)


def _consider_case(cases, record, full_probabilities, project):
    eligibility = {
        "locality_failure": is_locality_violation(record),
        "local_control_success": is_local_control(record),
        "layer0_failure": True,
    }
    for name, eligible in eligibility.items():
        if not eligible:
            continue
        candidate_score = _case_score(record, name)
        current = cases.get(name)
        choose_minimum = name != "local_control_success"
        better = current is None
        if current is not None:
            current_score = _case_score(current, name)
            if candidate_score != current_score:
                better = candidate_score < current_score if choose_minimum \
                    else candidate_score > current_score
            else:
                better = str(record["record_id"]) < str(current["record_id"])
        if better:
            payload = dict(record)
            payload["full_distribution_probabilities"] = full_probabilities
            payload["distribution_codebook"] = project
            payload["selection_rule"] = name
            cases[name] = payload


def _collect_batch(
    outputs,
    targets,
    matching,
    dataset,
    postprocessor,
    stage_names,
    records,
    cases,
    invariants,
):
    stage_boxes = torch.cat((
        outputs["diagnostic_pre_boxes"].unsqueeze(0),
        outputs["diagnostic_layer_boxes"],
    ), dim=0)
    stage_logits = torch.cat((
        outputs["diagnostic_pre_logits"].unsqueeze(0),
        outputs["diagnostic_layer_logits"],
    ), dim=0)
    layer_anchors = outputs["diagnostic_layer_anchors"]
    layer_distributions = outputs["diagnostic_layer_distributions"]
    layer_residuals = outputs["diagnostic_layer_adr_residuals"]
    layer_orthogonality = outputs[
        "diagnostic_layer_adr_raw_orthogonality_error"]
    layer_raw_logits = outputs[
        "diagnostic_layer_class_logits_before_lqe"]
    layer_lqe_delta = outputs["diagnostic_layer_lqe_logit_delta"]
    project = outputs["diagnostic_distribution_project"].detach().float()
    distribution_names = tuple(outputs["diagnostic_distribution_names"])
    if distribution_names != tuple(ADR_COMPONENT_NAMES):
        raise RuntimeError(
            f"Expected O² ADR components {ADR_COMPONENT_NAMES}, got {distribution_names}")
    bins = len(project)
    layer_count = layer_distributions.shape[0]

    for batch_index, ((query_indices, target_indices), target) in enumerate(
            zip(matching, targets)):
        invariants["ground_truth_count"] += int(len(target["boxes"]))
        if not len(query_indices):
            continue
        query_indices = query_indices.to(stage_boxes.device)
        target_indices = target_indices.to(stage_boxes.device)
        invariants["matched_count"] += int(len(query_indices))
        selected = stage_boxes[:, batch_index, query_indices]
        target_boxes = target["boxes"].index_select(0, target_indices)
        target_expanded = target_boxes.unsqueeze(0).expand(len(stage_names), -1, -1)

        # All reported geometry is measured after restoring boxes to the
        # original image.  The same IoU is independently evaluated in the
        # model coordinate system; disagreement is a hard error rather than a
        # plausible-looking paper figure.
        stage_pixel_boxes = postprocessor.restore_boxes(
            selected, [target] * len(stage_names))
        target_pixel_boxes = postprocessor.restore_boxes(
            target_boxes.unsqueeze(0), [target])[0]
        target_pixel_expanded = target_pixel_boxes.unsqueeze(0).expand(
            len(stage_names), -1, -1)
        stage_riou = rotated_iou(
            stage_pixel_boxes.reshape(-1, 5),
            target_pixel_expanded.reshape(-1, 5),
            aligned=True, model_space=False,
        ).reshape(len(stage_names), -1)
        model_stage_riou = rotated_iou(
            selected.reshape(-1, 5), target_expanded.reshape(-1, 5),
            aligned=True, model_space=True,
        ).reshape(len(stage_names), -1)
        model_pixel_difference = (
            model_stage_riou - stage_riou).abs().max().item()
        invariants["max_model_pixel_riou_difference"] = max(
            invariants["max_model_pixel_riou_difference"],
            model_pixel_difference,
        )
        if model_pixel_difference > MODEL_PIXEL_IOU_TOLERANCE:
            raise RuntimeError(
                "Model-space and original-pixel rIoU disagree: "
                f"max_abs_difference={model_pixel_difference:.6g}, "
                f"tolerance={MODEL_PIXEL_IOU_TOLERANCE:.6g}")

        center_error = torch.linalg.vector_norm(
            stage_pixel_boxes[..., :2] - target_pixel_expanded[..., :2], dim=-1)
        target_diagonal = torch.linalg.vector_norm(
            target_pixel_boxes[:, 2:4], dim=-1).clamp_min(1e-7)
        center_error = center_error / target_diagonal.unsqueeze(0)
        angle_error = angle_distance(
            stage_pixel_boxes[..., 4], target_pixel_expanded[..., 4],
            normalized=False) * (180.0 / math.pi)
        width_error = (
            stage_pixel_boxes[..., 2] - target_pixel_expanded[..., 2]
        ).abs() / target_pixel_expanded[..., 2].clamp_min(1e-7)
        height_error = (
            stage_pixel_boxes[..., 3] - target_pixel_expanded[..., 3]
        ).abs() / target_pixel_expanded[..., 3].clamp_min(1e-7)

        selected_logits = stage_logits[:, batch_index, query_indices]
        target_labels = target["labels"].index_select(0, target_indices)
        label_index = target_labels.unsqueeze(0).unsqueeze(-1).expand(
            len(stage_names), -1, 1)
        stage_target_score = selected_logits.sigmoid().gather(
            -1, label_index).squeeze(-1)

        anchors = layer_anchors[:, batch_index, query_indices]
        anchor_spread = (anchors - anchors[:1]).abs().max()
        invariants["max_fixed_anchor_difference"] = max(
            invariants["max_fixed_anchor_difference"], float(anchor_spread))
        target_residual = adr_target_residual(anchors[0], target_boxes)
        target_values, target_scale = rbox_to_adr(target_boxes)
        offset_fraction = (
            target_values[:, 4:] / target_scale.clamp_min(1e-7)
        ).clamp(0, 1)
        seam_distance = torch.minimum(
            offset_fraction, 1 - offset_fraction).amin(dim=-1)

        logits = layer_distributions[:, batch_index, query_indices].float().reshape(
            layer_count, len(query_indices), len(ADR_COMPONENT_NAMES), bins)
        probabilities = logits.softmax(dim=-1)
        observations = _distribution_summaries(
            probabilities, project.reshape(1, 1, 1, -1))
        predicted_residual = layer_residuals[:, batch_index, query_indices].float()
        orthogonality = layer_orthogonality[
            :, batch_index, query_indices].float()
        raw_logits = layer_raw_logits[:, batch_index, query_indices].float()
        raw_target_score = raw_logits.sigmoid().gather(
            -1, target_labels.unsqueeze(0).unsqueeze(-1).expand(
                layer_count, -1, 1),
        ).squeeze(-1)
        lqe_delta = layer_lqe_delta[:, batch_index, query_indices].float().gather(
            -1, target_labels.unsqueeze(0).unsqueeze(-1).expand(
                layer_count, -1, 1),
        ).squeeze(-1)

        anchor_pixel_boxes = postprocessor.restore_boxes(
            anchors, [target] * layer_count)

        image_id = int(target["image_id"].reshape(-1)[0])
        image_name, image_path = _image_metadata(dataset, image_id)
        difficulty = target.get("difficulty")
        tensors = {
            "stage_riou": stage_riou.detach().float().cpu(),
            "center_error": center_error.detach().float().cpu(),
            "angle_error": angle_error.detach().float().cpu(),
            "width_error": width_error.detach().float().cpu(),
            "height_error": height_error.detach().float().cpu(),
            "stage_target_score": stage_target_score.detach().float().cpu(),
            "target_residual": target_residual.detach().float().cpu(),
            "offset_fraction": offset_fraction.detach().float().cpu(),
            "seam_distance": seam_distance.detach().float().cpu(),
            "probabilities": probabilities.detach().float().cpu(),
            "predicted_residual": predicted_residual.detach().float().cpu(),
            "orthogonality": orthogonality.detach().float().cpu(),
            "raw_target_score": raw_target_score.detach().float().cpu(),
            "lqe_delta": lqe_delta.detach().float().cpu(),
            "stage_pixel_boxes": stage_pixel_boxes.detach().float().cpu(),
            "target_pixel_boxes": target_pixel_boxes.detach().float().cpu(),
            "anchor_pixel_boxes": anchor_pixel_boxes.detach().float().cpu(),
        }
        observations = {
            name: value.detach().float().cpu()
            for name, value in observations.items()
        }
        project_cpu = project.detach().float().cpu().tolist()

        for local_index, (query_index, gt_index) in enumerate(zip(
                query_indices.tolist(), target_indices.tolist())):
            stages = []
            for stage_index, name in enumerate(stage_names):
                stage = {
                    "name": name,
                    "box": tensors["stage_pixel_boxes"][stage_index, local_index].tolist(),
                    "rotated_iou": float(tensors["stage_riou"][stage_index, local_index]),
                    "center_error_gt_diagonal": float(
                        tensors["center_error"][stage_index, local_index]),
                    "angle_error_deg": float(
                        tensors["angle_error"][stage_index, local_index]),
                    "width_relative_error": float(
                        tensors["width_error"][stage_index, local_index]),
                    "height_relative_error": float(
                        tensors["height_error"][stage_index, local_index]),
                    "target_class_score": float(
                        tensors["stage_target_score"][stage_index, local_index]),
                }
                if stage_index:
                    layer_index = stage_index - 1
                    stage["initial_anchor_box"] = tensors[
                        "anchor_pixel_boxes"][layer_index, local_index].tolist()
                    stage["predicted_residual"] = tensors[
                        "predicted_residual"][layer_index, local_index].tolist()
                    stage["raw_orthogonality_error"] = float(
                        tensors["orthogonality"][layer_index, local_index])
                    stage["target_score_before_lqe"] = float(
                        tensors["raw_target_score"][layer_index, local_index])
                    stage["target_class_lqe_logit_delta"] = float(
                        tensors["lqe_delta"][layer_index, local_index])
                    stage["distributions"] = {
                        component: {
                            observation_name: float(
                                observations[observation_name][
                                    layer_index, local_index, component_index])
                            for observation_name in observations
                        }
                        for component_index, component in enumerate(ADR_COMPONENT_NAMES)
                    }
                stages.append(stage)

            residual = tensors["target_residual"][local_index].tolist()
            gt_pixel = tensors["target_pixel_boxes"][local_index].tolist()
            max_vertex_target_abs = max(abs(residual[4]), abs(residual[5]))
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_id": f"{image_name}:{int(gt_index)}",
                "image_id": image_id,
                "image_name": image_name,
                "image_path": image_path,
                "query_index": int(query_index),
                "gt_index": int(gt_index),
                "gt_label": int(target_labels[local_index]),
                "difficulty": int(difficulty[gt_index]) if difficulty is not None else 0,
                "gt_box": gt_pixel,
                "gt_angle_deg": float(gt_pixel[4] * 180.0 / math.pi),
                "gt_offset_fractions": tensors[
                    "offset_fraction"][local_index].tolist(),
                "gt_chart_seam_distance": float(
                    tensors["seam_distance"][local_index]),
                "target_residual": residual,
                "max_vertex_target_abs": float(max_vertex_target_abs),
                "target_outside_codebook": bool(
                    max(abs(value) for value in residual) > max(abs(project_cpu[0]), abs(project_cpu[-1]))),
                "stages": stages,
            }
            records.append(record)
            _consider_case(
                cases,
                record,
                tensors["probabilities"][:, local_index].tolist(),
                project_cpu,
            )


def _write_markdown(path, report):
    audit = report["audit"]
    effects = audit["effect_sizes"]
    lines = [
        "# O² ADR metric-consistency audit",
        "",
        f"Status: **{audit['status']}**",
        "",
        f"Matched objects: {audit['matched_count']}",
        "",
        "## Pre-registered effect sizes",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for name, value in effects.items():
        lines.append(f"| {name} | {value:.6f} |" if value is not None else f"| {name} | n/a |")
    lines.extend(["", "## Gates", ""])
    for name, passed in audit["pre_registered_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(["", "## Figures", ""])
    for name, figure in report["figures"].items():
        lines.append(f"- {name}: `{figure['png']}` / `{figure['pdf']}`")
    lines.append("")
    temporary = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "Metric-consistency audit is a deterministic single-process tool; "
            "run it on one GPU without torchrun")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = YAMLConfig(str(config_path))
    frozen_config = config.yaml_cfg
    config_hash = _json_hash(frozen_config)
    decoder_config = frozen_config.get("RotatedDFINETransformer", {})
    if decoder_config.get("refinement_mode") != "o2_adr":
        raise ValueError("Metric-consistency audit requires refinement_mode=o2_adr")
    evaluator_config = frozen_config.get("evaluator", {})
    if evaluator_config.get("type") != "DotaOBBEvaluator":
        raise ValueError("Metric-consistency audit requires a full-image DotaOBBEvaluator")

    # Resource-only overrides leave model and dataset semantics unchanged.
    config.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    config.yaml_cfg["val_dataloader"].pop("batch_size", None)
    config.yaml_cfg["val_dataloader"]["num_workers"] = args.workers
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False

    device = torch.device(args.device)
    model = config.model.to(device).eval()
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "set_diagnostic_mode"):
        raise TypeError("Configured model does not expose O² decoder diagnostics")
    decoder.set_diagnostic_mode(True, capture_attention=False)
    state, checkpoint_source, checkpoint_epoch = _checkpoint_state(checkpoint_path)
    model.load_state_dict(state, strict=True)
    del state
    data_loader = config.val_dataloader
    dataset = data_loader.dataset
    postprocessor = config.postprocessor.to(device).eval()
    matcher = config.criterion.matcher
    stage_evaluators = None
    stage_names = None
    records = []
    cases = {}
    invariants = {
        "ground_truth_count": 0,
        "matched_count": 0,
        "max_fixed_anchor_difference": 0.0,
        "max_model_pixel_riou_difference": 0.0,
    }
    started = time.perf_counter()
    processed_images = 0
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "event": "audit_started",
        "dataset_images": len(dataset),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_state_source": checkpoint_source,
    }))

    try:
        with torch.inference_mode():
            for batch_index, (samples, targets_cpu) in enumerate(data_loader):
                if args.max_images is not None and processed_images >= args.max_images:
                    break
                samples = samples.to(device)
                targets = [
                    {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in target.items()}
                    for target in targets_cpu
                ]
                if args.max_images is not None:
                    remaining = args.max_images - processed_images
                    if remaining < len(targets):
                        samples = samples[:remaining]
                        targets = targets[:remaining]
                outputs = model(samples)
                stage_map = _stage_outputs(outputs)
                current_names = list(stage_map)
                if stage_names is None:
                    stage_names = current_names
                    stage_evaluators = OrderedDict(
                        (name, DotaOBBEvaluator(
                            dataset,
                            use_07_metric=evaluator_config.get("use_07_metric", True)))
                        for name in stage_names
                    )
                elif current_names != stage_names:
                    raise RuntimeError("Decoder stage order changed between batches")

                for name, stage in stage_map.items():
                    predictions = postprocessor(stage, targets, apply_nms=False)
                    stage_evaluators[name].update(_prediction_map(targets, predictions))
                final = next(reversed(stage_map.values()))
                matching = matcher(final, targets)["indices"]
                _collect_batch(
                    outputs, targets, matching, dataset, postprocessor,
                    stage_names, records, cases, invariants,
                )
                processed_images += len(targets)
                if (batch_index + 1) % args.print_interval == 0 or \
                        processed_images == min(len(dataset), args.max_images or len(dataset)):
                    elapsed = time.perf_counter() - started
                    print(json.dumps({
                        "schema_version": SCHEMA_VERSION,
                        "event": "audit_progress",
                        "images": processed_images,
                        "total": min(len(dataset), args.max_images or len(dataset)),
                        "matched_objects": len(records),
                        "images_per_second": processed_images / max(elapsed, 1e-9),
                    }))
    finally:
        decoder.set_diagnostic_mode(False, capture_attention=False)

    if not records or stage_names is None:
        raise RuntimeError("Validation loader produced no matched audit records")
    complete = processed_images == len(dataset)
    stage_metrics = None
    if complete:
        shared_ground_truth = next(iter(stage_evaluators.values()))._ground_truth_cache()
        stage_metrics = OrderedDict(
            (name, _accumulate(evaluator, shared_ground_truth))
            for name, evaluator in stage_evaluators.items()
        )

    audit = summarize_records(records, stage_names)
    deterministic_selection = select_cases(records)
    if {
        name: record["record_id"] for name, record in deterministic_selection.items()
    } != {
        name: record["record_id"] for name, record in cases.items()
    }:
        raise RuntimeError("Online case capture disagrees with declared case-selection rules")

    records_path = output / "matched_records.jsonl.gz"
    _atomic_gzip_records(records_path, records)
    figures_dir = output / "figures"
    figures = OrderedDict()
    theoretical = figures_dir / "theoretical_chart_transition.png"
    render_theoretical_transition(theoretical)
    figures["theoretical_chart_transition"] = {
        "png": str(theoretical), "pdf": str(theoretical.with_suffix(".pdf"))}
    empirical = figures_dir / "empirical_metric_mismatch.png"
    render_empirical_mismatch(audit, empirical)
    figures["empirical_metric_mismatch"] = {
        "png": str(empirical), "pdf": str(empirical.with_suffix(".pdf"))}
    case_index = OrderedDict()
    case_payloads = {}
    for name, case in cases.items():
        path = figures_dir / f"case_{name}.png"
        render_case(case, path, name)
        figures[f"case_{name}"] = {
            "png": str(path), "pdf": str(path.with_suffix(".pdf"))}
        gain = case["stages"][-1]["rotated_iou"] - case["stages"][0]["rotated_iou"]
        case_index[name] = {
            "record_id": case["record_id"],
            "image_id": case["image_id"],
            "image_name": case["image_name"],
            "gt_index": case["gt_index"],
            "query_index": case["query_index"],
            "pre_to_final_gain": gain,
            "pre_to_layer0_gain": (
                case["stages"][1]["rotated_iou"] - case["stages"][0]["rotated_iou"]),
            "max_vertex_target_abs": case["max_vertex_target_abs"],
            "figure": figures[f"case_{name}"],
        }
        case_payloads[name] = case
    _atomic_json(output / "selected_cases.json", case_payloads)

    report = {
        "schema_version": SCHEMA_VERSION,
        "audit": audit,
        "config": str(config_path),
        "config_sha256": config_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_state_source": checkpoint_source,
        "checkpoint_epoch": checkpoint_epoch,
        "dataset": {
            "root": str(dataset.root),
            "images_total": len(dataset),
            "images_processed": processed_images,
            "complete_validation": complete,
            "transform_size": config.yaml_cfg.get("eval_spatial_size"),
        },
        "coverage_and_invariants": {
            **invariants,
            "matched_fraction": invariants["matched_count"] /
                max(invariants["ground_truth_count"], 1),
            "fixed_anchor_exact": invariants["max_fixed_anchor_difference"] == 0.0,
            "model_pixel_riou_consistent": (
                invariants["max_model_pixel_riou_difference"]
                <= MODEL_PIXEL_IOU_TOLERANCE),
        },
        "stage_metrics_without_nms": stage_metrics,
        "record_file": str(records_path),
        "record_file_sha256": _file_sha256(records_path),
        "selected_cases": case_index,
        "figures": figures,
        "source_snapshot": _source_snapshot(config_path),
        "audit_source_sha256": {
            str(path.relative_to(REPO_ROOT)): _file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                REPO_ROOT / "tools/research/o2/metric_consistency.py",
            )
        },
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "max_images": args.max_images,
            "elapsed_seconds": time.perf_counter() - started,
            "python": sys.version,
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)
    _write_markdown(output / "report.md", report)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "event": "audit_finished",
        "status": audit["status"],
        "matched_objects": audit["matched_count"],
        "report": str(report_path),
        "effect_sizes": audit["effect_sizes"],
    }))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs/dfine/dfine_obb_o2.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "logs/dfine_obb_o2/metric_consistency_audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--print-interval", type=int, default=25)
    parser.add_argument(
        "--max-images", type=int,
        help="Optional smoke-test limit; omit for the scientific full-validation audit")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.print_interval <= 0:
        parser.error("--print-interval must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    run(args)


if __name__ == "__main__":
    main()
