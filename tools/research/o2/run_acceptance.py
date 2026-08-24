#!/usr/bin/env python3
"""Freeze-and-audit O²-DFINE without training or changing its predictions.

The audit performs one validation forward pass and answers three separate
questions:

1. Does every decoder stage produce a usable full-dataset prediction set?
2. Does refinement improve DOTA AP, rather than only selected matched boxes?
3. Is the single-image model genuinely NMS-free, or does rotated NMS hide
   duplicate predictions?

All decoder stages use the same standard D-FINE flattened top-k selection.
The primary stage table disables overlap suppression while retaining the
configured score threshold and maximum-detection budget.  The final stage is
also evaluated with the historical rotated-NMS path on the exact same logits
and boxes, so their difference isolates NMS itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.evaluation.obb import DotaOBBEvaluator  # noqa: E402
from engine.rtv4.rotated_box_ops import rbox_to_corners, rotated_iou  # noqa: E402


SCHEMA_VERSION = "o2-acceptance-v1"
SOURCE_FILES = (
    "engine/rtv4/rotated_dfine_decoder.py",
    "engine/rtv4/obb/methods/o2/adr.py",
    "engine/rtv4/dfine_decoder.py",
    "engine/rtv4/rotated_criterion.py",
    "engine/rtv4/rotated_matcher.py",
    "engine/rtv4/rotated_denoising.py",
    "engine/rtv4/rotated_postprocessor.py",
    "engine/rtv4/rotated_box_ops.py",
    "engine/evaluation/obb/dota.py",
    "engine/data/transforms/rotated_transforms.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]
        state = state.get("module", state)
        source = "ema"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        source = "model"
    else:
        state = checkpoint
        source = "root"
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    return state, source, checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


def _git_value(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_snapshot(config_path: Path):
    status = _git_value("status", "--short")
    files = [config_path.resolve(), *(REPO_ROOT / name for name in SOURCE_FILES)]
    return {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(
            (status or "").encode("utf-8")
        ).hexdigest(),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT.resolve()))
            if path.is_relative_to(REPO_ROOT.resolve()) else str(path): _sha256(path)
            for path in files if path.is_file()
        },
    }


def _stage_outputs(outputs):
    required = (
        "diagnostic_pre_logits",
        "diagnostic_pre_boxes",
        "diagnostic_layer_logits",
        "diagnostic_layer_boxes",
    )
    missing = [name for name in required if name not in outputs]
    if missing:
        raise RuntimeError(
            "O² acceptance requires diagnostic decoder outputs; missing "
            + ", ".join(missing)
        )
    result = OrderedDict([
        ("prebox", {
            "pred_logits": outputs["diagnostic_pre_logits"],
            "pred_boxes": outputs["diagnostic_pre_boxes"],
        })
    ])
    logits = outputs["diagnostic_layer_logits"]
    boxes = outputs["diagnostic_layer_boxes"]
    if logits.shape[:3] != boxes.shape[:3]:
        raise RuntimeError(
            f"Layer logits/boxes disagree: {tuple(logits.shape)} vs {tuple(boxes.shape)}")
    for index in range(len(boxes)):
        result[f"layer{index}"] = {
            "pred_logits": logits[index],
            "pred_boxes": boxes[index],
        }
    return result


def _prediction_map(targets, predictions):
    return {
        int(target["image_id"].reshape(-1)[0]): prediction
        for target, prediction in zip(targets, predictions)
    }


def _mean(values):
    return float(torch.as_tensor(values, dtype=torch.float64).mean()) if len(values) else 0.0


def _quantile(values, q):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q)) if tensor.numel() else 0.0


def _candidate_count(outputs, postprocessor):
    probabilities = outputs["pred_logits"].sigmoid() \
        if postprocessor.use_focal_loss else outputs["pred_logits"].softmax(-1)
    count = min(
        postprocessor.num_top_queries,
        probabilities.shape[1] * probabilities.shape[2],
    )
    if count == 0:
        return [0] * len(probabilities)
    top_scores = torch.topk(probabilities.flatten(1), count, dim=1).values
    return (top_scores >= postprocessor.score_threshold).sum(dim=1).tolist()


def _matched_refinement_evidence(
    stage_map,
    outputs,
    targets,
    matcher,
    postprocessor,
    best_cases,
):
    final = next(reversed(stage_map.values()))
    matching = matcher(final, targets)["indices"]
    stage_names = tuple(stage_map)
    stage_boxes = torch.stack([value["pred_boxes"] for value in stage_map.values()])
    oracle_boxes = final["pred_boxes"].clone()
    batch_ious = []
    selections = torch.zeros(len(stage_names), dtype=torch.int64)

    distributions = outputs.get("diagnostic_layer_distributions")
    for batch_index, ((query_indices, target_indices), target) in enumerate(
            zip(matching, targets)):
        if not len(query_indices):
            continue
        query_indices = query_indices.to(stage_boxes.device)
        target_indices = target_indices.to(stage_boxes.device)
        selected = stage_boxes[:, batch_index, query_indices]
        target_boxes = target["boxes"].index_select(0, target_indices)
        expanded_targets = target_boxes.unsqueeze(0).expand(len(stage_names), -1, -1)
        ious = rotated_iou(
            selected.reshape(-1, 5), expanded_targets.reshape(-1, 5),
            aligned=True, normalized_angle=True,
        ).reshape(len(stage_names), -1)
        batch_ious.append(ious.transpose(0, 1).detach().cpu())
        best_stage = ious.argmax(dim=0)
        selections += torch.bincount(
            best_stage.detach().cpu(), minlength=len(stage_names))
        pair_indices = torch.arange(len(query_indices), device=stage_boxes.device)
        oracle_boxes[batch_index, query_indices] = selected[
            best_stage, pair_indices]

        # Cases are selected by the ADR-only layer0 -> final IoU change, not
        # by appearance or confidence, so the visualization cannot be chosen
        # post hoc for looking impressive.
        adr_delta = ious[-1] - ious[1]
        for case_kind, local_index in (
            ("improvement", int(adr_delta.argmax())),
            ("degradation", int(adr_delta.argmin())),
        ):
            value = float(adr_delta[local_index])
            current = best_cases.get(case_kind)
            better = current is None or (
                value > current["adr_delta"] if case_kind == "improvement"
                else value < current["adr_delta"])
            if not better:
                continue
            query_index = int(query_indices[local_index])
            gt_index = int(target_indices[local_index])
            restored = postprocessor.restore_boxes(
                selected[:, local_index].unsqueeze(1),
                [target] * len(stage_names),
            ).squeeze(1)
            record = {
                "adr_delta": value,
                "image_id": int(target["image_id"].reshape(-1)[0]),
                "query_index": query_index,
                "gt_index": gt_index,
                "gt_label": int(target["labels"][gt_index]),
                "stage_names": stage_names,
                "stage_ious": ious[:, local_index].detach().cpu(),
                "stage_boxes": restored.detach().cpu(),
            }
            if distributions is not None:
                record["distribution_logits"] = distributions[
                    :, batch_index, query_index].detach().float().cpu()
                record["distribution_project"] = outputs[
                    "diagnostic_distribution_project"].detach().float().cpu()
                record["distribution_names"] = tuple(
                    outputs["diagnostic_distribution_names"])
            raw_orthogonality = outputs.get(
                "diagnostic_layer_adr_raw_orthogonality_error")
            if raw_orthogonality is not None:
                record["raw_orthogonality_error"] = raw_orthogonality[
                    :, batch_index, query_index].detach().float().cpu()
            best_cases[case_kind] = record

    return {
        "pred_logits": final["pred_logits"],
        "pred_boxes": oracle_boxes,
    }, batch_ious, selections


def _render_case(case, dataset, output: Path):
    # Matplotlib is imported lazily so headless/cache setup cannot affect the
    # model audit and a plotting failure never invalidates computed metrics.
    cache = Path(tempfile.gettempdir()) / f"o2_acceptance_plot_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    image_id = case["image_id"]
    image = Image.open(dataset.images[image_id]).convert("RGB")
    ground_truth = dataset.get_ground_truth(image_id)
    gt_box = ground_truth["boxes"][case["gt_index"]:case["gt_index"] + 1]
    gt_corners = rbox_to_corners(gt_box)[0]
    stage_corners = rbox_to_corners(case["stage_boxes"])
    colors = ("#777777", "#e6ab02", "#e67e22", "#d62728", "#8e44ad")

    figure = plt.figure(figsize=(18, 9))
    grid = GridSpec(2, 4, figure=figure, width_ratios=(2.4, 1, 1, 1))
    image_axis = figure.add_subplot(grid[:, 0])
    image_axis.imshow(image)
    closed_gt = torch.cat((gt_corners, gt_corners[:1]), dim=0)
    image_axis.plot(closed_gt[:, 0], closed_gt[:, 1], color="#00e5ff",
                    linewidth=3.0, label="GT")
    for name, corners, iou, color in zip(
            case["stage_names"], stage_corners, case["stage_ious"], colors):
        closed = torch.cat((corners, corners[:1]), dim=0)
        image_axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.0,
                        label=f"{name}: IoU={float(iou):.3f}")
    all_points = torch.cat((gt_corners.unsqueeze(0), stage_corners), dim=0)
    x_min, y_min = all_points.amin(dim=(0, 1)).tolist()
    x_max, y_max = all_points.amax(dim=(0, 1)).tolist()
    margin = max(x_max - x_min, y_max - y_min, 20.0) * 1.2
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    image_axis.set_xlim(center_x - margin, center_x + margin)
    image_axis.set_ylim(center_y + margin, center_y - margin)
    image_axis.set_title(
        f"{case['stage_names'][1]} -> {case['stage_names'][-1]} "
        f"delta IoU={case['adr_delta']:+.3f}\n"
        f"image={dataset.image_ids[image_id]}, query={case['query_index']}, "
        f"GT={case['gt_index']}"
        + (
            "\nraw orthogonality: "
            + " -> ".join(
                f"{float(value):.3f}"
                for value in case["raw_orthogonality_error"])
            if "raw_orthogonality_error" in case else ""
        ))
    image_axis.legend(loc="upper right", fontsize=8)

    logits = case.get("distribution_logits")
    project = case.get("distribution_project")
    names = case.get("distribution_names", ())
    if logits is not None and project is not None:
        logits = logits.reshape(len(logits), len(names), len(project))
        probabilities = logits.softmax(dim=-1)
        for component, name in enumerate(names):
            axis = figure.add_subplot(grid[component // 3, component % 3 + 1])
            for layer_index in (0, len(probabilities) - 1):
                axis.plot(
                    project, probabilities[layer_index, component],
                    color=colors[layer_index + 1],
                    label=f"layer{layer_index}", linewidth=1.6,
                )
            axis.set_title(name)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.suptitle("O² ADR refinement acceptance: geometry and internal distributions")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _accumulate(evaluator, shared_ground_truth=None):
    if shared_ground_truth is not None:
        evaluator._ground_truth_by_class, evaluator._positives_by_class = shared_ground_truth
    evaluator.accumulate()
    return {
        **evaluator.metrics,
        "per_class_AP50_DOTA07": evaluator.per_class,
        "prediction_count": int(sum(
            len(prediction["scores"]) for prediction in evaluator.predictions.values())),
    }


def run(args):
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = YAMLConfig(str(config_path))
    frozen_config = config.yaml_cfg
    config_hash = _json_hash(frozen_config)
    decoder_config = frozen_config.get("RotatedDFINETransformer", {})
    if decoder_config.get("refinement_mode") != "o2_adr":
        raise ValueError("O² acceptance requires refinement_mode=o2_adr")
    evaluator_type = frozen_config.get("evaluator", {}).get("type")
    if evaluator_type != "DotaOBBEvaluator":
        raise ValueError(
            "This audit is only for full-image evaluation; expected "
            f"DotaOBBEvaluator, got {evaluator_type!r}")

    # Runtime-only resource overrides do not alter model/data semantics.
    config.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    config.yaml_cfg["val_dataloader"].pop("batch_size", None)
    config.yaml_cfg["val_dataloader"]["num_workers"] = args.workers
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False

    device = torch.device(args.device)
    model = config.model.to(device).eval()
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "set_diagnostic_mode"):
        raise TypeError("Configured model does not expose OBB decoder diagnostics")
    decoder.set_diagnostic_mode(True)
    state, checkpoint_source, checkpoint_epoch = _checkpoint_state(checkpoint_path)
    # Strict loading is itself part of acceptance: no missing/unexpected keys
    # are tolerated or hidden behind compatibility fallbacks.
    model.load_state_dict(state, strict=True)
    del state
    geometry_signature = tuple(
        int(value) for value in decoder.adr_geometry_signature.tolist()
    ) if hasattr(decoder, "adr_geometry_signature") else None
    if geometry_signature != (4, 2, 1):
        raise RuntimeError(
            "O² acceptance requires ADR geometry signature (4, 2, 1); "
            f"observed {geometry_signature!r}")

    data_loader = config.val_dataloader
    dataset = data_loader.dataset
    postprocessor = config.postprocessor.to(device).eval()
    matcher = config.criterion.matcher

    stage_evaluators = None
    final_nms_evaluator = DotaOBBEvaluator(
        dataset, use_07_metric=config.yaml_cfg["evaluator"].get("use_07_metric", True))
    oracle_evaluator = DotaOBBEvaluator(
        dataset, use_07_metric=config.yaml_cfg["evaluator"].get("use_07_metric", True))
    matched_ious = []
    adr_raw_orthogonality = []
    oracle_selections = None
    best_cases = {}
    candidate_total = nms_kept_total = no_nms_kept_total = 0
    started = time.perf_counter()

    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "event": "audit_started",
        "dataset_images": len(dataset),
        "resize": config.yaml_cfg.get("eval_spatial_size"),
        "checkpoint_source": checkpoint_source,
    }))
    with torch.inference_mode():
        for batch_index, (samples, targets_cpu) in enumerate(data_loader):
            samples = samples.to(device)
            targets = [
                {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in target.items()}
                for target in targets_cpu
            ]
            outputs = model(samples)
            raw_orthogonality = outputs.get(
                "diagnostic_layer_adr_raw_orthogonality_error")
            if raw_orthogonality is not None:
                adr_raw_orthogonality.append(
                    raw_orthogonality.permute(1, 2, 0).reshape(
                        -1, raw_orthogonality.shape[0]).detach().float().cpu())
            stage_map = _stage_outputs(outputs)
            if stage_evaluators is None:
                stage_evaluators = OrderedDict(
                    (name, DotaOBBEvaluator(
                        dataset,
                        use_07_metric=config.yaml_cfg["evaluator"].get(
                            "use_07_metric", True)))
                    for name in stage_map
                )
                oracle_selections = torch.zeros(len(stage_map), dtype=torch.int64)

            for name, stage in stage_map.items():
                predictions = postprocessor(stage, targets, apply_nms=False)
                stage_evaluators[name].update(_prediction_map(targets, predictions))

            final = next(reversed(stage_map.values()))
            final_nms = postprocessor(final, targets, apply_nms=True)
            final_no_nms = postprocessor(final, targets, apply_nms=False)
            final_nms_evaluator.update(_prediction_map(targets, final_nms))
            candidate_total += sum(_candidate_count(final, postprocessor))
            nms_kept_total += sum(len(item["scores"]) for item in final_nms)
            no_nms_kept_total += sum(len(item["scores"]) for item in final_no_nms)

            oracle, batch_evidence, selections = _matched_refinement_evidence(
                stage_map, outputs, targets, matcher, postprocessor, best_cases)
            matched_ious.extend(batch_evidence)
            oracle_selections += selections
            oracle_predictions = postprocessor(oracle, targets, apply_nms=False)
            oracle_evaluator.update(_prediction_map(targets, oracle_predictions))

            if (batch_index + 1) % args.print_interval == 0 or \
                    batch_index + 1 == len(data_loader):
                elapsed = time.perf_counter() - started
                seen = min((batch_index + 1) * args.batch_size, len(dataset))
                print(json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "event": "forward_progress",
                    "images": seen,
                    "total": len(dataset),
                    "images_per_second": seen / max(elapsed, 1e-9),
                }))

    if stage_evaluators is None:
        raise RuntimeError("Validation loader produced no batches")
    shared_ground_truth = next(iter(stage_evaluators.values()))._ground_truth_cache()
    stage_metrics = OrderedDict()
    for name, evaluator in stage_evaluators.items():
        print(f"\n=== NMS-free full-dataset evaluation: {name} ===")
        stage_metrics[name] = _accumulate(evaluator, shared_ground_truth)
        evaluator.summarize()
    print("\n=== Historical rotated-NMS evaluation: final layer ===")
    final_nms_metrics = _accumulate(final_nms_evaluator, shared_ground_truth)
    final_nms_evaluator.summarize()
    print("\n=== Matched-query oracle layer selection (diagnostic only) ===")
    oracle_metrics = _accumulate(oracle_evaluator, shared_ground_truth)
    oracle_evaluator.summarize()

    evidence = torch.cat(matched_ious, dim=0) if matched_ious \
        else torch.empty((0, len(stage_metrics)))
    layer_only = evidence[:, 1:]
    layer_delta = layer_only[:, -1] - layer_only[:, 0] \
        if len(layer_only) else torch.empty(0)
    refinement = {
        "matched_count": int(len(evidence)),
        "stage_names": list(stage_metrics),
        "mean_riou": evidence.mean(dim=0).tolist() if len(evidence) else [],
        "median_riou": evidence.median(dim=0).values.tolist() if len(evidence) else [],
        "layer0_to_final": {
            "mean_delta": _mean(layer_delta),
            "median_delta": _quantile(layer_delta, 0.5),
            "improved_fraction": _mean((layer_delta > 0).double()),
            "degraded_fraction": _mean((layer_delta < 0).double()),
            "strongly_improved_fraction": _mean((layer_delta >= 0.05).double()),
            "strongly_degraded_fraction": _mean((layer_delta <= -0.05).double()),
            "monotonic_non_decreasing_fraction": _mean(
                (layer_only[:, 1:] >= layer_only[:, :-1]).all(dim=1).double()
            ) if len(layer_only) else 0.0,
        },
        "oracle_stage_selection_count": {
            name: int(count)
            for name, count in zip(stage_metrics, oracle_selections.tolist())
        },
    }
    if adr_raw_orthogonality:
        raw = torch.cat(adr_raw_orthogonality, dim=0)
        refinement["adr_raw_orthogonality_before_completion"] = {
            "definition": (
                "absolute cosine of consecutive gliding-vertex edges before "
                "equal-diagonal rectangle completion; zero is consistent"
            ),
            "per_layer": [
                {
                    "layer": layer,
                    "mean": float(raw[:, layer].mean()),
                    "median": float(raw[:, layer].median()),
                    "p90": float(torch.quantile(raw[:, layer], .9)),
                    "p99": float(torch.quantile(raw[:, layer], .99)),
                    "max": float(raw[:, layer].max()),
                    "fraction_le_0p01": float((raw[:, layer] <= .01).float().mean()),
                    "fraction_ge_0p5": float((raw[:, layer] >= .5).float().mean()),
                }
                for layer in range(raw.shape[1])
            ],
        }

    visualizations = {}
    for name, case in best_cases.items():
        path = output / "visualizations" / f"{name}.png"
        _render_case(case, dataset, path)
        visualizations[name] = str(path)

    primary = "mAP50_75_DOTA07"
    first_layer_name = "layer0"
    final_layer_name = next(reversed(stage_metrics))
    final_no_nms_value = stage_metrics[final_layer_name][primary]
    final_nms_value = final_nms_metrics[primary]
    gates = {
        "strict_checkpoint_load": True,
        "adr_geometry_signature": geometry_signature == (4, 2, 1),
        "full_image_evaluator": evaluator_type == "DotaOBBEvaluator",
        "all_images_evaluated": len(final_nms_evaluator.predictions) == len(dataset),
        "final_ap_not_below_layer0": (
            final_no_nms_value >= stage_metrics[first_layer_name][primary]),
        "final_ap_not_below_prebox": (
            final_no_nms_value >= stage_metrics["prebox"][primary]),
        # This is deliberately reported rather than folded into overall pass:
        # whether a paper-derived baseline must be NMS-free is a reproduction
        # decision, while stage refinement is a direct mechanism requirement.
        "nms_free_not_below_historical_nms": final_no_nms_value >= final_nms_value,
    }
    required_gates = (
        "strict_checkpoint_load",
        "adr_geometry_signature",
        "full_image_evaluator",
        "all_images_evaluated",
        "final_ap_not_below_layer0",
        "final_ap_not_below_prebox",
    )
    accepted = all(gates[name] for name in required_gates)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if accepted else "FAIL",
        "acceptance_scope": "frozen full-image O2 decoder refinement",
        "config": str(config_path),
        "config_sha256": config_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_state_source": checkpoint_source,
        "checkpoint_epoch": checkpoint_epoch,
        "adr_geometry_signature": geometry_signature,
        "adr_geometry_contract": "dfine4_plus_vertex2_equal_diagonal",
        "dataset": {
            "root": str(dataset.root),
            "images": len(dataset),
            "transform_size": config.yaml_cfg.get("eval_spatial_size"),
            "partition_mode": "full_image_resize_pad",
        },
        "postprocessing": {
            "primary_policy": "flat_topk_score_threshold_max_detections_without_overlap_nms",
            "num_top_queries": postprocessor.num_top_queries,
            "score_threshold": postprocessor.score_threshold,
            "max_detections": postprocessor.max_detections,
            "historical_nms_iou_threshold": postprocessor.nms_iou_threshold,
            "candidate_count": candidate_total,
            "kept_without_nms": no_nms_kept_total,
            "kept_with_nms": nms_kept_total,
            "final_output_count_difference": (
                no_nms_kept_total - nms_kept_total),
        },
        "stage_metrics_without_nms": stage_metrics,
        "final_metrics_with_historical_nms": final_nms_metrics,
        "oracle_metrics_without_nms": oracle_metrics,
        "refinement": refinement,
        "gates": gates,
        "visualizations": visualizations,
        "source_snapshot": _source_snapshot(config_path),
        "runtime": {
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "python": sys.version,
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    summary_path = output / "acceptance.json"
    temporary = output / ".acceptance.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    print(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "event": "audit_finished",
        "status": summary["status"],
        "summary": str(summary_path),
        "final_no_nms": final_no_nms_value,
        "final_with_nms": final_nms_value,
        "layer0_no_nms": stage_metrics[first_layer_name][primary],
    }))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/dfine/dfine_obb_o2.yml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "logs/dfine_obb_o2/acceptance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--print-interval", type=int, default=25)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
