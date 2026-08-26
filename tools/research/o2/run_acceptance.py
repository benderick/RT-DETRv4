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
also evaluated with a rotated-NMS control on the exact same logits and boxes,
so their difference isolates overlap suppression itself.
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
from engine.rtv4.obb.methods.o2.adr import (  # noqa: E402
    ADR_COMPONENT_NAMES,
    adr_to_rbox,
)
from engine.rtv4.rotated_box_ops import rbox_to_corners, rotated_iou  # noqa: E402


SCHEMA_VERSION = "o2-acceptance-v1"
MODEL_PIXEL_IOU_TOLERANCE = 1e-3
DECODER_CONTRACT_TOLERANCE = 1e-6
O2_DFINE_M_PARAMETER_RANGE = (19_000_000, 20_000_000)
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
    epoch = None
    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", checkpoint.get("last_epoch"))
    return state, source, epoch


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


def _source_aligned_config_contract(yaml):
    """Validate the single supported O²-DFINE-M training contract."""

    decoder = yaml["RotatedDFINETransformer"]
    encoder = yaml["HybridEncoder"]
    criterion = yaml["RotatedRTv4Criterion"]
    matcher = criterion["matcher"]
    optimizer = yaml["optimizer"]
    postprocessor = yaml["RotatedPostProcessor"]
    train_loader = yaml["train_dataloader"]
    transform_types = tuple(
        operation["type"]
        for operation in train_loader["dataset"]["transforms"]["ops"]
    )
    actual = {
        "backbone": yaml["HGNetv2"]["name"],
        "encoder_expansion": encoder["expansion"],
        "encoder_depth_mult": encoder["depth_mult"],
        "num_queries": decoder["num_queries"],
        "num_decoder_layers": decoder["num_layers"],
        "auxiliary_training": decoder.get("aux_loss", True),
        "refinement_mode": decoder["refinement_mode"],
        "reg_max": decoder["reg_max"],
        "adr_a": decoder["adr_a"],
        "adr_c": decoder["adr_c"],
        "ocd_mode": decoder["ocd_mode"],
        "ocd_crowded_policy": decoder["ocd_crowded_policy"],
        "num_denoising": decoder["num_denoising"],
        "criterion_losses": tuple(criterion["losses"]),
        "criterion_alpha": criterion["alpha"],
        "criterion_kld": (
            criterion["kld_sqrt"], criterion["kld_fun"], criterion["kld_tau"]),
        "criterion_weights": dict(criterion["weight_dict"]),
        "union_matching": criterion["use_uni_set"],
        "matcher_weights": dict(matcher["weight_dict"]),
        "matcher_kld": (
            matcher["kld_sqrt"], matcher["kld_fun"], matcher["kld_tau"]),
        "epochs": yaml["epoches"],
        "total_batch_size": train_loader["total_batch_size"],
        "base_lr": optimizer["lr"],
        "backbone_lr": optimizer["params"][0]["lr"],
        "weight_decay": optimizer["weight_decay"],
        "gradient_clip": yaml["clip_max_norm"],
        "lr_milestones": tuple(yaml["lr_scheduler"]["milestones"]),
        "warmup_duration": yaml["lr_warmup_scheduler"]["warmup_duration"],
        "top_queries": yaml["num_top_queries"],
        "max_detections": postprocessor["max_detections"],
        "full_image_apply_nms": postprocessor["apply_nms"],
        "mixup_probability": train_loader["collate_fn"]["mixup_prob"],
        "train_transform_types": transform_types,
    }
    expected = {
        "backbone": "B2",
        "encoder_expansion": 1.0,
        "encoder_depth_mult": .67,
        "num_queries": 300,
        "num_decoder_layers": 4,
        "auxiliary_training": True,
        "refinement_mode": "o2_adr",
        "reg_max": 32,
        "adr_a": .5,
        "adr_c": .25,
        "ocd_mode": "box",
        "ocd_crowded_policy": "released_dynamic",
        "num_denoising": 100,
        "criterion_losses": ("vfl", "boxes", "local"),
        "criterion_alpha": .75,
        "criterion_kld": (False, "log1p", 1.0),
        "criterion_weights": {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_angle": 5.0,
            "loss_kld": 2.0,
            "loss_fgl": .15,
        },
        "union_matching": True,
        "matcher_weights": {
            "cost_class": 2.0,
            "cost_bbox": 0.0,
            "cost_angle": 0.0,
            "cost_kld": 2.0,
            "cost_chamfer": 5.0,
        },
        "matcher_kld": (False, "log1p", 1.0),
        "epochs": 72,
        "total_batch_size": 8,
        "base_lr": 5e-5,
        "backbone_lr": 5e-6,
        "weight_decay": 1e-4,
        "gradient_clip": .1,
        "lr_milestones": (500,),
        "warmup_duration": 500,
        "top_queries": 300,
        "max_detections": 300,
        "full_image_apply_nms": False,
        "mixup_probability": 0.0,
        "train_transform_types": (
            "RotatedResize",
            "RotatedRandomFlip",
            "RotatedRandomRotate",
            "RotatedSanitizeBoxes",
            "RotatedPad",
            "RotatedConvertToTensor",
        ),
    }
    differences = {
        name: {"expected": expected[name], "observed": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    return {
        "status": "PASS" if not differences else "FAIL",
        "actual": actual,
        "differences": differences,
    }


def _decoder_contract_differences(outputs):
    """Measure the runtime contracts that distinguish O² refinement.

    These checks use tensors from the actual validation forward pass.  They
    complement unit tests by preventing a future wiring change from silently
    turning six-value, fixed-anchor, layer-to-layer refinement into a
    different decoder while leaving the configuration name unchanged.
    """

    required = (
        "diagnostic_pre_boxes",
        "diagnostic_layer_boxes",
        "diagnostic_layer_anchors",
        "diagnostic_layer_input_refs",
        "diagnostic_layer_distributions",
        "diagnostic_layer_adr_residuals",
        "diagnostic_layer_logits",
        "diagnostic_layer_class_logits_before_lqe",
        "diagnostic_layer_lqe_logit_delta",
    )
    missing = [name for name in required if name not in outputs]
    if missing:
        raise RuntimeError(
            "O² runtime contract is missing diagnostic tensors: "
            + ", ".join(missing))
    if outputs.get("diagnostic_refinement_kind") != "adr":
        raise RuntimeError("O² runtime contract requires ADR refinement")
    if tuple(outputs.get("diagnostic_distribution_names", ())) != \
            tuple(ADR_COMPONENT_NAMES):
        raise RuntimeError(
            "O² runtime contract requires exactly four boundary and two "
            "vertex distributions")

    boxes = outputs["diagnostic_layer_boxes"]
    anchors = outputs["diagnostic_layer_anchors"]
    input_refs = outputs["diagnostic_layer_input_refs"]
    residuals = outputs["diagnostic_layer_adr_residuals"]
    distributions = outputs["diagnostic_layer_distributions"]
    if residuals.shape != (*boxes.shape[:-1], 6):
        raise RuntimeError(
            "O² ADR residual shape does not implement six-value refinement: "
            f"boxes={tuple(boxes.shape)}, residuals={tuple(residuals.shape)}")
    if distributions.shape[:-1] != boxes.shape[:-1] or \
            distributions.shape[-1] % 6:
        raise RuntimeError(
            "O² distribution head is not partitionable into six variables: "
            f"{tuple(distributions.shape)}")

    fixed_anchor = outputs["diagnostic_pre_boxes"].unsqueeze(0).expand_as(anchors)
    fixed_anchor_difference = (anchors - fixed_anchor).abs().max().item()
    reference_chain_difference = (
        (input_refs[1:] - boxes[:-1]).abs().max().item()
        if len(boxes) > 1 else 0.0)

    reconstructed = adr_to_rbox(anchors, residuals, normalized_angle=True)
    reconstructed_corners = rbox_to_corners(
        reconstructed, normalized_angle=True)
    observed_corners = rbox_to_corners(boxes, normalized_angle=True)
    reconstruction_corner_difference = (
        reconstructed_corners - observed_corners).abs().max().item()

    scores = outputs["diagnostic_layer_logits"]
    raw_scores = outputs["diagnostic_layer_class_logits_before_lqe"]
    lqe_delta = outputs["diagnostic_layer_lqe_logit_delta"]
    lqe_identity_difference = (
        scores - raw_scores - lqe_delta).abs().max().item()
    return {
        "max_fixed_anchor_difference": fixed_anchor_difference,
        "max_reference_chain_difference": reference_chain_difference,
        "max_adr_reconstruction_corner_difference": reconstruction_corner_difference,
        "max_lqe_identity_difference": lqe_identity_difference,
    }


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
    max_model_pixel_riou_difference = 0.0

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
        restored = postprocessor.restore_boxes(
            selected, [target] * len(stage_names))
        restored_targets = postprocessor.restore_boxes(
            target_boxes.unsqueeze(0), [target])[0]
        expanded_restored_targets = restored_targets.unsqueeze(0).expand(
            len(stage_names), -1, -1)
        ious = rotated_iou(
            restored.reshape(-1, 5),
            expanded_restored_targets.reshape(-1, 5),
            aligned=True, model_space=False,
        ).reshape(len(stage_names), -1)
        model_ious = rotated_iou(
            selected.reshape(-1, 5), expanded_targets.reshape(-1, 5),
            aligned=True, model_space=True,
        ).reshape(len(stage_names), -1)
        difference = (model_ious - ious).abs().max().item()
        max_model_pixel_riou_difference = max(
            max_model_pixel_riou_difference, difference)
        if difference > MODEL_PIXEL_IOU_TOLERANCE:
            raise RuntimeError(
                "Model-space and original-pixel rIoU disagree: "
                f"max_abs_difference={difference:.6g}, "
                f"tolerance={MODEL_PIXEL_IOU_TOLERANCE:.6g}")
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
            record = {
                "adr_delta": value,
                "image_id": int(target["image_id"].reshape(-1)[0]),
                "query_index": query_index,
                "gt_index": gt_index,
                "gt_label": int(target["labels"][gt_index]),
                "stage_names": stage_names,
                "stage_ious": ious[:, local_index].detach().cpu(),
                "stage_boxes": restored[:, local_index].detach().cpu(),
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
    }, batch_ious, selections, max_model_pixel_riou_difference


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
    training_contract = _source_aligned_config_contract(config.yaml_cfg)
    if training_contract["differences"]:
        raise RuntimeError(
            "Configuration is not the supported O²-DFINE-M training contract: "
            + json.dumps(
                training_contract["differences"], ensure_ascii=False,
                sort_keys=True, default=str))
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad)
    minimum_parameters, maximum_parameters = O2_DFINE_M_PARAMETER_RANGE
    if not minimum_parameters <= parameter_count < maximum_parameters:
        raise RuntimeError(
            "Configured model does not have O²-DFINE-M capacity: "
            f"parameters={parameter_count}, expected in "
            f"[{minimum_parameters}, {maximum_parameters})")
    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "set_diagnostic_mode"):
        raise TypeError("Configured model does not expose OBB decoder diagnostics")
    if not bool(getattr(decoder, "aux_loss", False)):
        raise RuntimeError(
            "O²-DFINE requires auxiliary training: the detached traditional "
            "OBB pre-box must receive its own pre_outputs supervision")
    # Full-dataset acceptance needs layer outputs, not the much larger
    # deformable-attention traces used only for selected paper examples.
    decoder.set_diagnostic_mode(True, capture_attention=False)
    state, checkpoint_source, checkpoint_epoch = _checkpoint_state(checkpoint_path)
    # Strict loading is itself part of acceptance: no missing/unexpected keys
    # are tolerated or hidden behind compatibility fallbacks.
    model.load_state_dict(state, strict=True)
    del state
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
    max_model_pixel_riou_difference = 0.0
    decoder_contract = {
        "max_fixed_anchor_difference": 0.0,
        "max_reference_chain_difference": 0.0,
        "max_adr_reconstruction_corner_difference": 0.0,
        "max_lqe_identity_difference": 0.0,
    }
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
            batch_contract = _decoder_contract_differences(outputs)
            decoder_contract = {
                name: max(decoder_contract[name], value)
                for name, value in batch_contract.items()
            }
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

            oracle, batch_evidence, selections, batch_geometry_difference = \
                _matched_refinement_evidence(
                stage_map, outputs, targets, matcher, postprocessor, best_cases)
            max_model_pixel_riou_difference = max(
                max_model_pixel_riou_difference, batch_geometry_difference)
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
    print("\n=== Rotated-NMS control: final layer ===")
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
        "source_aligned_training_config": training_contract["status"] == "PASS",
        "prebox_auxiliary_supervision_configured": bool(decoder.aux_loss),
        "strict_checkpoint_load": True,
        "full_image_evaluator": evaluator_type == "DotaOBBEvaluator",
        "all_images_evaluated": len(final_nms_evaluator.predictions) == len(dataset),
        "model_pixel_riou_consistent": (
            max_model_pixel_riou_difference <= MODEL_PIXEL_IOU_TOLERANCE),
        "fixed_initial_adr_anchor": (
            decoder_contract["max_fixed_anchor_difference"]
            <= DECODER_CONTRACT_TOLERANCE),
        "layer_output_drives_next_query": (
            decoder_contract["max_reference_chain_difference"]
            <= DECODER_CONTRACT_TOLERANCE),
        "six_value_adr_decodes_reported_box": (
            decoder_contract["max_adr_reconstruction_corner_difference"]
            <= DECODER_CONTRACT_TOLERANCE),
        "lqe_is_applied_to_layer_logits": (
            decoder_contract["max_lqe_identity_difference"]
            <= DECODER_CONTRACT_TOLERANCE),
        "final_ap_not_below_layer0": (
            final_no_nms_value >= stage_metrics[first_layer_name][primary]),
        "final_ap_not_below_prebox": (
            final_no_nms_value >= stage_metrics["prebox"][primary]),
        # This is deliberately reported rather than folded into overall pass:
        # whether a paper-derived baseline must be NMS-free is a reproduction
        # decision, while stage refinement is a direct mechanism requirement.
        "nms_free_not_below_nms_control": final_no_nms_value >= final_nms_value,
    }
    required_gates = (
        "source_aligned_training_config",
        "prebox_auxiliary_supervision_configured",
        "strict_checkpoint_load",
        "full_image_evaluator",
        "all_images_evaluated",
        "model_pixel_riou_consistent",
        "fixed_initial_adr_anchor",
        "layer_output_drives_next_query",
        "six_value_adr_decodes_reported_box",
        "lqe_is_applied_to_layer_logits",
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
        "adr_geometry_contract": "dfine4_plus_vertex2_equal_diagonal",
        "training_contract": training_contract,
        "model_capacity": {
            "parameters": parameter_count,
            "trainable_parameters": trainable_parameter_count,
            "expected_parameter_range": list(O2_DFINE_M_PARAMETER_RANGE),
        },
        "geometry_crosscheck": {
            "reported_coordinate_system": "original_image_pixels_and_radians",
            "max_model_pixel_riou_difference": max_model_pixel_riou_difference,
            "tolerance": MODEL_PIXEL_IOU_TOLERANCE,
        },
        "decoder_contract": {
            "definition": (
                "fixed initial ADR anchor; cumulative layer boxes feed the "
                "next query; four boundary plus two vertex distributions "
                "decode every reported OBB; LQE updates layer logits"
            ),
            **decoder_contract,
            "tolerance": DECODER_CONTRACT_TOLERANCE,
        },
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
            "nms_control_iou_threshold": postprocessor.nms_iou_threshold,
            "candidate_count": candidate_total,
            "kept_without_nms": no_nms_kept_total,
            "kept_with_nms": nms_kept_total,
            "final_output_count_difference": (
                no_nms_kept_total - nms_kept_total),
        },
        "stage_metrics_without_nms": stage_metrics,
        "final_metrics_with_nms_control": final_nms_metrics,
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
