"""Pure statistics and paper figures for the O² ADR metric-consistency audit.

The runtime collector lives in :mod:`run_metric_consistency_audit`.  Keeping
the statistical contract here makes the scientific gates testable without a
GPU or a model checkpoint.  A record always follows one final Hungarian query
through the traditional pre-box and every decoder layer.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.rtv4.obb.methods.o2.adr import (
    ADR_COMPONENT_NAMES,
    adr_target_residual,
)
from engine.rtv4.rotated_box_ops import rbox_to_corners, rotated_iou


SCHEMA_VERSION = "o2-adr-metric-consistency-v1"
EDGE_COMPONENTS = tuple(ADR_COMPONENT_NAMES[:4])
VERTEX_COMPONENTS = tuple(ADR_COMPONENT_NAMES[4:])

# These thresholds are part of the audit protocol, not values fitted after
# looking at a particular checkpoint.  They encode the scientific statement:
# a geometrically good, locally aligned pre-box should not require a codebook-
# endpoint vertex correction.
PROTOCOL = {
    "good_pre_riou": 0.80,
    "good_pre_angle_error_deg": 5.0,
    "large_vertex_target_abs": 0.80,
    "small_vertex_target_abs": 0.20,
    "strong_degradation_riou": -0.05,
    "minimum_locality_violation_count": 30,
    "minimum_locality_violation_fraction": 0.01,
    "minimum_gain_gap": 0.005,
    "minimum_degradation_risk_gap": 0.10,
    "minimum_vertex_to_edge_endpoint_mass_ratio": 2.0,
    "minimum_vertex_to_edge_variance_ratio": 4.0,
}

SEAM_BIN_EDGES = np.asarray(
    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50 + 1e-9],
    dtype=np.float64,
)
OFFSET_BIN_EDGES = np.asarray(
    [0.0, 0.20, 0.50, 0.80, 1.00, np.inf], dtype=np.float64,
)


def _finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(values):
    values = _finite(values)
    return float(values.mean()) if len(values) else None


def _mean_ci95(values):
    values = _finite(values)
    if not len(values):
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(values.mean())
    if len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    radius = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return {"mean": mean, "ci95_low": mean - radius, "ci95_high": mean + radius}


def _ratio(numerator, denominator):
    if denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _record_values(record):
    stages = record["stages"]
    pre, final = stages[0], stages[-1]
    return {
        "pre_riou": float(pre["rotated_iou"]),
        "pre_angle_error_deg": float(pre["angle_error_deg"]),
        "final_riou": float(final["rotated_iou"]),
        "gain": float(final["rotated_iou"] - pre["rotated_iou"]),
        "layer0_gain": float(stages[1]["rotated_iou"] - pre["rotated_iou"]),
        "max_vertex_target_abs": float(record["max_vertex_target_abs"]),
        "seam_distance": float(record["gt_chart_seam_distance"]),
    }


def is_locality_violation(record):
    values = _record_values(record)
    return (
        values["pre_riou"] >= PROTOCOL["good_pre_riou"]
        and values["pre_angle_error_deg"] <= PROTOCOL["good_pre_angle_error_deg"]
        and values["max_vertex_target_abs"] >= PROTOCOL["large_vertex_target_abs"]
    )


def is_local_control(record):
    values = _record_values(record)
    return (
        values["pre_riou"] >= PROTOCOL["good_pre_riou"]
        and values["pre_angle_error_deg"] <= PROTOCOL["good_pre_angle_error_deg"]
        and values["max_vertex_target_abs"] <= PROTOCOL["small_vertex_target_abs"]
    )


def _group_summary(records, stage_names):
    records = list(records)
    if not records:
        return {
            "count": 0,
            "mean_stage_riou": {name: None for name in stage_names},
            "pre_to_final_gain": _mean_ci95([]),
            "pre_to_layer0_gain": _mean_ci95([]),
            "degraded_fraction": None,
            "strongly_degraded_fraction": None,
        }
    stage_values = np.asarray([
        [stage["rotated_iou"] for stage in record["stages"]]
        for record in records
    ], dtype=np.float64)
    final_gain = stage_values[:, -1] - stage_values[:, 0]
    layer0_gain = stage_values[:, 1] - stage_values[:, 0]
    return {
        "count": len(records),
        "mean_stage_riou": {
            name: float(stage_values[:, index].mean())
            for index, name in enumerate(stage_names)
        },
        "pre_to_final_gain": _mean_ci95(final_gain),
        "pre_to_layer0_gain": _mean_ci95(layer0_gain),
        "improved_fraction": float((final_gain > 0).mean()),
        "degraded_fraction": float((final_gain < 0).mean()),
        "strongly_degraded_fraction": float(
            (final_gain <= PROTOCOL["strong_degradation_riou"]).mean()),
    }


def _binned_summaries(records, stage_names, field, edges):
    summaries = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [
            record for record in records
            if float(record[field]) >= left and float(record[field]) < right
        ]
        label = f"[{left:g}, {right:g})" if np.isfinite(right) else f"[{left:g}, inf)"
        summaries.append({
            "bin_index": index,
            "label": label,
            "left": float(left),
            "right": float(right) if np.isfinite(right) else None,
            **_group_summary(selected, stage_names),
        })
    return summaries


def _distribution_group(records, stage_index, components):
    values = {name: [] for name in (
        "entropy", "variance", "endpoint_mass", "peak_probability")}
    for record in records:
        distributions = record["stages"][stage_index].get("distributions") or {}
        for component in components:
            observation = distributions.get(component)
            if observation is None:
                continue
            for name in values:
                values[name].append(observation[name])
    return {name: _mean(items) for name, items in values.items()}


def summarize_records(records, stage_names):
    """Return the pre-registered hypothesis test and all supporting strata."""

    records = list(records)
    if not records:
        raise ValueError("Metric-consistency audit received no matched records")
    stage_names = list(stage_names)
    if len(stage_names) < 3:
        raise ValueError("Metric-consistency audit requires pre-box and decoder stages")
    for record in records:
        if [stage["name"] for stage in record["stages"]] != stage_names:
            raise ValueError("Stage order changed inside metric-consistency records")

    locality = [record for record in records if is_locality_violation(record)]
    control = [record for record in records if is_local_control(record)]
    large = [
        record for record in records
        if record["max_vertex_target_abs"] >= PROTOCOL["large_vertex_target_abs"]
    ]
    small = [
        record for record in records
        if record["max_vertex_target_abs"] <= PROTOCOL["small_vertex_target_abs"]
    ]
    locality_summary = _group_summary(locality, stage_names)
    control_summary = _group_summary(control, stage_names)
    large_summary = _group_summary(large, stage_names)
    small_summary = _group_summary(small, stage_names)

    final_index = len(stage_names) - 1
    final_edge = _distribution_group(records, final_index, EDGE_COMPONENTS)
    final_vertex = _distribution_group(records, final_index, VERTEX_COMPONENTS)
    endpoint_ratio = _ratio(
        final_vertex["endpoint_mass"], final_edge["endpoint_mass"])
    variance_ratio = _ratio(final_vertex["variance"], final_edge["variance"])
    gain_gap = None
    risk_gap = None
    risk_ratio = None
    if locality_summary["count"] and control_summary["count"]:
        gain_gap = (
            control_summary["pre_to_final_gain"]["mean"]
            - locality_summary["pre_to_final_gain"]["mean"]
        )
        risk_gap = (
            locality_summary["degraded_fraction"]
            - control_summary["degraded_fraction"]
        )
        risk_ratio = _ratio(
            locality_summary["degraded_fraction"],
            control_summary["degraded_fraction"],
        )

    minimum_count = max(
        PROTOCOL["minimum_locality_violation_count"],
        math.ceil(PROTOCOL["minimum_locality_violation_fraction"] * len(records)),
    )
    gates = OrderedDict([
        ("enough_locality_violations", len(locality) >= minimum_count),
        ("locality_gain_gap", gain_gap is not None and
         gain_gap >= PROTOCOL["minimum_gain_gap"]),
        ("locality_degradation_risk_gap", risk_gap is not None and
         risk_gap >= PROTOCOL["minimum_degradation_risk_gap"]),
        ("vertex_endpoint_mass_concentration", endpoint_ratio is not None and
         endpoint_ratio >= PROTOCOL["minimum_vertex_to_edge_endpoint_mass_ratio"]),
        ("vertex_distribution_variance", variance_ratio is not None and
         variance_ratio >= PROTOCOL["minimum_vertex_to_edge_variance_ratio"]),
    ])
    supported = all(gates.values())

    max_offsets = np.asarray(
        [record["max_vertex_target_abs"] for record in records], dtype=np.float64)
    seams = np.asarray(
        [record["gt_chart_seam_distance"] for record in records], dtype=np.float64)
    gains = np.asarray([
        record["stages"][-1]["rotated_iou"] - record["stages"][0]["rotated_iou"]
        for record in records
    ], dtype=np.float64)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "claim_scope": (
            "paper-defined ADR linear-chart targets are not locally metric-"
            "consistent with OBB geometry on this frozen validation run"
        ),
        "protocol": dict(PROTOCOL),
        "matched_count": len(records),
        "stage_names": stage_names,
        "all_matches": _group_summary(records, stage_names),
        "large_vertex_target": large_summary,
        "small_vertex_target": small_summary,
        "geometric_locality_violation": locality_summary,
        "geometric_local_control": control_summary,
        "effect_sizes": {
            "control_minus_violation_mean_gain": gain_gap,
            "violation_minus_control_degradation_risk": risk_gap,
            "degradation_risk_ratio": risk_ratio,
            "vertex_to_edge_endpoint_mass_ratio": endpoint_ratio,
            "vertex_to_edge_variance_ratio": variance_ratio,
        },
        "final_distribution": {
            "edge": final_edge,
            "vertex": final_vertex,
        },
        "pre_registered_gates": gates,
        "seam_bins": _binned_summaries(
            records, stage_names, "gt_chart_seam_distance", SEAM_BIN_EDGES),
        "target_offset_bins": _binned_summaries(
            records, stage_names, "max_vertex_target_abs", OFFSET_BIN_EDGES),
        "population": {
            "seam_distance_le_0p01_fraction": float((seams <= .01).mean()),
            "seam_distance_le_0p05_fraction": float((seams <= .05).mean()),
            "large_vertex_target_fraction": float(
                (max_offsets >= PROTOCOL["large_vertex_target_abs"]).mean()),
            "small_vertex_target_fraction": float(
                (max_offsets <= PROTOCOL["small_vertex_target_abs"]).mean()),
            "pre_to_final_gain_mean": float(gains.mean()),
        },
    }


def select_cases(records):
    """Select mechanism-near cases by declared numeric rules only."""

    records = list(records)
    locality = [record for record in records if is_locality_violation(record)]
    controls = [record for record in records if is_local_control(record)]
    selected = OrderedDict()
    if locality:
        selected["locality_failure"] = min(
            locality,
            key=lambda record: (
                record["stages"][-1]["rotated_iou"]
                - record["stages"][0]["rotated_iou"],
                str(record["record_id"]),
            ),
        )
    if controls:
        selected["local_control_success"] = min(
            controls,
            key=lambda record: (
                -(record["stages"][-1]["rotated_iou"]
                  - record["stages"][0]["rotated_iou"]),
                str(record["record_id"]),
            ),
        )
    selected["layer0_failure"] = min(
        records,
        key=lambda record: (
            record["stages"][1]["rotated_iou"]
            - record["stages"][0]["rotated_iou"],
            str(record["record_id"]),
        ),
    )
    return selected


def theoretical_transition_data(
    reference_angle_deg=0.30,
    target_min_deg=-2.0,
    target_max_deg=2.0,
    samples=401,
):
    """Construct a controlled axis-crossing with continuous OBB geometry."""

    physical_degrees = torch.linspace(
        target_min_deg, target_max_deg, samples, dtype=torch.float64)
    reference = torch.tensor(
        [[0.5, 0.5, 0.30, 0.10, math.radians(reference_angle_deg)]],
        dtype=torch.float64,
    ).expand(samples, -1).clone()
    target = reference.clone()
    target[:, 4] = torch.remainder(
        physical_degrees * math.pi / 180.0, math.pi)
    residual = adr_target_residual(
        reference, target, normalized_angle=False)
    iou = rotated_iou(
        reference, target, aligned=True, model_space=False)
    return {
        "reference_angle_deg": float(reference_angle_deg),
        "target_physical_angle_deg": physical_degrees.numpy(),
        "rotated_iou": iou.numpy(),
        "epsilon_target_residual": residual[:, 4].numpy(),
        "eta_target_residual": residual[:, 5].numpy(),
    }


def _plot_environment():
    cache = Path(tempfile.gettempdir()) / f"o2_metric_plot_{os.getuid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "xdg"))


def _save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def render_theoretical_transition(path):
    _plot_environment()
    import matplotlib.pyplot as plt

    data = theoretical_transition_data()
    x = data["target_physical_angle_deg"]
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    axes[0].plot(x, data["rotated_iou"], color="#1f77b4", linewidth=2.2)
    axes[0].axvline(0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("rIoU(reference, target)")
    axes[0].set_title(
        "OBB geometry stays continuous while the ADR vertex chart changes branch")
    axes[0].grid(alpha=.22)
    axes[1].plot(
        x, data["epsilon_target_residual"], color="#8e44ad",
        linewidth=2.0, label=r"$\epsilon$ target residual")
    axes[1].plot(
        x, data["eta_target_residual"], color="#2ca02c",
        linewidth=2.0, label=r"$\eta$ target residual")
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[1].set_xlabel("Physical target angle around the image x-axis (degrees)")
    axes[1].set_ylabel("ADR normalized target residual")
    axes[1].set_ylim(-1.08, .12)
    axes[1].grid(alpha=.22)
    axes[1].legend(frameon=False, ncol=2)
    figure.text(
        .5, .01,
        "Reference angle = 0.30°. An infinitesimal target rotation across 0° "
        "moves the vertex target from the codebook centre to its endpoint.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, .04, 1, 1))
    _save_figure(figure, path)
    plt.close(figure)
    return True


def _summary_arrays(bins, key):
    selected = [item for item in bins if item["count"]]
    labels = [item["label"] for item in selected]
    means = np.asarray([item[key]["mean"] for item in selected], dtype=float)
    lows = np.asarray([item[key]["ci95_low"] for item in selected], dtype=float)
    highs = np.asarray([item[key]["ci95_high"] for item in selected], dtype=float)
    return selected, labels, means, lows, highs


def render_empirical_mismatch(summary, path):
    _plot_environment()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    seam, labels, means, lows, highs = _summary_arrays(
        summary["seam_bins"], "pre_to_final_gain")
    x = np.arange(len(seam))
    axes[0, 0].errorbar(
        x, means, yerr=np.vstack((means - lows, highs - means)),
        color="#d62728", marker="o", capsize=3, linewidth=1.8,
        label="Pre-box → final")
    layer = np.asarray([
        item["pre_to_layer0_gain"]["mean"] for item in seam], dtype=float)
    axes[0, 0].plot(
        x, layer, color="#e67e22", marker="s", linewidth=1.6,
        label="Pre-box → Decoder 0")
    axes[0, 0].axhline(0, color="#777777", linewidth=.8)
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 0].set_ylabel("Mean rIoU change")
    axes[0, 0].set_xlabel("GT ADR chart-seam distance")
    axes[0, 0].set_title("Refinement gain near the chart transition")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=.2)

    offset, labels, means, lows, highs = _summary_arrays(
        summary["target_offset_bins"], "pre_to_final_gain")
    x = np.arange(len(offset))
    axes[0, 1].errorbar(
        x, means, yerr=np.vstack((means - lows, highs - means)),
        color="#1f77b4", marker="o", capsize=3, linewidth=1.8)
    axes[0, 1].axhline(0, color="#777777", linewidth=.8)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 1].set_ylabel("Mean pre-box → final rIoU change")
    axes[0, 1].set_xlabel(r"max($|\Delta\epsilon|, |\Delta\eta|$) target")
    axes[0, 1].set_title("Large coordinate targets do not imply large geometric gains")
    axes[0, 1].grid(alpha=.2)

    final = summary["final_distribution"]
    metric_names = ("endpoint_mass", "variance")
    positions = np.arange(len(metric_names))
    width = .34
    edge = [final["edge"][name] for name in metric_names]
    vertex = [final["vertex"][name] for name in metric_names]
    axes[1, 0].bar(
        positions - width / 2, edge, width, color="#1f77b4", label="4 edges")
    axes[1, 0].bar(
        positions + width / 2, vertex, width, color="#8e44ad", label="ε/η")
    axes[1, 0].set_xticks(positions, ("Endpoint probability mass", "Variance"))
    axes[1, 0].set_ylabel("Mean at final decoder layer")
    axes[1, 0].set_title("Vertex distributions retain endpoint ambiguity")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=.2)

    violation = summary["geometric_locality_violation"]
    control = summary["geometric_local_control"]
    labels = ("Local control", "Locality violation")
    value_or_zero = lambda value: 0.0 if value is None else value
    degraded = (
        value_or_zero(control["degraded_fraction"]),
        value_or_zero(violation["degraded_fraction"]),
    )
    strong = (
        value_or_zero(control["strongly_degraded_fraction"]),
        value_or_zero(violation["strongly_degraded_fraction"]),
    )
    positions = np.arange(2)
    axes[1, 1].bar(
        positions - width / 2, degraded, width,
        color="#d62728", label="Any degradation")
    axes[1, 1].bar(
        positions + width / 2, strong, width,
        color="#7f0000", label="rIoU drop ≥ 0.05")
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_ylabel("Fraction of matched objects")
    axes[1, 1].set_ylim(0, max((*degraded, *strong, .1)) * 1.25)
    axes[1, 1].set_title("Risk under the pre-registered locality definition")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=.2)

    figure.suptitle(
        f"Full-validation ADR metric-consistency audit — {summary['status']}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, .96))
    _save_figure(figure, path)
    plt.close(figure)
    return True


def _corners(box):
    return rbox_to_corners(
        torch.as_tensor(box, dtype=torch.float32).reshape(1, 5),
        normalized_angle=False,
    )[0].numpy()


def _draw_box(axis, box, color, label, linewidth=2.0, linestyle="-"):
    corners = _corners(box)
    closed = np.concatenate((corners, corners[:1]), axis=0)
    axis.plot(
        closed[:, 0], closed[:, 1], color=color, linewidth=linewidth,
        linestyle=linestyle, label=label)


def _target_bin(value, codebook):
    return float(np.interp(
        np.clip(value, codebook[0], codebook[-1]), codebook,
        np.arange(len(codebook), dtype=float)))


def render_case(case, path, selection_rule):
    """Render box progression and the two vertex distributions for one case."""

    _plot_environment()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    image_path = Path(case["image_path"])
    if not image_path.is_file():
        return False
    image = Image.open(image_path).convert("RGB")
    stages = case["stages"]
    gt_box = case["gt_box"]
    all_corners = np.concatenate(
        [_corners(gt_box), *(_corners(stage["box"]) for stage in stages)], axis=0)
    low, high = all_corners.min(axis=0), all_corners.max(axis=0)
    span = max(float((high - low).max()), 16.0)
    center = (low + high) / 2
    margin = span * 1.15
    xlim = (max(0, center[0] - margin), min(image.width, center[0] + margin))
    ylim = (max(0, center[1] - margin), min(image.height, center[1] + margin))

    stage_count = len(stages)
    grid_columns = 4 * stage_count
    figure = plt.figure(figsize=(max(16, 4 * stage_count), 9.5))
    grid = GridSpec(2, grid_columns, figure=figure, height_ratios=(1.05, 1.0))
    colors = ("#d4a017", "#e67e22", "#2ca02c", "#1f77b4", "#d62728",
              "#8e44ad", "#17becf", "#7f7f7f")
    previous = None
    for index, (stage, color) in enumerate(zip(stages, colors)):
        axis = figure.add_subplot(grid[0, 4 * index:4 * index + 4])
        axis.imshow(image)
        _draw_box(axis, gt_box, "#00bcd4", "GT", linewidth=2.4)
        _draw_box(axis, stage["box"], color, "prediction", linewidth=2.2)
        if stage.get("initial_anchor_box") is not None:
            _draw_box(
                axis, stage["initial_anchor_box"], "#777777", "fixed pre-box anchor",
                linewidth=1.1, linestyle="--")
        delta = "" if previous is None else (
            f", Δ={stage['rotated_iou'] - previous:+.3f}")
        axis.set_title(f"{stage['name']}\nrIoU={stage['rotated_iou']:.3f}{delta}")
        axis.set_xlim(*xlim)
        axis.set_ylim(ylim[1], ylim[0])
        axis.set_xticks([])
        axis.set_yticks([])
        if index == 0:
            axis.legend(loc="upper right", fontsize=7)
        previous = stage["rotated_iou"]

    codebook = np.asarray(case["distribution_codebook"], dtype=float)
    bins = np.arange(len(codebook))
    probabilities = np.asarray(case["full_distribution_probabilities"], dtype=float)
    target = np.asarray(case["target_residual"], dtype=float)
    for component_offset, component_index in enumerate((4, 5)):
        name = ADR_COMPONENT_NAMES[component_index]
        for weighted in (False, True):
            panel_index = component_offset * 2 + int(weighted)
            column = panel_index * stage_count
            axis = figure.add_subplot(grid[1, column:column + stage_count])
            for layer_index, color, label in (
                (0, "#e67e22", "Decoder 0"),
                (probabilities.shape[0] - 1, "#d62728", "Final decoder"),
            ):
                values = probabilities[layer_index, component_index]
                if weighted:
                    values = values * codebook
                axis.plot(bins, values, color=color, linewidth=1.8, label=label)
            target_bin = _target_bin(target[component_index], codebook)
            axis.axvline(
                target_bin, color="#00bcd4", linestyle="--", linewidth=1.3,
                label="GT residual")
            axis.set_xlim(0, len(codebook) - 1)
            axis.set_xlabel("bin index n")
            axis.set_ylabel("A(n)P(n)" if weighted else "P(n)")
            axis.set_title(name.replace("_", " "))
            axis.grid(alpha=.2)
            axis.legend(frameon=False, fontsize=8)

    gain = stages[-1]["rotated_iou"] - stages[0]["rotated_iou"]
    figure.suptitle(
        f"{selection_rule}: {case['image_name']} / GT {case['gt_index']} / "
        f"query {case['query_index']}\n"
        f"pre→final ΔrIoU={gain:+.3f}; target "
        f"ε={target[4]:+.3f}, η={target[5]:+.3f}; "
        f"GT θ={case['gt_angle_deg']:.3f}°, "
        f"pre θ={case['stages'][0]['box'][4] * 180 / math.pi:.3f}°",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, .93))
    _save_figure(figure, path)
    plt.close(figure)
    return True
