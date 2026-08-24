#!/usr/bin/env python3
"""Render mechanism-near O^2 evidence directly from structured run logs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PLOT_CACHE = Path(tempfile.gettempdir()) / f"codrone_plot_cache_{os.getuid()}"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE / "xdg"))

import matplotlib.pyplot as plt
import numpy as np

from tools.analysis.summarize_obb_diagnostics import (
    _diagnostics_root, _epoch_dir, _mean, _records,
)


def _safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in str(value))


def _adr_figure(record, path):
    layers = [layer for layer in record.get("layers", [])
              if layer.get("fine_grained_distributions")]
    codebook = np.asarray(record.get("distribution_codebook"), dtype=float)
    if not layers or not len(codebook):
        return False
    first, last = layers[0], layers[-1]
    names = list(first["fine_grained_distributions"])
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for axis, name in zip(axes.flat, names):
        for layer, color, label in ((first, "#d99b21", f"layer {first['layer']}"),
                                    (last, "#d62728", f"layer {last['layer']}")):
            component = layer["fine_grained_distributions"][name]
            probability = np.asarray(component["probabilities"], dtype=float)
            axis.plot(codebook, probability, marker=".", linewidth=1.5, color=color,
                      label=f"{label}, H={component['entropy']:.3f}")
            axis.axvline(component["expected_residual"], color=color, alpha=0.35)
        axis.set_title(name)
        axis.set_xlabel("ADR residual codebook value")
        axis.set_ylabel("probability")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"ADR internal refinement: {record.get('image_name')} / "
        f"query {record.get('query_index')} / GT {record.get('matched_gt_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _attention_figure(record, path):
    layers = [layer for layer in record.get("layers", [])
              if layer.get("rotated_cross_attention")]
    if not layers:
        return False
    chosen = [layers[0]] if len(layers) == 1 else [layers[0], layers[-1]]
    figure, axes = plt.subplots(1, len(chosen), figsize=(7 * len(chosen), 6), squeeze=False)
    for axis, layer in zip(axes.flat, chosen):
        attention = layer["rotated_cross_attention"]
        before = np.asarray(attention["unrotated_offsets"], dtype=float).reshape(-1, 2)
        after = np.asarray(attention["rotated_offsets"], dtype=float).reshape(-1, 2)
        weights = np.asarray(attention["attention_weights"], dtype=float).reshape(-1)
        sizes = 12 + 220 * weights / max(float(weights.max()), 1e-12)
        axis.scatter(before[:, 0], before[:, 1], s=sizes, facecolors="none",
                     edgecolors="#777777", alpha=0.75, label="axis-aligned offsets")
        axis.scatter(after[:, 0], after[:, 1], s=sizes, c=weights, cmap="viridis",
                     alpha=0.8, label="angle-rotated offsets")
        for source, destination in zip(before, after):
            axis.plot((source[0], destination[0]), (source[1], destination[1]),
                      color="#bbbbbb", linewidth=0.35, alpha=0.4)
        axis.scatter([0], [0], marker="x", s=70, color="red", label="reference centre")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(f"decoder layer {layer['layer']}")
        axis.set_xlabel("normalized x offset")
        axis.set_ylabel("normalized y offset")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        f"Rotation-aware cross-attention: {record.get('image_name')} / "
        f"query {record.get('query_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
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
    axes[0].grid(axis="y", alpha=0.2)
    x = np.arange(len(candidates))
    width = 0.8 / max(len(terms), 1)
    for term_index, term in enumerate(terms):
        axes[1].bar(x - 0.4 + width / 2 + term_index * width,
                    [row.get(term, 0.0) * weights[term] for row in candidates],
                    width=width, label=f"{term} x {weights[term]:g}")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Weighted cost components for the same GT")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8)
    definition = {
        "paper_squared": "squared four-corner Chamfer",
        "released_l2": "released-source L2 corner Chamfer",
    }.get(record.get("chamfer_distance"), "unspecified Chamfer")
    figure.suptitle(
        f"Matching mechanism ({definition}): {record.get('image_name')} / "
        f"GT {record.get('gt_index')} / assigned q{record.get('query_index')}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
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
    axis.set_title("OCD assignment instability (paper definition)")
    axis.grid(alpha=0.2)
    axis.legend(title="noise mode")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("o2_mechanisms"))
    args = parser.parse_args()
    root = _diagnostics_root(args.run)
    directory = _epoch_dir(root, args.epoch)
    args.output.mkdir(parents=True, exist_ok=True)
    index = {"adr": [], "rotated_cross_attention": [], "chamfer": [], "ocd": []}

    query_records = [record for record in _records(directory, "queries")
                     if record.get("matched_gt_index") is not None]
    for record in query_records[:max(args.count, 0)]:
        stem = (f"{_safe_name(record.get('image_name'))}_q{record.get('query_index')}_"
                f"gt{record.get('matched_gt_index')}")
        adr_path = args.output / f"adr_{stem}.png"
        if _adr_figure(record, adr_path):
            index["adr"].append(str(adr_path.resolve()))
        attention_path = args.output / f"rotated_attention_{stem}.png"
        if _attention_figure(record, attention_path):
            index["rotated_cross_attention"].append(str(attention_path.resolve()))

    match_records = [record for record in _records(directory, "matches")
                     if record.get("hungarian_candidate_costs")]
    for record in match_records[:max(args.count, 0)]:
        path = args.output / (
            f"chamfer_{_safe_name(record.get('image_name'))}_gt{record.get('gt_index')}.png")
        if _chamfer_figure(record, path):
            index["chamfer"].append(str(path.resolve()))

    instability_path = args.output / "ocd_assignment_instability.png"
    if _ocd_instability_figure(root, instability_path):
        index["ocd"].append(str(instability_path.resolve()))
    with (args.output / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote O^2 mechanism evidence to {args.output.resolve()}")


if __name__ == "__main__":
    main()
