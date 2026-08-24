#!/usr/bin/env python3
"""Build published-protocol CODrone OBB patches with analysis metadata.

Examples:

    conda run -n wyq-deim python tools/dataset/split_codrone_obb.py \
      --source-root /home/liuxiaolong/futurama/data/CODrone \
      --output-root /home/liuxiaolong/futurama/data/CODrone/standard_patches \
      --splits train val test --nproc 10

    conda run -n wyq-deim python tools/dataset/split_codrone_obb.py \
      --source-root /home/liuxiaolong/futurama/data/CODrone \
      --output-root /home/liuxiaolong/futurama/data/CODrone/standard_patches_t \
      --splits train_t val_t test_t --nproc 4
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root()))

from engine.data.dataset.codrone_tiling import (  # noqa: E402
    CODroneTilingProtocol,
    split_codrone_split,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split original CODrone train/val/test images using the published DOTA protocol")
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/home/liuxiaolong/futurama/data/CODrone"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--window-size", type=int, default=1180)
    parser.add_argument(
        "--gap", type=int, default=200,
        help="DOTA gap/overlap; actual sliding step is window-size minus gap")
    parser.add_argument("--image-rate-threshold", type=float, default=0.6)
    parser.add_argument("--iof-threshold", type=float, default=0.7)
    parser.add_argument(
        "--padding-value", nargs=3, type=int, default=[104, 116, 124],
        metavar=("B", "G", "R"), help="OpenCV BGR padding, matching the official script")
    parser.add_argument(
        "--image-extension", default=".jpg",
        help="Patch encoding; JPEG is the storage-efficient formal default, PNG is lossless")
    parser.add_argument(
        "--image-quality", type=int, default=95,
        help="JPEG quality recorded in the protocol manifest (ignored for PNG)")
    parser.add_argument("--nproc", type=int, default=10)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--preview-seed", type=int, default=0)
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Development-only deterministic prefix limit; omitted for official data")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly remove and regenerate an existing split output")
    return parser.parse_args()


def _git_provenance(root: Path):
    def run(*arguments):
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "repository_root": str(root),
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "command": sys.argv,
    }


def main():
    args = parse_args()
    if args.nproc <= 0:
        raise ValueError("--nproc must be positive")
    protocol = CODroneTilingProtocol(
        window_size=args.window_size,
        gap=args.gap,
        image_rate_threshold=args.image_rate_threshold,
        iof_threshold=args.iof_threshold,
        padding_value=tuple(args.padding_value),
        image_extension=args.image_extension,
        image_quality=args.image_quality,
    )
    provenance = _git_provenance(_repo_root())
    all_summaries = {}
    print("CODrone published tiling protocol")
    print(json.dumps(protocol.to_manifest(), ensure_ascii=False, indent=2, sort_keys=True))
    for split in args.splits:
        source = args.source_root / split
        output = args.output_root / split
        print(f"\n[{split}] {source} -> {output}")
        summary = split_codrone_split(
            source,
            output,
            protocol,
            split_name=split,
            nproc=args.nproc,
            preview_samples=args.preview_samples,
            preview_seed=args.preview_seed,
            max_images=args.max_images,
            overwrite=args.overwrite,
            provenance=provenance,
        )
        all_summaries[split] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("\nAll requested CODrone splits completed successfully.")
    return all_summaries


if __name__ == "__main__":
    main()
