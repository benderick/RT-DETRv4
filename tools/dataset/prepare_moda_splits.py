#!/usr/bin/env python3
"""Write hashed train/dev manifests without copying or modifying MODA data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.data.dataset.moda_dataset import annotation_inventory


def prepare_splits(root, output, *, group_map=None, date_prefix_proxy=False,
                   dev_fraction=0.15, seed=42, debug_images=0):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Split output must be outside the read-only source dataset")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned directory: {output}")
    labels = {p.stem: p for p in sorted((root / "labels").glob("*.txt"))}
    if not labels:
        raise ValueError("No source labels")
    base = dict(schema_version="moda-split-v1", source_split=root.name,
                source_annotation_sha256=annotation_inventory(root), seed=seed)
    if debug_images:
        available = sorted(s for s in labels if (root / "images" / f"{s}.npy").is_file())
        if debug_images < 1 or debug_images > len(available):
            raise ValueError("debug_images must fit the available paired images")
        chosen = sorted(random.Random(seed).sample(available, debug_images))
        payloads = {"debug": dict(base, partition="debug", protocol="explicit_debug_only",
                                  image_ids=chosen)}
    else:
        if not 0 < dev_fraction < 1:
            raise ValueError("dev_fraction must lie strictly between zero and one")
        if bool(group_map) == bool(date_prefix_proxy):
            raise ValueError("Supply a scene group map, or explicitly opt into date-prefix proxy grouping")
        if group_map:
            mapping = json.loads(Path(group_map).read_text())
            if not isinstance(mapping, dict) or set(mapping) != set(labels):
                raise ValueError("group_map must map every source image stem exactly once")
            if any(not isinstance(v, str) or not v for v in mapping.values()):
                raise ValueError("group IDs must be nonempty strings")
            protocol = "user_supplied_scene_groups"
            base["group_map_sha256"] = hashlib.sha256(Path(group_map).read_bytes()).hexdigest()
        else:
            if any(len(s) < 8 or not s[:8].isdigit() for s in labels):
                raise ValueError("Cannot infer eight-digit date prefixes")
            mapping = {s: s[:8] for s in labels}
            protocol = "date_prefix_proxy_not_verified_scene_disjoint"
        groups, classes = defaultdict(list), {}
        for stem, path in labels.items():
            groups[mapping[stem]].append(stem)
            classes[stem] = {line.split()[8] for line in path.read_text().splitlines() if line.strip()}
        all_classes = set().union(*classes.values())
        best, rng = None, random.Random(seed)
        for _ in range(256):
            order = sorted(groups)
            rng.shuffle(order)
            chosen, n = set(), 0
            for group in order:
                if abs(n + len(groups[group]) - len(labels) * dev_fraction) < abs(n - len(labels) * dev_fraction):
                    chosen.add(group)
                    n += len(groups[group])
            dev = sorted(s for s in labels if mapping[s] in chosen)
            train = sorted(set(labels) - set(dev))
            if not dev or not train:
                continue
            if set().union(*(classes[s] for s in dev)) != all_classes or set().union(*(classes[s] for s in train)) != all_classes:
                continue
            error = abs(len(dev) / len(labels) - dev_fraction)
            if best is None or error < best[0]:
                best = error, train, dev
        if best is None:
            raise ValueError("Could not split groups while retaining all classes in train and dev")
        payloads = {}
        for partition, stems in zip(("train", "dev"), best[1:]):
            payloads[partition] = dict(base, partition=partition, protocol=protocol,
                image_ids=stems, group_ids=sorted({mapping[s] for s in stems}),
                requested_dev_fraction=dev_fraction, actual_dev_fraction=len(best[2])/len(labels))
    output.mkdir(parents=True)
    for partition, payload in payloads.items():
        (output / f"{partition}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return {k: {"images": len(v["image_ids"]), "protocol": v["protocol"]} for k, v in payloads.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/MODA/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, help="JSON: image stem -> verified scene group")
    parser.add_argument("--date-prefix-proxy", action="store_true", help="Opt in to a provisional split; scene separation is NOT verified")
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-images", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare_splits(**vars(args)), indent=2))
