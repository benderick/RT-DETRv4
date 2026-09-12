#!/usr/bin/env python3
"""Run D-FINE OBB inference and save visualization plus DOTA text output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.data.transforms import RotatedConvertToTensor, RotatedResizePad  # noqa: E402
from engine.data.transforms import RotatedResize, RotatedPad  # noqa: E402
from engine.data.transforms.rotated_transforms import image_size  # noqa: E402
from engine.data.dataset.moda_dataset import load_moda_image  # noqa: E402
from engine.rtv4.obb_visualization import save_obb_visualization  # noqa: E402
from engine.rtv4.rotated_box_ops import rbox_to_corners  # noqa: E402


def _checkpoint_state(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if "ema" in state:
        state = state["ema"].get("module", state["ema"])
    elif "model" in state:
        state = state["model"]
    return {key.removeprefix("module."): value for key, value in state.items()}


def _images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"}
    return sorted(item for item in path.iterdir() if item.suffix.lower() in suffixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image or image directory")
    parser.add_argument("--output", default="./obb_predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=None)
    args = parser.parse_args()

    config = YAMLConfig(args.config)
    # A full inference checkpoint supplies the backbone too; no download needed.
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
    device = torch.device(args.device)
    model = config.model.to(device).eval()
    model.load_state_dict(_checkpoint_state(args.checkpoint), strict=True)
    postprocessor = config.postprocessor.to(device).eval()
    if args.score_threshold is not None:
        postprocessor.score_threshold = args.score_threshold
    height, width = config.yaml_cfg.get("eval_spatial_size", [1024, 1024])
    resize = RotatedResizePad((width, height), fill=config.yaml_cfg.get("inference_padding_fill", 114))
    to_tensor = RotatedConvertToTensor(normalize_boxes=True,
        box_coordinate_mode=getattr(model.decoder, "box_coordinate_mode", "per_axis"))
    output_dir = Path(args.output)
    visualization_dir = output_dir / "visualizations"
    dota_dir = output_dir / "dota"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    dota_dir.mkdir(parents=True, exist_ok=True)

    class_names = config.yaml_cfg.get("class_names")
    if class_names is None:
        class_names = config.val_dataloader.dataset.classes
    for image_path in _images(args.input):
        original = load_moda_image(image_path) if image_path.suffix.lower() == ".npy" else Image.open(image_path).convert("RGB")
        orig_width, orig_height = image_size(original)
        target = {
            "boxes": torch.empty((0, 5)), "labels": torch.empty(0, dtype=torch.long),
            "area": torch.empty(0), "difficulty": torch.empty(0, dtype=torch.long),
            "iscrowd": torch.empty(0, dtype=torch.long), "image_id": torch.tensor([0]),
            "idx": torch.tensor([0]), "orig_size": torch.tensor([orig_width, orig_height]),
            "size": torch.tensor([orig_width, orig_height]), "scale_factor": torch.ones(2),
            "padding": torch.zeros(4, dtype=torch.long),
        }
        if torch.is_tensor(original):
            target["valid_mask"] = torch.ones((orig_height, orig_width), dtype=torch.bool)
        if "inference_resize_size" in config.yaml_cfg:
            image, target, _ = RotatedResize(config.yaml_cfg["inference_resize_size"])((original,target,None))
            image, target, _ = RotatedPad((width,height),fill=config.yaml_cfg.get("inference_padding_fill",114))((image,target,None))
        else:
            image, target, _ = resize((original, target, None))
        image, target, _ = to_tensor((image, target, None))
        target = {key: value.to(device) for key, value in target.items()}
        with torch.inference_mode():
            prediction = postprocessor(model(image.unsqueeze(0).to(device), targets=[target]), [target])[0]
        prediction = {key: value.cpu() for key, value in prediction.items()}
        save_obb_visualization(
            visualization_dir / f"{image_path.stem}.jpg", original,
            prediction["boxes"], prediction["labels"], prediction["scores"], class_names)
        corners = rbox_to_corners(prediction["boxes"]).reshape(-1, 8)
        with (dota_dir / f"{image_path.stem}.txt").open("w", encoding="utf-8") as handle:
            for polygon, score, label in zip(corners, prediction["scores"], prediction["labels"]):
                coords = " ".join(f"{float(value):.2f}" for value in polygon)
                handle.write(f"{coords} {class_names[int(label)]} {float(score):.6f}\n")
        print(f"{image_path.name}: {len(prediction['boxes'])} detections")


if __name__ == "__main__":
    main()
