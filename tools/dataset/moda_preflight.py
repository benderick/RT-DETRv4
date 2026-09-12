#!/usr/bin/env python3
"""Exercise real MODA preprocessing, optimizer steps, OBB losses and inference."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from engine.core import YAMLConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/moda/dfine_obb_o2.yml")
    parser.add_argument("--debug-split", help="Explicit hashed subset; never inferred from missing data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dense", action="store_true", help="Use images with the most source objects")
    parser.add_argument("--output", type=Path, default=Path("logs/moda/preflight.json"))
    args = parser.parse_args()
    if args.size < 64 or args.size % 32 or min(args.queries, args.batch_size, args.steps) < 1:
        parser.error("size must be a multiple of 32 >=64; other counts must be positive")
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        parser.error("--amp requires CUDA")
    torch.manual_seed(42)
    cfg = YAMLConfig(args.config, eval_spatial_size=[args.size, args.size],
                     RotatedDFINETransformer={"num_queries": args.queries})
    spec = cfg.yaml_cfg["train_dataloader"]["dataset"]
    if args.debug_split:
        spec["split_file"] = args.debug_split
    for operation in spec["transforms"]["ops"]:
        if operation["type"] in ("RotatedResize", "RotatedPad"):
            operation["size"] = [args.size, args.size]
    cfg.yaml_cfg["train_dataloader"].update(total_batch_size=args.batch_size, num_workers=0)
    cfg.yaml_cfg["train_dataloader"]["collate_fn"].update(base_size=args.size)
    dataset = cfg.train_dataloader.dataset
    indices = list(range(len(dataset)))
    if args.dense:
        indices.sort(key=lambda i: len(dataset.get_ground_truth(i)["boxes"]), reverse=True)
    if len(indices) < args.batch_size:
        raise ValueError("Partition has fewer images than the requested batch")
    batch = [dataset[i] for i in indices[:args.batch_size]]
    images = torch.stack([pair[0] for pair in batch]).to(device)
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k,v in pair[1].items()} for pair in batch]
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    steps = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp):
            outputs = model(images, targets)
        losses = criterion(outputs, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        bad = [n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise RuntimeError(f"Nonfinite gradients: {bad[:10]}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.yaml_cfg.get("clip_max_norm", 0.1))
        scaler.step(optimizer)
        scaler.update()
        steps.append({"loss": float(loss.detach()), "loss_terms": len(losses),
                      "dn_split": outputs.get("dn_meta", {}).get("dn_num_split"),
                      "amp_scale": scaler.get_scale()})
    model.eval()
    model.decoder.decoder.diagnostic_mode = True
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=args.amp):
        result = model(images, targets)
    predictions = cfg.postprocessor(result, targets)
    for key in ("pred_boxes", "pred_logits"):
        if not torch.isfinite(result[key]).all():
            raise RuntimeError(f"Nonfinite inference {key}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = dict(config=args.config, device=str(device), amp=args.amp, size=args.size,
                  queries=args.queries, batch_size=args.batch_size, steps=steps,
                  elapsed_seconds=time.perf_counter()-start,
                  scope="code_smoke_only_not_accuracy_or_training_time_estimate",
                  images=[dataset.images[i].stem for i in indices[:args.batch_size]],
                  target_counts=[len(t["boxes"]) for t in targets],
                  prediction_counts=[len(p["boxes"]) for p in predictions],
                  parameters=sum(p.numel() for p in model.parameters()),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                  dataset=dataset.get_dataset_provenance())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
