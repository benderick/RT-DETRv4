#!/usr/bin/env python3
"""Train or evaluate the MODA eight-band and pseudo-RGB baselines."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "baseline": "dfine_obb_o2_fressdet.yml",
    "rgb": "dfine_obb_o2_rgb.yml",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("train", "eval"))
    result.add_argument("variant", choices=tuple(EXPERIMENTS))
    result.add_argument("--gpus", default="0,1", help="GPU IDs; default: 0,1")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--checkpoint", type=Path,
                        help="Evaluate this checkpoint, or resume training from it")
    result.add_argument("--runs-dir", type=Path, default=Path("logs/moda/experiments"))
    result.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    return result


def check_checkpoint(checkpoint, args):
    """Check the saved dataset and input width before resuming or evaluating."""
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    metadata = checkpoint.parent / "configs.json"
    if not metadata.is_file():
        raise ValueError(f"Keep configs.json beside the checkpoint: {metadata}")
    cfg = json.loads(metadata.read_text())["yaml_cfg"]
    expected = "MODADetection" if args.variant == "baseline" else "MODARGBDetection"
    for split in ("train_dataloader", "val_dataloader"):
        dataset = cfg[split]["dataset"]
        if dataset["type"] != expected:
            raise ValueError(f"Checkpoint input does not match {args.variant}")
    channels = 8 if args.variant == "baseline" else 3
    if cfg["HGNetv2"].get("in_channels") != channels:
        raise ValueError(f"Checkpoint input width does not match {args.variant} ({channels} channels)")
    if cfg.get("use_amp", False):
        raise ValueError("This entry uses FP32; checkpoint uses AMP")
    if args.action == "train" and cfg.get("seed", args.seed) != args.seed:
        raise ValueError("Resume with the original --seed")


def command(args):
    if args.seed < 0 or args.workers < 0:
        raise ValueError("Seed and workers must be nonnegative")
    gpu_ids = args.gpus.split(",")
    if (any(not item.isdigit() for item in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids) or 8 % len(gpu_ids)):
        raise ValueError("Use distinct GPU IDs; their count must divide global batch 8")
    config = ROOT / "configs/experiments/moda" / EXPERIMENTS[args.variant]
    run = (ROOT / args.runs_dir / args.variant / f"seed{args.seed}_fp32").resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.action == "eval" and checkpoint is None:
        checkpoint = run / "best_stg1.pth"
    if checkpoint is not None:
        check_checkpoint(checkpoint, args)
        run = checkpoint.parent
    elif any((run / name).exists() for name in
             ("run_spec.json", "configs.json", "last.pth", "best_stg1.pth")):
        raise ValueError(f"Run exists: {run}. Resume with --checkpoint {run / 'last.pth'}")
    output = run if args.action == "train" else run / f"eval_{checkpoint.stem}_detr"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("MPLCONFIGDIR", "/tmp/moda-matplotlib")
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(gpu_ids)}", str(ROOT / "train.py"),
            "-c", str(config), "--seed", str(args.seed), "--output-dir", str(output)]
    if checkpoint is not None:
        argv += ["-r", str(checkpoint)]
    if args.action == "eval":
        argv.append("--test-only")
    argv += ["-u", "use_amp=False", f"train_dataloader.num_workers={args.workers}",
             f"val_dataloader.num_workers={args.workers}"]
    spec = dict(variant=args.variant, seed=args.seed, precision="fp32", config=str(config),
                global_batch=8, epochs=20, gpus=gpu_ids, postprocess="detr", command=argv,
                retained_source_bands=list(range(8)) if args.variant == "baseline" else [4, 2, 1])
    return argv, env, (run, spec) if args.action == "train" and checkpoint is None else None


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        launch, env, record = command(args)
    except (ValueError, KeyError, OSError) as error:
        cli.error(str(error))
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
    print(shlex.join(launch), flush=True)
    if args.dry_run:
        return
    if record:
        run, spec = record
        run.mkdir(parents=True, exist_ok=True)
        (run / "run_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    raise SystemExit(subprocess.run(launch, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
