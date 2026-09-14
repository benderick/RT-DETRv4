#!/usr/bin/env python3
"""Explicit MODA experiment entry points. Run from the wyq-deim environment."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENTS={
    "baseline":("dfine_obb_o2_fressdet.yml","O²; no additional spectral branch"),
    "plain":("spectral_plain.yml","Same branch; ordinary semantic aggregation"),
    "diagonal":("spectral_diagonal.yml","Per-channel background variance + object compatibility"),
    "no_compatibility":("spectral_no_compatibility.yml","Joint background covariance; omit object compatibility"),
    "full":("spectral_full.yml","Joint background covariance + object compatibility"),
}


def command(args):
    config=ROOT/"configs/experiments/moda"/EXPERIMENTS[args.variant][0]
    root=(ROOT/args.runs_dir).resolve()
    run=root/args.variant/f"seed{args.seed}_{args.precision}"
    env=os.environ.copy()
    env.setdefault("OMP_NUM_THREADS","2")
    env.setdefault("MKL_NUM_THREADS","2")
    env.setdefault("MPLCONFIGDIR","/tmp/moda-matplotlib")
    if args.action=="check":
        return [sys.executable,"-m","unittest","test.research.test_spectral_evidence","test.research.test_source_evidence","test.framework.model.test_model_pipeline","-v"],env,None
    if args.action=="preflight":
        argv=[sys.executable,str(ROOT/"tools/dataset/moda_preflight.py"),"--config",str(config),
              "--device",args.device,"--batch-size",str(args.batch_size),"--steps",str(args.steps),
              "--output",str(run/("preflight_full.json" if args.full_size else "preflight_cpu.json"))]
        if args.full_size:argv += ["--paper-size","--queries","500"]
        else:argv += ["--height","64","--width","96","--queries","20"]
        if args.dense:argv.append("--dense")
        if args.precision=="amp":
            if not args.device.startswith("cuda"):raise ValueError("AMP preflight needs --device cuda:0")
            argv.append("--amp")
        if args.device.startswith("cuda"):env["CUDA_VISIBLE_DEVICES"]=args.gpus
        return argv,env,None
    gpu_ids=args.gpus.split(',')
    if not gpu_ids or any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError("--gpus must contain distinct numeric GPU IDs, e.g. 0,1")
    if 8%len(gpu_ids):raise ValueError("GPU count must divide the fixed global batch 8")
    env["CUDA_VISIBLE_DEVICES"]=args.gpus
    argv=[sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={len(gpu_ids)}",
          str(ROOT/"train.py"),"-c",str(config),"--seed",str(args.seed)]
    output=run
    previous=None
    if args.action=="train":
        if args.checkpoint:raise ValueError("Use resume or eval with --checkpoint")
        # Preflight artifacts are allowed, existing training artifacts are not.
        if any((run/name).exists() for name in ("run_spec.json","configs.json","last.pth","best_stg1.pth")):
            raise ValueError(f"Training output already exists: {run}. Use resume or another --runs-dir/--seed.")
    else:
        checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else run/("last.pth" if args.action=="resume" else "best_stg1.pth")
        if not checkpoint.is_file() and not args.dry_run:raise ValueError(f"Checkpoint not found: {checkpoint}")
        manifest=checkpoint.parent/"run_spec.json"
        if manifest.is_file():
            previous=json.loads(manifest.read_text())
            if previous['variant']!=args.variant:raise ValueError("Checkpoint variant does not match requested experiment")
            if args.action=="resume" and (previous['seed']!=args.seed or previous['precision']!=args.precision):
                raise ValueError("Resume requires the original seed and precision")
        argv += ["-r",str(checkpoint)]
        if args.action=="eval":
            argv.append("--test-only");output=run/f"eval_{checkpoint.stem}"
    policy=args.postprocess
    if args.action=="resume" and previous and policy!=previous.get("postprocess"):
        raise ValueError("Keep the original postprocess when resuming; compare policies with eval or replay, so best-checkpoint scores remain comparable")
    if args.action=="eval":output=run/f"eval_{checkpoint.stem}_{policy}{'_riou' if args.geometric else ''}"
    argv += ["--output-dir",str(output),"-u",f"use_amp={'True' if args.precision=='amp' else 'False'}",
             f"train_dataloader.num_workers={args.workers}",f"val_dataloader.num_workers={args.workers}"]
    count=300
    argv += [f"num_top_queries={count}",f"RotatedPostProcessor.num_top_queries={count}",
             f"RotatedPostProcessor.apply_nms={'False' if policy=='detr' else 'True'}",
             f"evaluator.compute_geometric={'True' if args.geometric else 'False'}"]
    if args.action=="eval" and args.details:
        argv += ["diagnostics_eval_records=full","diagnostics_detailed_image_limit=16",
                 "diagnostics_detailed_epoch_interval=1"]
    spec={"variant":args.variant,"seed":args.seed,"precision":args.precision,"config":str(config),
          "global_batch":8,"epochs":20,"gpus":gpu_ids,"postprocess":policy,"command":argv}
    return argv,env,(run,spec) if args.action=="train" else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("list","check","preflight","train","resume","eval"))
    parser.add_argument("variant",nargs="?",choices=tuple(EXPERIMENTS),default="baseline")
    parser.add_argument("--gpus",default="0,1")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--precision",choices=("fp32","amp"),default="fp32")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--runs-dir",type=Path,default=Path("logs/moda/experiments"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--postprocess",choices=("detr","nms"),default="detr",help="DETR top300 by default; nms is an optional same-candidate control")
    parser.add_argument("--geometric",action="store_true",help="Also calculate geometric rotated-IoU AP (slower; main AP unchanged)")
    parser.add_argument("--device",default="cpu",help="Preflight only")
    parser.add_argument("--batch-size",type=int,default=2,help="Preflight per-process batch")
    parser.add_argument("--steps",type=int,default=4,help="Preflight optimizer updates only")
    parser.add_argument("--full-size",action="store_true",help="Preflight original resize/pad and 500 queries")
    parser.add_argument("--dense",action="store_true",help="Preflight most crowded images")
    parser.add_argument("--details",action="store_true",help="Eval source-sampling diagnostics for the configured fixed image subset")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.action=="list":
        for key,(file,meaning) in EXPERIMENTS.items():print(f"{key:18s} {file:36s} {meaning}")
        return
    if args.seed<0 or args.workers<0 or min(args.batch_size,args.steps)<1:parser.error("Invalid seed, worker or preflight count")
    try:argv,env,record=command(args)
    except ValueError as error:parser.error(str(error))
    print(f"Working directory: {ROOT}",flush=True)
    if "CUDA_VISIBLE_DEVICES" in env:print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}",flush=True)
    print(shlex.join(argv),flush=True)
    if args.dry_run:return
    if record:
        run,spec=record;run.mkdir(parents=True,exist_ok=True)
        (run/"run_spec.json").write_text(json.dumps(spec,indent=2)+"\n")
    raise SystemExit(subprocess.run(argv,cwd=ROOT,env=env).returncode)


if __name__=="__main__":main()
