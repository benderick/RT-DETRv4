#!/usr/bin/env python3
"""Small two-process correctness probe, never a detector accuracy experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from engine.rtv4.rtv4 import RTv4
from engine.rtv4.obb.incubator.bcsr import BoundarySpectralRefinement
from test.framework.model.test_model_pipeline import build_tiny_model, build_criterion
from test.research.bcsr.test_adapter import TinyBackbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("logs/research/bcsr/ddp_cpu.json"))
    args = parser.parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        reports = []
        for weighting, reference in (("edge","iterative"),("object","iterative"),("edge","initial")):
            torch.manual_seed(42)
            decoder = build_tiny_model()
            decoder.geometry_adapter = BoundarySpectralRefinement(hidden_dim=32,band_dim=4,routing_dim=8,
                weighting=weighting,sampling_reference=reference)
            model = DistributedDataParallel(RTv4(TinyBackbone(),nn.Identity(),decoder),find_unused_parameters=False)
            criterion = build_criterion()
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
            torch.manual_seed(99+rank)
            images = torch.rand(2,8,64,64)
            targets = [dict(labels=torch.tensor([rank%3]),boxes=torch.tensor([[.5,.5,.4,.2,.2]]),valid_mask=torch.ones(64,64,dtype=torch.bool)),
                       dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,5),valid_mask=torch.ones(64,64,dtype=torch.bool))]
            losses = []
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                result = model(images,targets)
                loss = sum(criterion(result,targets).values())
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite distributed loss/gradient")
                optimizer.step()
                losses.append(float(loss.detach()))
            branch = model.module.decoder.geometry_adapter
            grad = float(branch.query.weight.grad.abs().sum())
            if grad <= 0:
                raise RuntimeError("Spectral routing branch was not activated")
            vector = torch.cat([p.detach().flatten() for p in branch.parameters()])
            minimum, maximum = vector.clone(),vector.clone()
            dist.all_reduce(minimum,op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum,op=dist.ReduceOp.MAX)
            if not torch.equal(minimum,maximum):
                raise RuntimeError("Parameters differ across ranks")
            reports.append(dict(weighting=weighting,sampling_reference=reference,
                                losses=losses,routing_gradient_l1=grad,parameter_sync=True))
        if rank == 0:
            report = dict(scope="synthetic_cpu_ddp_correctness_only",world_size=dist.get_world_size(),results=reports)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report,indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
