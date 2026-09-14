#!/usr/bin/env python3
"""Two-process CPU correctness check for the new branch and all four controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test.research.test_spectral_evidence import tiny_spectral_model,tiny_batch
from test.framework.model.test_model_pipeline import build_criterion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('logs/moda/implementation/ddp_cpu.json'))
    args=parser.parse_args()
    dist.init_process_group('gloo')
    try:
        rank=dist.get_rank();reports=[]
        for mode in ('plain','diagonal','no_compatibility','full'):
            torch.manual_seed(41);model=DistributedDataParallel(tiny_spectral_model(mode),find_unused_parameters=False)
            criterion=build_criterion();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            torch.manual_seed(71+rank);images,targets=tiny_batch();losses=[]
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss=sum(criterion(model(images,targets),targets).values());loss.backward()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                for name,p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():raise RuntimeError(f'Nonfinite gradient: {name}')
                optimizer.step();losses.append(float(loss.detach()))
            adapter=model.module.decoder.query_adapter
            for name,p in adapter.named_parameters():
                if p.grad is None:raise RuntimeError(f'Unused branch parameter: {name}')
            if adapter.semantic_query.weight.grad.abs().sum()<=0:raise RuntimeError('Inactive spectral branch')
            vector=torch.cat([p.detach().flatten() for p in adapter.parameters()])
            low,high=vector.clone(),vector.clone()
            dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
            if not torch.equal(low,high):raise RuntimeError('Parameters not synchronized')
            reports.append({'mode':mode,'losses':losses,'parameters_synchronized':True,
                            'all_branch_parameters_have_gradient_tensors':True})
        if rank==0:
            result={'scope':'synthetic_cpu_ddp_correctness_only','world_size':dist.get_world_size(),'results':reports}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
