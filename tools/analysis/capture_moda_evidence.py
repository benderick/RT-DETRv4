#!/usr/bin/env python3
"""Backfill fixed source-evidence panels from retained epoch checkpoints."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from engine.core import YAMLConfig
from engine.diagnostics.source_evidence import SourceEvidenceRecorder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',default='0,4,9,14,19',help='Zero-based retained checkpoint epochs; absent checkpoints are skipped')
    args=parser.parse_args()
    manifest=json.loads((args.run/'run_spec.json').read_text())
    config_path=manifest['config']
    cfg=YAMLConfig(config_path);cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.to(args.device);device=torch.device(args.device)
    recorder=SourceEvidenceRecorder(cfg.diagnostics_source_evidence,args.run,
        cfg.train_dataloader.dataset,cfg.val_dataloader.dataset.transforms,cfg.epoches)
    completed=0
    for epoch in map(int,args.epochs.split(',')):
        path=args.run/f'checkpoint{epoch:04d}.pth'
        if not path.is_file():print(f'Skip absent checkpoint: {path}');continue
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if int(checkpoint['last_epoch'])!=epoch:raise ValueError(f'Checkpoint epoch mismatch: {path}')
        use_ema='ema' in checkpoint and checkpoint['ema'] is not None
        state=checkpoint['ema']['module'] if use_ema else checkpoint['model']
        model.load_state_dict(state,strict=True)
        recorder.capture(model,cfg.criterion,cfg.postprocessor,device,epoch,'ema' if use_ema else 'model')
        completed+=1
    print(f'Processed {completed} checkpoint(s); output: {args.run}/diagnostics/source_evidence')


if __name__=='__main__':main()
