#!/usr/bin/env python3
"""Compare no-NMS/NMS on identical saved query candidates, without a GPU."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from engine.data.dataset.moda_dataset import MODADetection
from engine.evaluation.obb.benchmark import BenchmarkOBBEvaluator,probiou_fast_nms
from engine.diagnostics.obb_diagnostics import _write_json


def raw_candidates(logits,boxes):
    scores=logits.sigmoid().flatten()
    classes=logits.shape[-1]
    indices=torch.arange(len(scores))
    return dict(boxes=boxes[indices//classes],scores=scores,labels=indices%classes)


def select(candidates,policy,score_threshold=.01,max_detections=300):
    count=min(300,len(candidates['scores']))
    indices=candidates['scores'].topk(count).indices
    indices=indices[candidates['scores'][indices]>score_threshold]
    values={k:v[indices] for k,v in candidates.items()}
    if policy=='nms' and len(indices):
        keep=probiou_fast_nms(values['boxes'],values['scores'],values['labels'],.7)[0][:max_detections]
    else:keep=torch.arange(min(max_detections,len(indices)))
    return {k:v[keep] for k,v in values.items()}


def archived_images(directory):
    files=sorted((directory/'predictions').glob('*.pt'))
    if not files:raise FileNotFoundError('No final-query prediction bundles found')
    for path in files:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        for row,image_id in enumerate(payload['image_ids']):
            yield image_id,raw_candidates(payload['pred_logits'][row],payload['query_boxes_pixels'][row])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--epoch',default='latest',help='Zero-based epoch, or latest complete')
    parser.add_argument('--policy',choices=('both','detr','nms'),default='both')
    parser.add_argument('--geometric',action='store_true')
    parser.add_argument('--limit-images',type=int,help='Debug only; subset AP is not a benchmark result')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.limit_images is not None and args.limit_images<1:parser.error('--limit-images must be positive')
    epochs=args.run/'diagnostics/eval'
    directory=sorted(p.parent for p in epochs.glob('epoch_*/metrics.json'))[-1] if args.epoch=='latest' else epochs/f'epoch_{int(args.epoch):04d}'
    context=json.loads((directory/'context.json').read_text())
    dataset=MODADetection(context['dataset_root'])
    policies=('detr','nms') if args.policy=='both' else (args.policy,)
    evaluators={p:BenchmarkOBBEvaluator(dataset,require_complete=not bool(args.limit_images),compute_geometric=args.geometric) for p in policies}
    seen=set();started=time.perf_counter()
    for image_id,candidates in archived_images(directory):
        if image_id in seen:continue
        seen.add(image_id)
        for policy,evaluator in evaluators.items():evaluator.update({image_id:select(candidates,policy)})
        if len(seen)%200==0:print(f'{len(seen)} images replayed',flush=True)
        if args.limit_images and len(seen)>=args.limit_images:break
    if not args.limit_images:
        if not (directory/'metrics.json').is_file():parser.error('Incomplete source evaluation')
        missing=set(range(len(dataset)))-seen
        if missing:
            parser.error(f'Prediction archive is missing {len(missing)} images')
    elif args.limit_images:
        # Restrict GT as well as predictions for a labelled debug subset.
        ids=sorted(seen)
        class Subset:
            classes=dataset.classes
            def __len__(self):return len(ids)
            def get_ground_truth(self,index):return dataset.get_ground_truth(ids[index])
        for evaluator in evaluators.values():
            evaluator.dataset=Subset();evaluator.predictions={i:evaluator.predictions[j] for i,j in enumerate(ids)}
    destination=args.output or args.run/f'replay_{directory.name}{"_debug" if args.limit_images else ""}'
    for policy,evaluator in evaluators.items():
        evaluator.accumulate();evaluator.summarize();target=destination/policy;target.mkdir(parents=True,exist_ok=True)
        evaluator.export_paper_table(target)
        _write_json(target/'metrics.json',dict(source=str(directory),policy=policy,metrics=evaluator.metrics,
            per_class_metrics=evaluator.per_class_metrics,images=len(evaluator.dataset),
            scope='debug_subset_not_benchmark' if args.limit_images else 'full_official_test_replay',elapsed_seconds=time.perf_counter()-started))
    print(destination)


if __name__=='__main__':main()
