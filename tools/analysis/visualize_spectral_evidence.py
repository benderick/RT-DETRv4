#!/usr/bin/env python3
"""Plot actual spectral sampling, background support, and query contributions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from engine.core import YAMLConfig
from engine.rtv4.rotated_box_ops import rbox_to_corners
from tools.inference.obb_infer import _checkpoint_state


def run_image(model,dataset,index,device):
    image,target=dataset[index]
    # Only valid image geometry enters model context; labels are not needed.
    metadata={"valid_mask":target["valid_mask"].to(device)}
    with torch.inference_mode():result=model(image[None].to(device),targets=[metadata])
    return image,result


def query_record(result,index):
    extension=result['diagnostic_query_extensions'];layer=extension['apply_layer']
    return {k:v[layer,0,index].detach().float().cpu().numpy() for k,v in extension.items() if torch.is_tensor(v)}


def draw(image,record,title,path):
    cube=image.detach().float().cpu().numpy();_,height,width=cube.shape
    canvas=np.array([width,height]);points=record['candidate_points']*canvas
    background=record['background_points']*canvas
    positions=np.r_[points,background]
    lo=np.maximum(positions.min(0)-8,[0,0]);hi=np.minimum(positions.max(0)+8,[width,height])
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    def raw(ax,band=4):
        ax.imshow(cube[band],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xlim(lo[0],max(hi[0],lo[0]+1));ax.set_ylim(max(hi[1],lo[1]+1),lo[1]);ax.tick_params(labelsize=8)
    for ax,band in zip(axes[0,:3],(0,4,7)):
        raw(ax,band);ax.scatter(points[:,0],points[:,1],s=8,c='#2ce7ff',alpha=.7)
        ax.set_title(f'Band {band}: actual candidate positions')
    raw(axes[0,3]);weights=record['background_weights'];chosen=weights>0
    axes[0,3].scatter(background[~chosen,0],background[~chosen,1],marker='x',s=20,c='gray')
    scatter=axes[0,3].scatter(background[chosen,0],background[chosen,1],s=25,c=weights[chosen],cmap='viridis',vmin=0,vmax=1)
    fig.colorbar(scatter,ax=axes[0,3],fraction=.035)
    axes[0,3].set_title(f"Background support; effective n={record['effective_background_support']:.1f}")
    names=[('semantic_attention','Semantic attention',False),('background_score','Background-relative score',True),
           ('object_compatibility','Object compatibility',False),('aggregation_contribution','Actual aggregation contribution',False)]
    valid=record['candidate_valid']>.5
    for ax,(key,label,signed) in zip(axes[1],names):
        raw(ax);values=record[key];limit=max(float(np.abs(values[valid]).max()) if valid.any() else 0,1e-8)
        scatter=ax.scatter(points[valid,0],points[valid,1],s=38,c=values[valid],cmap='coolwarm' if signed else 'viridis',
            vmin=-limit if signed else 0,vmax=limit,edgecolors='black',linewidths=.25)
        fig.colorbar(scatter,ax=ax,fraction=.035);ax.set_title(label)
    fig.suptitle(title+'\nRaw input gray scale: [0,1]. Colored dots are sampled values, not segmentation or calibrated probabilities.',fontsize=11)
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/moda/spectral_full.yml')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--split',choices=('train','test'),default='train')
    parser.add_argument('--image',help='Image stem; defaults to a fixed random image')
    parser.add_argument('--query',type=int,help='Query index; defaults to highest predicted class score')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--donor-image',help='Optional different image: freeze target queries, replace only background statistics')
    parser.add_argument('--output',type=Path,default=Path('logs/moda/spectral_visualization'))
    args=parser.parse_args()
    cfg=YAMLConfig(args.config);cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.to(args.device).eval();adapter=model.decoder.query_adapter
    if adapter is None:parser.error('Choose a spectral experiment, not the baseline')
    model.load_state_dict(_checkpoint_state(args.checkpoint),strict=True)
    model.decoder.set_diagnostic_mode(True,capture_attention=False)
    dataset=(cfg.train_dataloader if args.split=='train' else cfg.val_dataloader).dataset
    stems=[p.stem for p in dataset.images]
    if args.image and args.image not in stems:parser.error('Image stem not found in this split')
    index=stems.index(args.image) if args.image else int(np.random.default_rng(args.seed).integers(len(stems)))
    image,result=run_image(model,dataset,index,args.device)
    query=int(result['pred_logits'][0].sigmoid().amax(-1).argmax()) if args.query is None else args.query
    if not 0<=query<result['pred_logits'].shape[1]:parser.error('Query index out of range')
    record=query_record(result,query)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'{stems[index]}_q{query}'
    draw(image,record,f'{stems[index]} | query {query} | {adapter.mode} | native background',path.with_suffix('.png'))
    np.savez_compressed(path.with_suffix('.npz'),image=image.numpy(),**record)
    meta={'image':stems[index],'split':args.split,'query':query,'checkpoint':str(Path(args.checkpoint).resolve()),
          'selection':'requested query or highest predicted score; no GT matching','mode':adapter.mode,
          'source_coordinates':'normalized padded canvas, multiply x by canvas width and y by height',
          'candidate_points_are_unregistered':True,'baseline_prediction_logits':result['pred_logits'][0,query].cpu().tolist()}
    if args.donor_image:
        if args.donor_image not in stems or args.donor_image==stems[index]:parser.error('Donor must be a different image in the same split')
        _,donor=run_image(model,dataset,stems.index(args.donor_image),args.device)
        donor_query=int(donor['pred_logits'][0].sigmoid().amax(-1).argmax())
        donor_record=query_record(donor,donor_query)
        adapter.background_override={
            'mean':torch.from_numpy(donor_record['background_mean']),
            'covariance':torch.from_numpy(donor_record['background_covariance']),
            'effective_support':torch.as_tensor(donor_record['effective_background_support'])}
        _,changed=run_image(model,dataset,index,args.device);changed_record=query_record(changed,query)
        # The one insertion point guarantees the same target input query and
        # sampling geometry; this validates that the intervention stays local.
        for key in ('candidate_points','semantic_attention','target_prototype','object_compatibility'):
            np.testing.assert_allclose(record[key],changed_record[key],atol=1e-6,rtol=1e-5)
        draw(image,changed_record,f'{stems[index]} | query {query} | background stats from {args.donor_image} q{donor_query}\nShown target support is NOT used for donor statistics',args.output/f'{path.name}_donor.png')
        np.savez_compressed(args.output/f'{path.name}_donor.npz',image=image.numpy(),**changed_record)
        meta['intervention']={'donor_image':args.donor_image,'donor_query':donor_query,'target_inputs_unchanged':True,
            'changed_prediction_logits':changed['pred_logits'][0,query].cpu().tolist(),
            'scope':'frozen-model mechanism intervention; not a natural-scene causal or AP result'}
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(path.with_suffix('.png'))


if __name__=='__main__':main()
