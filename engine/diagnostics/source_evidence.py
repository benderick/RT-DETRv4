"""Bounded, fixed-panel source evidence; annotations are used after inference.

The population sample is uniform over images, not over objects. The separate
class/scale gallery is explanatory, not a prevalence estimate. Every chosen
object is retained even if missed; query IDs are rematched to object identities.
"""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..misc import dist_utils
from .obb_diagnostics import _write_json


def save_npz(path, **arrays):
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(path)


def make_panel(dataset, seed=20260913, population_images=96, gallery_per_class=2):
    if population_images<0 or gallery_per_class<1:
        raise ValueError('Invalid source-panel sample counts')
    names=[p.stem for p in dataset.images]
    rng=np.random.default_rng(seed)
    population=sorted(rng.choice(len(names),min(population_images,len(names)),replace=False).tolist())
    candidates={i:[] for i in range(len(dataset.classes))}
    for index in range(len(dataset)):
        truth=dataset.get_ground_truth(index)
        for gt,(box,label) in enumerate(zip(truth['boxes'],truth['labels'])):
            candidates[int(label)].append((float(box[2]*box[3]),names[index],index,gt))
    gallery=[]
    for label,objects in candidates.items():
        ordered=sorted(objects)
        # Equal-count area strata within class, followed by deterministic hash
        # sampling. Neither model scores nor manual visibility enter selection.
        for stratum,indices in enumerate(np.array_split(np.arange(len(ordered)),gallery_per_class)):
            if not len(indices):continue
            chosen=min((ordered[int(i)] for i in indices),key=lambda x:
                hashlib.sha256(f'{seed}:{x[1]}:{x[3]}'.encode()).hexdigest())
            area,name,index,gt=chosen
            gallery.append(dict(image_id=index,image_name=name,gt_index=gt,class_name=dataset.classes[label],
                                label=label,area_px=area,scale_stratum=stratum))
    return dict(schema_version='source-panel-v1',seed=seed,population_image_ids=population,
        population_image_names=[names[i] for i in population],gallery=gallery,
        population_sampling='uniform without replacement over training images; all matched objects retained',
        gallery_sampling='deterministic hash within equal-count object-area strata per class',
        dataset_root=str(dataset.root),image_count=len(names),classes=list(dataset.classes),
        image_inventory_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
        source_annotation_sha256=dataset.get_dataset_provenance().get('source_annotation_sha256'))


def render_record(image, record, output, title, gt_box=None, limits=None, display_bands=(4,2,1)):
    """Source-pixel panels with fixed numeric scales for matched comparisons."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..rtv4.rotated_box_ops import rbox_to_corners
    image=np.asarray(image);_,height,width=image.shape
    canvas=np.array([width,height])
    points=record['attention_points']*canvas
    positions=points.reshape(-1,2)
    polygon=None
    if gt_box is not None:
        polygon=rbox_to_corners(torch.as_tensor(gt_box,dtype=torch.float32)[None],normalized_angle=False)[0].numpy()
        positions=np.concatenate((positions,polygon))
    lower=np.clip(positions.min(0)-8,[0,0],[width-1,height-1])
    upper=np.clip(positions.max(0)+8,lower+1,[width,height])
    def source(ax,band=None):
        if band is None:
            ax.imshow(image[list(display_bands)].transpose(1,2,0),vmin=0,vmax=1,interpolation='nearest')
        else:
            ax.imshow(image[band],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(upper[1],lower[1]);ax.tick_params(labelsize=7)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');source(ax)
    if polygon is not None:
        outline=np.concatenate((polygon,polygon[:1]));ax.plot(*outline.T,color='#00ffff',linewidth=1)
    ax.set_title('Pseudo RGB: R=B%d, G=B%d, B=B%d'%tuple(display_bands))
    fig.suptitle(title,fontsize=9)
    fig.savefig(output.with_name(output.name+'_rgb.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
    for band,ax in enumerate(axes.flat):
        source(ax,band);ax.set_title(f'B{band}: observed input')
        if polygon is not None:
            outline=np.concatenate((polygon,polygon[:1]));ax.plot(*outline.T,color='#00ffff',linewidth=.7)
    fig.suptitle(title+'\nAnnotation contour is for identification only; band coordinates are not realigned.',fontsize=10)
    fig.savefig(output.with_name(output.name+'_bands.png'),dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
    for ax in axes:source(ax)
    axes[0].set_title('Observed pseudo RGB')
    axes[1].set_title('Decoder cross-attention samples')
    values=np.asarray(record['attention_weights']).reshape(-1)
    scatter=axes[1].scatter(*points.reshape(-1,2).T,c=values,s=18,vmin=0,vmax=1,cmap='viridis')
    fig.colorbar(scatter,ax=axes[1]);fig.suptitle(title,fontsize=10)
    fig.savefig(output.with_name(output.name+'_mechanism.png'),dpi=130);plt.close(fig)


class SourceEvidenceRecorder:
    def __init__(self, settings, root, dataset, transforms, total_epochs):
        self.settings=settings;self.enabled=bool(settings.get('enabled',False))
        if settings.get('epoch_interval',5)<1:raise ValueError('Evidence epoch_interval must be positive')
        self.root=Path(root)/'diagnostics/source_evidence';self.total_epochs=total_epochs
        self.panel=None
        if not self.enabled or not dist_utils.is_main_process():return
        self.dataset=copy.copy(dataset);self.dataset.transforms=copy.deepcopy(transforms)
        self.panel=make_panel(self.dataset,settings.get('seed',20260913),settings.get('population_images',96),
                              settings.get('gallery_per_class',2))
        path=self.root/'panel.json'
        if path.exists() and json.loads(path.read_text())!=self.panel:
            raise ValueError('Fixed evidence panel changed; use a separate run directory')
        _write_json(path,self.panel)

    def due(self,epoch):
        if not self.enabled:return False
        if epoch<0:return bool(self.settings.get('save_initial',True))
        return epoch==0 or (epoch+1)%self.settings.get('epoch_interval',5)==0 or epoch+1==self.total_epochs

    def capture(self, model, criterion, postprocessor, device, epoch, model_source='ema'):
        if not self.due(epoch):return
        # All ranks meet at an epoch boundary; only rank zero performs the
        # additional fixed-panel inference on the unwrapped, frozen model.
        if dist_utils.is_dist_available_and_initialized():torch.distributed.barrier()
        if dist_utils.is_main_process():
            owner=dist_utils.de_parallel(model);decoder=owner.decoder
            states=[(module,module.training) for module in owner.modules()]
            flags=(decoder.decoder.diagnostic_mode,decoder.decoder.diagnostic_attention_mode)
            python_rng=random.getstate();numpy_rng=np.random.get_state()
            devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
            try:
                with torch.random.fork_rng(devices=devices),torch.inference_mode():
                    owner.eval();self._capture(owner,criterion,postprocessor,device,epoch,model_source)
            finally:
                for module,training in states:module.training=training
                decoder.set_diagnostic_mode(flags[0],capture_attention=flags[1])
                random.setstate(python_rng);np.random.set_state(numpy_rng)
        if dist_utils.is_dist_available_and_initialized():torch.distributed.barrier()

    def _capture(self,model,criterion,postprocessor,device,epoch,model_source):
        started=time.perf_counter();stage='initial' if epoch<0 else f'epoch_{epoch:04d}'
        directory=self.root/stage
        if (directory/'summary.json').is_file():return
        directory.mkdir(parents=True,exist_ok=True)
        gallery={}
        for item in self.panel['gallery']:gallery.setdefault(item['image_id'],[]).append(item)
        indices=sorted(set(self.panel['population_image_ids'])|set(gallery))
        summaries=[];artifacts=[]
        for index in indices:
            image,target=self.dataset[index];name=self.dataset.images[index].stem
            device_target={k:v.to(device) if torch.is_tensor(v) else v for k,v in target.items()}
            model.decoder.set_diagnostic_mode(True,capture_attention=index in gallery)
            result=model(image[None].to(device))
            # GT is introduced only after forward, for stable object identity.
            matching=criterion.matcher(result,[device_target])['indices'][0]
            query_to_gt={int(q):int(g) for q,g in zip(*matching)}
            gt_to_query={g:q for q,g in query_to_gt.items()}
            chosen=set(query_to_gt)
            chosen.update(result['pred_logits'][0].sigmoid().amax(-1).topk(
                min(self.settings.get('query_topk',20),result['pred_logits'].shape[1])).indices.tolist())
            chosen=sorted(chosen);selected=torch.tensor(chosen,device=device,dtype=torch.long)
            archive={'query_indices':np.asarray(chosen,dtype=np.int64),
                'matched_gt_indices':np.asarray([query_to_gt.get(q,-1) for q in chosen],dtype=np.int64),
                'pred_logits':result['pred_logits'][0,selected].float().cpu().numpy(),
                'pred_boxes_pixels':postprocessor.restore_boxes(result['pred_boxes'],[device_target])[0,selected].float().cpu().numpy()}
            truth=self.dataset.get_ground_truth(index)
            archive.update(gt_boxes_pixels=truth['boxes'].numpy(),gt_labels=truth['labels'].numpy())
            for key in ('diagnostic_layer_logits','diagnostic_layer_class_logits_before_lqe','diagnostic_layer_boxes'):
                if key in result:archive[key]=result[key][:,0,selected].float().cpu().numpy().transpose(1,0,2)
            if index in gallery:
                source_dir=self.root/'sources';source_dir.mkdir(exist_ok=True)
                source_path=source_dir/f'{name}.npz'
                if not source_path.exists():
                    save_npz(source_path,image_uint8=image.mul(255).round().byte().numpy(),
                        valid_mask=target['valid_mask'].numpy())
                locations=result.get('diagnostic_sampling_locations')
                if locations is not None:
                    layer=min(1,locations.shape[0]-1)
                    archive['attention_points']=locations[layer,0,selected].float().cpu().numpy()
                    archive['attention_weights']=result['diagnostic_sampling_attention_weights'][layer,0,selected].float().cpu().numpy()
            save_npz(directory/f'{name}.npz',**archive)
            for gt in sorted(set(range(len(truth['boxes'])))-set(gt_to_query)):
                summaries.append(dict(image_name=name,query_index=None,gt_index=gt,
                    population=index in self.panel['population_image_ids'],label=int(truth['labels'][gt]),
                    status='unmatched_no_query'))
            for row,q in enumerate(chosen):
                gt=query_to_gt.get(q,-1)
                item=dict(image_name=name,query_index=q,gt_index=gt,population=index in self.panel['population_image_ids'],
                    label=int(truth['labels'][gt]) if gt>=0 else None)
                if gt>=0:
                    label=int(truth['labels'][gt])
                    item['matched_class_score']=float(result['pred_logits'][0,q,label].sigmoid())
                summaries.append(item)
            for item in gallery.get(index,[]):
                q=gt_to_query.get(item['gt_index'])
                if q is None:
                    artifacts.append({**item,'status':'unmatched_no_query'});continue
                row=chosen.index(q);record={k:v[row] for k,v in archive.items()
                    if k not in {'query_indices','matched_gt_indices','gt_boxes_pixels','gt_labels'}}
                title=f'{stage} | {model_source} | {name} | {item["class_name"]} GT{item["gt_index"]} -> q{q}'
                title+=f' | class score {float(result["pred_logits"][0,q,item["label"]].sigmoid()):.3f}'
                stem=f'{name}_gt{item["gt_index"]}'
                if self.settings.get('render',True):
                    canvas_gt=truth['boxes'][item['gt_index']].clone()
                    canvas_gt[:2]=canvas_gt[:2]*target['scale_factor']+target['padding'][:2]
                    canvas_gt[2:4]*=target['scale_factor'].mean()
                    render_record(image.numpy(),record,directory/stem,title,canvas_gt.numpy(),
                        display_bands=self.settings.get('pseudo_rgb_bands',(4,2,1)))
                artifacts.append({**item,'query_index':q,'row':row,'archive':f'{name}.npz',
                                  'rgb':stem+'_rgb.png','bands':stem+'_bands.png','mechanism':stem+'_mechanism.png'})
            _write_json(directory/f'{name}.json',dict(image_name=name,image_id=index,
                source_path=str(self.dataset.images[index]),source_sha256=hashlib.sha256(self.dataset.images[index].read_bytes()).hexdigest(),
                canvas_hw=list(image.shape[-2:]),scale_factor=target['scale_factor'],padding=target['padding'],
                box_normalization_size=target.get('box_normalization_size',target['size']),
                method='baseline', schema='decoder-attention-v1',
                model_source=model_source,epoch=epoch,selection='all Hungarian matches plus top class-score queries; matching only after inference'))
        _write_json(directory/'population.json',summaries)
        _write_json(directory/'gallery.json',artifacts)
        html=['<!doctype html><meta charset="utf-8"><title>Source evidence</title><h1>'+stage+'</h1>',
              '<p>Fixed class/scale gallery; annotations identify objects only. Quantitative population uses a separate random image sample.</p>']
        for item in artifacts:
            html.append(f'<h2>{item["class_name"]}: {item["image_name"]}, GT {item["gt_index"]}</h2>')
            if 'mechanism' in item:
                html.extend(f'<a href="{item[key]}"><img loading="lazy" style="width:100%;max-width:1300px" src="{item[key]}"></a>' for key in ('rgb','bands','mechanism'))
            else:html.append('<p>Unmatched: no query assigned; retained as a failure.</p>')
        (directory/'index.html').write_text('\n'.join(html))
        _write_json(directory/'summary.json',dict(epoch=epoch,model_source=model_source,images=len(indices),
            query_records=sum(row['query_index'] is not None for row in summaries),
            unmatched_objects=sum(row['query_index'] is None for row in summaries),
            gallery_objects=len(artifacts),seconds=time.perf_counter()-started,
            scope='fixed training subset prediction observation; no validation AP or calibrated pixel truth'))
        print(f'Source evidence {stage}: {len(indices)} images, {time.perf_counter()-started:.1f}s -> {directory}',flush=True)
