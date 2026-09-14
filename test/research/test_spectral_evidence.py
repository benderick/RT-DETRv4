import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from engine.core import YAMLConfig
from engine.data.dataset.moda_dataset import MODADetection
from engine.data.transforms.rotated_transforms import RotatedConvertToTensor
from engine.diagnostics import OBBDiagnostics
from engine.evaluation.obb.dota import DotaOBBEvaluator
from engine.rtv4 import RotatedPostProcessor
from engine.rtv4.rtv4 import RTv4
from engine.solver.det_engine import evaluate
from engine.rtv4.spectral_evidence import (
    BackgroundConditionedSpectralEvidence, background_statistics, sampling_support,
)
from test.framework.model.test_model_pipeline import build_tiny_model, build_criterion


class TinySpectralBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8,32,3,stride=8,padding=1)

    def forward(self, images):
        first = self.conv(images)
        return [first,F.avg_pool2d(first,2),F.avg_pool2d(first,4)]


def tiny_spectral_model(mode="full", refinement_mode="o2_adr"):
    decoder = build_tiny_model(refinement_mode=refinement_mode)
    decoder.query_adapter = BackgroundConditionedSpectralEvidence(
        hidden_dim=32,band_dim=2,mode=mode,candidate_side=5,background_side=7,
        query_chunk_size=7,box_coordinate_mode="per_axis")
    return RTv4(TinySpectralBackbone(),nn.Identity(),decoder)


def tiny_batch():
    return torch.rand(2,8,64,96),[
        dict(labels=torch.tensor([1]),boxes=torch.tensor([[.5,.5,.4,.2,.2]]),valid_mask=torch.ones(64,96,dtype=torch.bool)),
        dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,5),valid_mask=torch.ones(64,96,dtype=torch.bool))]


class SpectralEvidenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.rand(2,8,64,96)
        self.queries = torch.rand(2,3,32)
        self.boxes = torch.tensor([[[.5,.4,.4,.15,.13],[.2,.3,.15,.1,.8],[.75,.5,.15,.1,.4]]]).repeat(2,1,1)

    def adapter(self, **kwargs):
        return BackgroundConditionedSpectralEvidence(hidden_dim=32,band_dim=2,**kwargs)

    def context(self, model, images=None, targets=None):
        context = model.build_context(self.images if images is None else images,targets,diagnostics=True)
        context['encoder_objectness'] = torch.zeros(2,1,8,12)
        return context

    def test_weighted_covariance_against_numpy_and_correlated_background(self):
        values = torch.tensor([[[[0.,0.],[1.,1.1],[2.,1.9],[3.,3.1],[1.,.8]]]])
        weights = torch.tensor([[[1.,.2,.8,.5,.4]]])
        mean,cov,effective = background_statistics(values,weights)
        x,w = values[0,0].numpy().astype(float),weights[0,0].numpy().astype(float)
        expected_mean = np.average(x,axis=0,weights=w)
        expected_cov = np.cov(x,rowvar=False,aweights=w)
        expected_cov = .8*expected_cov+.2*np.diag(np.diag(expected_cov))+(.01*np.trace(expected_cov)/2+.001)*np.eye(2)
        np.testing.assert_allclose(mean[0,0].numpy(),expected_mean,rtol=1e-6)
        np.testing.assert_allclose(cov[0,0].numpy(),expected_cov,rtol=2e-6)
        self.assertAlmostEqual(float(effective),float(w.sum()**2/(w*w).sum()),places=5)
        # A background common direction should be discounted relative to an
        # independent target contrast; diagonal statistics cannot rotate it.
        d = torch.tensor([1.,0.])
        full = torch.linalg.solve(cov[0,0],d)
        diagonal = d/cov[0,0].diag()
        self.assertLess(float(full[1]),0.)
        quality = lambda a: (a@d).square()/(a@cov[0,0]@a)
        self.assertGreater(float(quality(full)),float(quality(diagonal)))
        for weight in (torch.zeros_like(weights),torch.tensor([[[1.,0,0,0,0]]])):
            m,c,n = background_statistics(values,weight)
            self.assertTrue(torch.isfinite(m).all() and torch.isfinite(c).all())
            self.assertTrue((torch.linalg.eigvalsh(c)>0).all())

    def test_canvas_sampling_and_equivalent_box_charts(self):
        box = torch.tensor([[[.5,.25,.4,.2,0.]]])
        points,background,outside = sampling_support(box,48,96,candidate_side=3,background_side=5)
        # Isotropic center=(48,24), width=38.4, height=19.2.
        self.assertAlmostEqual(float(points[0,0,4,0]*96),48.,places=5)
        self.assertAlmostEqual(float(points[0,0,4,1]*48),24.,places=5)
        self.assertFalse(bool(outside[0,0,12]))
        axis = box.clone();axis[..., [1,3]]*=2
        other,_,_=sampling_support(axis,48,96,3,5,box_coordinate_mode='per_axis')
        torch.testing.assert_close(points,other)
        model=self.adapter();nn.init.normal_(model.readout.weight,std=.02)
        baseline=model(self.queries,self.boxes,self.context(model),1)
        for swapped in (False,True):
            alternative=self.boxes.clone()
            if swapped:
                alternative[..., [2,3]]=self.boxes[..., [3,2]];alternative[...,4]+=.5
            else:alternative[...,4]+=1
            result=model(self.queries,alternative,self.context(model),1)
            torch.testing.assert_close(result,baseline,atol=1e-6,rtol=1e-4)

    def test_padding_annotations_and_low_support(self):
        model=self.adapter();nn.init.normal_(model.readout.weight,std=.02)
        valid=torch.ones(2,64,96,dtype=torch.bool);valid[:,:,72:]=False
        targets=[dict(valid_mask=v,boxes=torch.rand(5,5),labels=torch.arange(5)) for v in valid]
        first=model(self.queries,self.boxes,self.context(model,targets=targets),1)
        altered=self.images.clone();altered[:,:,:,72:]=1e8
        for t in targets:t['boxes'].fill_(float('nan'));t['labels'].fill_(-123)
        second=model(self.queries,self.boxes,self.context(model,altered,targets),1)
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        for t in targets:t['valid_mask'].zero_()
        context=self.context(model,targets=targets)
        empty=model(self.queries,self.boxes,context,1)
        self.assertEqual(float(empty.abs().max()),0.)
        self.assertTrue(context['records'][1]['diagonal_fallback'].all())
        self.assertEqual(float(context['records'][1]['aggregation_contribution'].abs().max()),0.)

    def test_gate_bound_background_intervention_and_chunk_invariance(self):
        model=self.adapter(query_chunk_size=1).eval();nn.init.normal_(model.readout.weight,std=.02)
        ctx=self.context(model);original=model(self.queries,self.boxes,ctx,1)
        record=ctx['records'][1]
        self.assertTrue((record['evidence_gate']<=record['object_compatibility']+1e-7).all())
        self.assertTrue((record['aggregation_contribution'].sum(-1)<=1+1e-6).all())
        model.query_chunk_size=99
        other=model(self.queries,self.boxes,self.context(model),1)
        torch.testing.assert_close(original,other,rtol=2e-5,atol=2e-7)
        model.background_override={'mean':torch.ones(8)*2,'covariance':torch.eye(8),'effective_support':torch.tensor(50.)}
        changed_context=self.context(model);changed=model(self.queries,self.boxes,changed_context,1)
        self.assertGreater(float((original-changed).abs().max()),1e-6)
        torch.testing.assert_close(record['object_compatibility'],changed_context['records'][1]['object_compatibility'])
        model.train()
        with self.assertRaisesRegex(ValueError,'evaluation-only'):model(self.queries,self.boxes,self.context(model),1)

    def test_modes_hold_semantic_attention_and_compatibility_fixed(self):
        shared=self.adapter()
        records={}
        for mode in ('plain','diagonal','no_compatibility','full'):
            model=self.adapter(mode=mode);model.load_state_dict(shared.state_dict(),strict=True)
            ctx=self.context(model);model(self.queries,self.boxes,ctx,1);records[mode]=ctx['records'][1]
        for mode,r in records.items():
            torch.testing.assert_close(r['semantic_attention'],records['full']['semantic_attention'])
            torch.testing.assert_close(r['object_compatibility'],records['full']['object_compatibility'])
        torch.testing.assert_close(records['plain']['evidence_gate'],torch.ones_like(records['plain']['evidence_gate']))
        self.assertTrue((records['full']['evidence_gate']<=records['no_compatibility']['evidence_gate']+1e-7).all())
        diagonal=records['diagonal']['background_covariance']
        torch.testing.assert_close(diagonal,torch.diag_embed(diagonal.diagonal(dim1=-2,dim2=-1)))

    def test_end_to_end_zero_initialization_training_and_classification_path(self):
        for refinement in ('o2_adr','direct_angle'):
            model=tiny_spectral_model(refinement_mode=refinement)
            baseline=copy.deepcopy(model);baseline.decoder.query_adapter=None
            images,targets=tiny_batch()
            torch.manual_seed(11);before=baseline(images,targets)
            torch.manual_seed(11);after=model(images,targets)
            for key in ('pred_boxes','pred_logits'):
                torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
            criterion=build_criterion(refinement);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                result=model(images,targets);self.assertIn('dn_outputs',result)
                loss=sum(criterion(result,targets).values());self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name,p in model.named_parameters():
                    if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
                optimizer.step()
            for name in ('band_stem.0.weight','target.weight','semantic_query.weight','compatibility_scale.weight','readout.weight'):
                gradient=dict(model.decoder.query_adapter.named_parameters())[name].grad
                self.assertIsNotNone(gradient,name);self.assertGreater(float(gradient.abs().sum()),0.,name)
            model.eval();control=copy.deepcopy(model);control.decoder.query_adapter=None
            model.decoder.set_diagnostic_mode(True,capture_attention=False)
            with torch.inference_mode():
                result=model(images,targets);plain=control(images,targets)
                extension=result['diagnostic_query_extensions']
                self.assertEqual(extension['candidate_points'].shape,(2,2,20,25,2))
                self.assertFalse(extension['active'][0].any());self.assertTrue(extension['active'][1].all())
                self.assertGreater(float((result['pred_logits']-plain['pred_logits']).abs().max()),1e-6)
                for t in targets:t['boxes']=torch.full((10,5),float('nan'));t['labels']=torch.full((10,),-99)
                independent=model(images,targets)
                torch.testing.assert_close(result['pred_logits'],independent['pred_logits'],atol=0,rtol=0)

    def test_config_opt_in_same_shared_initialization_and_protocol(self):
        torch.manual_seed(23);base=YAMLConfig('configs/experiments/moda/dfine_obb_o2_fressdet.yml')
        base_model=base.model
        self.assertIsNone(base_model.decoder.query_adapter)
        states=[]
        for mode in ('full','plain','diagonal','no_compatibility'):
            torch.manual_seed(23);cfg=YAMLConfig(f'configs/experiments/moda/spectral_{mode}.yml');model=cfg.model
            for key,value in base_model.state_dict().items():
                torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
            states.append({k:tuple(v.shape) for k,v in model.decoder.query_adapter.state_dict().items()})
            for key in ('epoches','eval_spatial_size','optimizer','lr_scheduler','evaluator','train_dataloader','val_dataloader'):
                self.assertEqual(base.yaml_cfg[key],cfg.yaml_cfg[key],key)
            self.assertEqual(model.decoder.query_adapter.mode,mode)
        self.assertTrue(all(x==states[0] for x in states))

    def test_evaluation_serializes_actual_query_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'images').mkdir();(root/'labels').mkdir()
            np.save(root/'images/a.npy',np.full((8,96,64),120,dtype=np.uint8))
            (root/'labels/a.txt').write_text('20 24 44 24 44 40 20 40 car 2\n')
            dataset=MODADetection(root);image,target=dataset[0]
            image,target,_=RotatedConvertToTensor()((image,target,dataset))
            model=tiny_spectral_model().eval()
            cfg=SimpleNamespace(diagnostics_enabled=True,diagnostics_detailed_image_limit=1,
                diagnostics_query_topk=1,diagnostics_layerwise_epoch_interval=1,
                yaml_cfg={'RotatedDFINETransformer':{'refinement_mode':'o2_adr'}},resume=None)
            diagnostics=OBBDiagnostics(cfg,root/'run',model=model,train_dataset=dataset)
            try:
                evaluate(model,build_criterion(),RotatedPostProcessor(num_classes=3,num_top_queries=20),
                    [(image[None],[target])],DotaOBBEvaluator(dataset),torch.device('cpu'),diagnostics=diagnostics,epoch=0)
            finally:diagnostics.close()
            path=next((root/'run/diagnostics/eval/epoch_0000').glob('queries.rank*.jsonl.gz'))
            with gzip.open(path,'rt') as handle:rows=[json.loads(line) for line in handle]
            active=[s['method_diagnostics'] for r in rows for s in r['stages']
                    if s.get('method_diagnostics',{}).get('active')]
            self.assertTrue(active)
            self.assertEqual(active[0]['schema_version'],'spectral-evidence-v1')
            self.assertEqual(np.asarray(active[0]['candidate_points']).shape,(25,2))
            self.assertEqual(np.asarray(active[0]['background_covariance']).shape,(8,8))


if __name__=='__main__':unittest.main()
