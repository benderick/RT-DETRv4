import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from engine.data.dataset.moda_dataset import MODADetection
from engine.data.transforms.rotated_transforms import RotatedConvertToTensor
from engine.diagnostics import OBBDiagnostics
from engine.diagnostics.source_evidence import SourceEvidenceRecorder,make_panel,render_record
from engine.evaluation.obb.benchmark import BenchmarkOBBEvaluator
from engine.rtv4 import RTv4
from engine.rtv4.rotated_postprocessor import RotatedPostProcessor
from engine.solver.det_engine import evaluate
from test.framework.model.test_model_pipeline import build_criterion,build_tiny_model
from tools.analysis.replay_moda_postprocess import archived_images,select


class TensorTransform:
    def __call__(self,image,target,dataset):
        return RotatedConvertToTensor()((image,target,dataset))


class TinyBaselineBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(8,32,3,stride=8,padding=1)

    def forward(self,images):
        first=self.conv(images)
        return [first,F.avg_pool2d(first,2),F.avg_pool2d(first,4)]


def tiny_baseline_model():
    return RTv4(TinyBaselineBackbone(),nn.Identity(),build_tiny_model())


def dataset_at(root):
    (root/'images').mkdir();(root/'labels').mkdir()
    for i in range(3):
        np.save(root/f'images/{i}.npy',np.random.default_rng(i).integers(0,256,(8,96,64),dtype=np.uint8))
        (root/f'labels/{i}.txt').write_text('20 24 44 24 44 40 20 40 car 2\n' if i<2 else '')
    return MODADetection(root,transforms=TensorTransform())


class SourceEvidenceTest(unittest.TestCase):
    def test_fixed_population_gallery_and_observation_preserves_model_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=dataset_at(root)
            a=make_panel(dataset,population_images=2,gallery_per_class=2)
            b=make_panel(dataset,population_images=2,gallery_per_class=2)
            self.assertEqual(a,b);self.assertEqual(len(a['gallery']),2)
            settings=dict(enabled=True,population_images=2,gallery_per_class=2,render=False,query_topk=1,
                pseudo_rgb_bands=(4,2,1))
            recorder=SourceEvidenceRecorder(settings,root/'run',dataset,dataset.transforms,20)
            model=tiny_baseline_model().train()
            before={k:v.clone() for k,v in model.state_dict().items()}
            random_before=torch.get_rng_state().clone()
            post=RotatedPostProcessor(num_classes=3,num_top_queries=300,apply_nms=False,
                score_threshold=.01,nms_method='probiou_fast')
            recorder.capture(model,build_criterion(),post,torch.device('cpu'),-1,'model')
            torch.testing.assert_close(torch.get_rng_state(),random_before)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
            self.assertTrue(model.training);self.assertFalse(model.decoder.decoder.diagnostic_mode)
            stage=root/'run/diagnostics/source_evidence/initial'
            gallery=json.loads((stage/'gallery.json').read_text())
            self.assertEqual([g['gt_index'] for g in gallery],[0,0])
            archive=np.load(stage/gallery[0]['archive']);row=gallery[0]['row']
            for key in ('attention_points','attention_weights','diagnostic_layer_logits','diagnostic_layer_boxes'):
                self.assertIn(key,archive.files)
            self.assertFalse(any(recorder.due(e) for e in (1,2,3,5,6)))
            self.assertTrue(all(recorder.due(e) for e in (-1,0,4,9,14,19)))
            data=np.load(root/'run/diagnostics/source_evidence/sources/0.npz')
            record={k:archive[k][row] for k in archive.files if k not in ('gt_boxes_pixels','gt_labels')}
            render_record(data['image_uint8']/255.,record,stage/'render_smoke','Untrained tool check')
            self.assertTrue((stage/'render_smoke_mechanism.png').is_file())
            self.assertTrue((stage/'render_smoke_rgb.png').is_file())

    def test_training_preserves_outputs_and_backward(self):
        model=tiny_baseline_model().train()
        images=torch.randn(2,8,64,96)
        targets=[{"labels":torch.tensor([0]),"boxes":torch.tensor([[.5,.5,.2,.1,.1]])},
                 {"labels":torch.tensor([1]),"boxes":torch.tensor([[.3,.4,.1,.2,.2]])}]
        outputs=model(images,targets)
        loss=sum(build_criterion()(outputs,targets).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_compact_eval_preserves_primary_ap_and_archives_replayable_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=dataset_at(root);model=tiny_baseline_model().eval()
            batches=[(im[None],[target]) for im,target in [dataset[i] for i in range(len(dataset))]]
            post=RotatedPostProcessor(num_classes=3,num_top_queries=300,apply_nms=False,
                score_threshold=.01,nms_method='probiou_fast')
            evaluator=BenchmarkOBBEvaluator(dataset,compute_geometric=False)
            cfg=SimpleNamespace(diagnostics_enabled=True,diagnostics_eval_records='compact',
                diagnostics_detailed_image_limit=0,diagnostics_layerwise_epoch_interval=0,resume=None)
            diagnostics=OBBDiagnostics(cfg,root/'run')
            with mock.patch.object(diagnostics,'record_evaluation_batch',side_effect=AssertionError('heavy record called')), \
                 mock.patch('engine.evaluation.obb.benchmark.rotated_iou',side_effect=AssertionError('geometric AP called')):
                evaluate(model,build_criterion(),post,batches,evaluator,torch.device('cpu'),diagnostics=diagnostics,epoch=0)
            diagnostics.close()
            stage=root/'run/diagnostics/eval/epoch_0000'
            self.assertFalse(list(stage.glob('nms.*')))
            self.assertTrue((stage/'paper_metrics.csv').exists())
            replay=list(archived_images(stage));self.assertEqual(len(replay),3)
            replay_evaluator=BenchmarkOBBEvaluator(dataset,compute_geometric=False)
            for index,candidates in replay:
                # This test has 20x3 query/class entries; archived values must
                # exactly reproduce a postprocessor with all candidates.
                prediction=select(candidates,'detr')
                self.assertLessEqual(len(prediction['scores']),300)
                for key in prediction:torch.testing.assert_close(prediction[key],evaluator.predictions[index][key],rtol=0,atol=0)
            direct=BenchmarkOBBEvaluator(dataset,compute_geometric=True)
            direct.update(evaluator.predictions);direct.accumulate(False)
            for key,value in evaluator.metrics.items():self.assertEqual(value,direct.metrics[key])
            self.assertNotIn('riou_AP50',evaluator.metrics)
            totals=json.loads((stage/'summary.rank000.json').read_text())['performance_totals_seconds']
            self.assertIn('prediction_archive',totals)


if __name__=='__main__':unittest.main()
