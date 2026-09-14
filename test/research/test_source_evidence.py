import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from engine.data.dataset.moda_dataset import MODADetection
from engine.data.transforms.rotated_transforms import RotatedConvertToTensor
from engine.diagnostics import OBBDiagnostics
from engine.diagnostics.source_evidence import SourceEvidenceRecorder,make_panel,render_record
from engine.evaluation.obb.benchmark import BenchmarkOBBEvaluator
from engine.rtv4.rotated_postprocessor import RotatedPostProcessor
from engine.solver.det_engine import evaluate
from test.research.test_spectral_evidence import tiny_spectral_model,tiny_batch
from test.framework.model.test_model_pipeline import build_criterion
from tools.analysis.replay_moda_postprocess import archived_images,select


class TensorTransform:
    def __call__(self,image,target,dataset):
        return RotatedConvertToTensor()((image,target,dataset))


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
            settings=dict(enabled=True,population_images=2,gallery_per_class=2,render=False,query_topk=1)
            recorder=SourceEvidenceRecorder(settings,root/'run',dataset,dataset.transforms,20)
            model=tiny_spectral_model().train();before={k:v.clone() for k,v in model.state_dict().items()}
            random_before=torch.get_rng_state().clone()
            def guard(module,args,kwargs):
                self.assertEqual(set(kwargs['targets'][0]),{'valid_mask'})
            hook=model.register_forward_pre_hook(guard,with_kwargs=True)
            post=RotatedPostProcessor(num_classes=3,num_top_queries=300,apply_nms=False,
                score_threshold=.01,nms_method='probiou_fast')
            recorder.capture(model,build_criterion(),post,torch.device('cpu'),-1,'model')
            hook.remove();torch.testing.assert_close(torch.get_rng_state(),random_before)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
            self.assertTrue(model.training);self.assertFalse(model.decoder.decoder.diagnostic_mode)
            stage=root/'run/diagnostics/source_evidence/initial'
            gallery=json.loads((stage/'gallery.json').read_text())
            self.assertEqual([g['gt_index'] for g in gallery],[0,0])
            archive=np.load(stage/gallery[0]['archive']);row=gallery[0]['row']
            for key in ('candidate_embeddings','background_embeddings','query_before','query_after','target_prototype'):
                self.assertIn(key,archive.files)
            self.assertFalse(any(recorder.due(e) for e in (1,2,3,5,6)))
            self.assertTrue(all(recorder.due(e) for e in (-1,0,4,9,14,19)))
            data=np.load(root/'run/diagnostics/source_evidence/sources/0.npz')
            record={k:archive[k][row] for k in archive.files if k not in ('gt_boxes_pixels','gt_labels')}
            render_record(data['image_uint8']/255.,record,stage/'render_smoke','Untrained tool check')
            self.assertTrue((stage/'render_smoke_mechanism.png').is_file())
            baseline=copy.deepcopy(model);baseline.decoder.query_adapter=None
            base_recorder=SourceEvidenceRecorder(settings,root/'base',dataset,dataset.transforms,20)
            base_recorder.capture(baseline,build_criterion(),post,torch.device('cpu'),0,'model')
            base_archive=np.load(root/'base/diagnostics/source_evidence/epoch_0000/0.npz')
            self.assertIn('attention_points',base_archive.files)

    def test_training_summary_is_detached_separates_dn_and_preserves_outputs(self):
        model=tiny_spectral_model().train();images,targets=tiny_batch()
        torch.manual_seed(12);plain=model(images,targets)
        model.decoder.collect_train_diagnostics=True
        torch.manual_seed(12);observed=model(images,targets)
        torch.testing.assert_close(plain['pred_logits'],observed['pred_logits'],rtol=0,atol=0)
        summary=observed['method_train_diagnostics']['spectral_evidence']
        self.assertEqual(set(summary['groups']),{'ordinary','denoising'})
        for group in summary['groups'].values():
            for values in group.values():
                for value in values.values():self.assertFalse(value.requires_grad)
        loss=sum(build_criterion()(observed,targets).values());loss.backward()

    def test_compact_eval_preserves_primary_ap_and_archives_replayable_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=dataset_at(root);model=tiny_spectral_model().eval()
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
