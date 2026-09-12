import ast
from pathlib import Path
import types
import unittest

import numpy as np
import torch

from engine.evaluation.obb.benchmark import (
    BenchmarkOBBEvaluator, benchmark_ap, benchmark_matches, pairwise_probiou, probiou_fast_nms)
from engine.rtv4.rotated_box_ops import rotated_iou
from engine.rtv4.rotated_postprocessor import RotatedPostProcessor


REFERENCE = Path("research/references/FressDet/code/FressDet/ultralytics")


def reference_functions(path, names):
    """Read only selected numerical definitions, without importing Ultralytics."""
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            functions.extend(item for item in node.body if isinstance(item,ast.FunctionDef) and item.name in names)
    namespace = dict(torch=torch,np=np)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


class BenchmarkProtocolTest(unittest.TestCase):
    def test_probiou_changes_threshold_decisions_from_geometric_iou(self):
        a=torch.tensor([[0.,0.,10.,10.,0.]])
        b=torch.tensor([[4.,0.,10.,10.,0.]])
        self.assertLess(float(rotated_iou(a,b,model_space=False)),.5)
        self.assertGreater(float(pairwise_probiou(a,b)),.5)
        self.assertEqual(pairwise_probiou(a[:0],b).shape,(0,1))

    def test_ap_interpolation_and_fast_nms_suppression_chain(self):
        # Released trapezoidal AP gives .995, not the DOTA perfect AP of 1.
        self.assertAlmostEqual(benchmark_ap(np.array([1.]),np.array([1.])),.995)
        boxes=torch.tensor([[0.,0.,10.,10.,0.],[2.,0.,10.,10.,0.],[4.,0.,10.,10.,0.]])
        keep,status,parent,_=probiou_fast_nms(boxes,torch.tensor([.9,.8,.7]),torch.zeros(3,dtype=torch.long),.7)
        self.assertEqual(keep.tolist(),[0])
        self.assertEqual(parent[2].item(),1)  # removed box 1 still suppresses 2
        self.assertEqual(status.tolist(),[0,1,1])
        outputs=dict(pred_boxes=torch.tensor([[[.5,.5,.4,.2,0.]]]),pred_logits=torch.zeros(1,1,1))
        for method, count in (("geometric",1),("probiou_fast",0)):
            post=RotatedPostProcessor(num_classes=1,score_threshold=.5,nms_method=method)
            self.assertEqual(len(post(outputs,torch.tensor([[10.,10.]]))[0]["boxes"]),count)

    def test_evaluator_is_complete_and_reports_both_protocols(self):
        class Dataset:
            classes=("car",)
            def __len__(self): return 1
            def get_ground_truth(self,index):
                return dict(boxes=torch.tensor([[0.,0.,10.,10.,0.]]),labels=torch.tensor([0]),difficulty=torch.tensor([0]))
        evaluator=BenchmarkOBBEvaluator(Dataset())
        with self.assertRaisesRegex(ValueError,"every image"):
            evaluator.accumulate(False)
        evaluator.update({0:dict(boxes=torch.tensor([[4.,0.,10.,10.,0.]]),scores=torch.tensor([.9]),labels=torch.tensor([0]))})
        evaluator.accumulate(False)
        self.assertAlmostEqual(evaluator.metrics["AP50"],.995)
        self.assertEqual(evaluator.metrics["riou_AP50"],0.)
        self.assertEqual(evaluator.selection_index,1)
        clone=evaluator.clone_empty()
        self.assertEqual(clone.protocol,evaluator.protocol)
        self.assertFalse(clone.predictions)

    @unittest.skipUnless(REFERENCE.is_dir(),"local FressDet reference not available")
    def test_differential_against_released_fressdet_functions(self):
        reference=reference_functions(REFERENCE/"utils/metrics.py",{"_get_covariance_matrix","batch_probiou","compute_ap"})
        matcher=reference_functions(REFERENCE/"engine/validator.py",{"match_predictions"})["match_predictions"]
        torch.manual_seed(7)
        first=torch.rand(31,5)*20
        second=torch.rand(47,5)*20
        actual=pairwise_probiou(first,second)
        torch.testing.assert_close(actual,reference["batch_probiou"](first,second),atol=1e-6,rtol=1e-6)
        nms_reference=reference_functions(REFERENCE/"utils/ops.py",{"nms_rotated"})
        nms_reference["batch_probiou"]=reference["batch_probiou"]
        scores=torch.linspace(.1,.9,len(second))
        expected_keep=nms_reference["nms_rotated"](second,scores,.7)
        actual_keep=probiou_fast_nms(second,scores,torch.zeros(len(second),dtype=torch.long),.7)[0]
        torch.testing.assert_close(actual_keep,expected_keep,atol=0,rtol=0)
        thresholds=torch.linspace(.5,.95,10)
        gt=torch.randint(3,(31,));pred=torch.randint(3,(47,))
        expected=matcher(types.SimpleNamespace(iouv=thresholds),pred,gt,actual).numpy()
        np.testing.assert_array_equal(benchmark_matches(actual,gt,pred,thresholds.tolist()),expected)
        for tp in (np.array([1,0,1,1]),np.array([0,0,0,0]),np.array([1])):
            recall=tp.cumsum()/4
            precision=tp.cumsum()/np.arange(1,len(tp)+1)
            self.assertEqual(benchmark_ap(recall,precision),reference["compute_ap"](recall,precision)[0])


if __name__ == "__main__": unittest.main()
