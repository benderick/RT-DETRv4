import copy
import gzip
import json
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
from engine.rtv4.obb.incubator.bcsr import BoundarySpectralRefinement
from engine.rtv4.obb.incubator.bcsr.adapter import boundary_points
from test.framework.model.test_model_pipeline import build_tiny_model, build_criterion


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8,32,3,stride=8,padding=1)
    def forward(self, images):
        first = self.conv(images)
        return [first, F.avg_pool2d(first,2), F.avg_pool2d(first,4)]


class BCSRTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.images = torch.rand(2,8,48,80)
        self.queries = torch.randn(2,3,32)
        self.boxes = torch.tensor([[[.5,.5,.5,.25,.13], [.25,.3,.12,.08,.95], [.7,.7,.25,.17,.5]]]).repeat(2,1,1)

    def adapter(self, **kwargs):
        return BoundarySpectralRefinement(hidden_dim=32, band_dim=4, routing_dim=8, **kwargs)

    def activate(self, model):
        nn.init.normal_(model.readout[-1].weight, std=.02)
        nn.init.normal_(model.readout[-1].bias, std=.02)

    def run_adapter(self, model, boxes=None, images=None, targets=None):
        context = model.build_context(self.images if images is None else images, targets, diagnostics=True)
        output = model(self.queries, self.boxes if boxes is None else boxes, context, 0)
        return output, context

    def test_boundary_grid_in_pixel_space_on_rectangular_canvas(self):
        boxes = torch.tensor([[[.5,.5,.5,.25,0.]]])
        points, _ = boundary_points(boxes,48,80)
        pixel = points[0,0] * torch.tensor([80,48])
        torch.testing.assert_close(pixel[0,:,0,0], torch.tensor([25.,35.,45.,55.]))
        torch.testing.assert_close(pixel[0,:,0,1], torch.full((4,),19.8))
        torch.testing.assert_close(pixel[0,:,1,1], torch.full((4,),16.2))

    def test_zero_initialization_and_equivalent_rectangle_parameterizations(self):
        model = self.adapter()
        result, _ = self.run_adapter(model)
        self.assertEqual(float(result.abs().max()), 0.)
        self.activate(model)
        baseline, _ = self.run_adapter(model)
        self.assertGreater(float(baseline.abs().max()), .001)
        halfturn = self.boxes.clone()
        halfturn[...,4] += 1
        swapped = self.boxes.clone()
        swapped[...,2] = self.boxes[...,3] * 48/80
        swapped[...,3] = self.boxes[...,2] * 80/48
        swapped[...,4] += .5
        for alternative in (halfturn, swapped):
            actual, _ = self.run_adapter(model, boxes=alternative)
            torch.testing.assert_close(baseline, actual, atol=2e-6, rtol=2e-5)

    def test_padding_values_and_ground_truth_do_not_control_routing(self):
        model = self.adapter()
        self.activate(model)
        valid = torch.ones(2,48,80,dtype=torch.bool)
        valid[:,:,60:] = False
        targets = [dict(valid_mask=v, boxes=torch.rand(5,5)) for v in valid]
        altered = self.images.clone()
        altered[:,:,:,60:] = 1e5
        a, _ = self.run_adapter(model, targets=targets)
        for target in targets:
            target["boxes"] = torch.full((5,5),float("nan"))
        b, _ = self.run_adapter(model, images=altered, targets=targets)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for target in targets:
            target["valid_mask"] = torch.zeros_like(target["valid_mask"])
        empty, context = self.run_adapter(model, targets=targets)
        self.assertEqual(float(empty.abs().max()),0.)
        self.assertEqual(float(context["records"][0]["valid_fraction"].max()),0.)

    def test_object_control_shares_weights_and_initial_control_freezes_grid(self):
        edge = self.adapter()
        obj = self.adapter(weighting="object")
        obj.load_state_dict(edge.state_dict(),strict=True)
        _, edge_context = self.run_adapter(edge)
        _, object_context = self.run_adapter(obj)
        edge_weights = edge_context["records"][0]["band_weights"]
        object_weights = object_context["records"][0]["band_weights"]
        torch.testing.assert_close(object_weights,object_weights[:,:,:1].expand_as(object_weights),atol=0,rtol=0)
        self.assertGreater(float((edge_weights[:,:,0]-edge_weights[:,:,1]).abs().max()),1e-5)
        for reference in ("iterative", "initial"):
            model = self.adapter(sampling_reference=reference)
            _, context = self.run_adapter(model)
            moved = self.boxes.clone()
            moved[...,:2] += .1
            model(self.queries,moved,context,1)
            left,right = [r["sampling_points"] for r in context["records"]]
            if reference == "initial":
                torch.testing.assert_close(left,right,atol=0,rtol=0)
            else:
                self.assertGreater(float((left-right).abs().max()),.09)

    def test_config_is_opt_in_and_controls_have_identical_parameter_shapes(self):
        states = []
        for variant in ("edge", "object", "initial"):
            torch.manual_seed(23)
            cfg = YAMLConfig(f"configs/incubator/bcsr/moda_{variant}.yml", HGNetv2={"pretrained":False})
            model = cfg.model
            states.append({k:tuple(v.shape) for k,v in model.decoder.geometry_adapter.state_dict().items()})
        self.assertEqual(states[0],states[1])
        self.assertEqual(states[0],states[2])
        torch.manual_seed(23)
        baseline = YAMLConfig("configs/experiments/moda/dfine_obb_o2.yml",HGNetv2={"pretrained":False})
        self.assertIsNone(baseline.model.decoder.geometry_adapter)
        self.assertNotIn("imports",baseline.yaml_cfg)
        for key,value in baseline.model.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)

    def test_end_to_end_training_denoising_empty_gt_and_diagnostics(self):
        for mode in ("o2_adr", "direct_angle"):
            decoder = build_tiny_model(refinement_mode=mode)
            decoder.geometry_adapter = self.adapter()
            model = RTv4(TinyBackbone(),nn.Identity(),decoder)
            baseline = copy.deepcopy(model)
            baseline.decoder.geometry_adapter = None
            images = torch.rand(2,8,64,64)
            targets = [dict(labels=torch.tensor([1]),boxes=torch.tensor([[.5,.5,.4,.2,.2]]),valid_mask=torch.ones(64,64,dtype=torch.bool)),
                       dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,5),valid_mask=torch.ones(64,64,dtype=torch.bool))]
            torch.manual_seed(2)
            before = baseline(images,targets)
            torch.manual_seed(2)
            initial = model(images,targets)
            for key in ("pred_boxes","pred_logits"):
                torch.testing.assert_close(before[key],initial[key],atol=0,rtol=0)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
            criterion = build_criterion(mode)
            for _ in range(4):
                optimizer.zero_grad()
                outputs = model(images,targets)
                self.assertIn("dn_outputs",outputs)
                loss = sum(criterion(outputs,targets).values())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all(),name)
                optimizer.step()
            for name in ("band_stem.0.weight","query.weight","key.weight","readout.2.weight"):
                gradient = dict(decoder.geometry_adapter.named_parameters())[name].grad
                self.assertIsNotNone(gradient,name)
                self.assertGreater(float(gradient.abs().sum()),0.,name)
            model.eval()
            decoder.decoder.diagnostic_mode = True
            with torch.inference_mode():
                result = model(images,targets)
            extra = result["diagnostic_query_extensions"]
            self.assertEqual(extra["schema_version"],"bcsr-query-v1")
            self.assertEqual(extra["band_weights"].shape,(2,2,20,4,8))
            self.assertGreater(float(extra["residual_norm"].max()),0.)

    def test_evaluation_writes_boundary_evidence_to_query_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            (root/"labels").mkdir()
            np.save(root/"images/a.npy",np.full((8,64,64),120,dtype=np.uint8))
            (root/"labels/a.txt").write_text("20 24 44 24 44 40 20 40 car 2\n")
            dataset = MODADetection(root)
            image,target = dataset[0]
            image,target,_ = RotatedConvertToTensor()((image,target,dataset))
            decoder = build_tiny_model()
            decoder.geometry_adapter = self.adapter()
            model = RTv4(TinyBackbone(),nn.Identity(),decoder).eval()
            cfg = SimpleNamespace(diagnostics_enabled=True,diagnostics_detailed_image_limit=1,
                diagnostics_query_topk=1,diagnostics_layerwise_epoch_interval=1,
                yaml_cfg={"RotatedDFINETransformer":{"refinement_mode":"o2_adr"}},resume=None)
            diagnostics = OBBDiagnostics(cfg,root/"run",model=model,train_dataset=dataset)
            try:
                evaluate(model,build_criterion(),RotatedPostProcessor(num_classes=3,num_top_queries=20),
                         [(image.unsqueeze(0),[target])],DotaOBBEvaluator(dataset),torch.device("cpu"),
                         diagnostics=diagnostics,epoch=0)
            finally:
                diagnostics.close()
            path = next((root/"run/diagnostics/eval/epoch_0000").glob("queries.rank*.jsonl.gz"))
            with gzip.open(path,"rt") as handle:
                rows = [json.loads(line) for line in handle]
            extensions = [stage["method_diagnostics"] for row in rows for stage in row["stages"] if "method_diagnostics" in stage]
            self.assertTrue(extensions)
            self.assertTrue(all(e["schema_version"]=="bcsr-query-v1" for e in extensions))
            self.assertEqual(np.asarray(extensions[0]["sampling_points"]).shape,(4,4,2,2))
            manifest = json.loads((root/"run/diagnostics/manifest.json").read_text())
            self.assertEqual(manifest["train_dataset_provenance"]["input_channels"],8)


if __name__ == "__main__":
    unittest.main()
