import unittest
import torch
from engine.core import YAMLConfig
from engine.core.yaml_utils import merge_dict
from engine.optim.warmup import LinearEpochLR, GroupLinearWarmup


class PaperRecipeTest(unittest.TestCase):
    def test_explicit_replacement_removes_previous_constructor_arguments(self):
        merged=merge_dict({"evaluator":{"type":"Old","extra":3}},
                          {"evaluator":{"__replace__":True,"type":"New"}})
        self.assertEqual(merged,{"evaluator":{"type":"New"}})

    def test_paper_budget_and_released_evaluation_settings_are_explicit(self):
        config=YAMLConfig("configs/experiments/moda/dfine_obb_o2_fressdet.yml")
        value=config.yaml_cfg
        self.assertEqual(value["epoches"],20)
        self.assertEqual(value["eval_spatial_size"],[928,1216])
        self.assertEqual(value["train_dataloader"]["total_batch_size"],8)
        self.assertIsNone(value["train_dataloader"]["dataset"]["split_file"])
        self.assertEqual(value["val_dataloader"]["dataset"]["root"],"./data/MODA/test")
        self.assertEqual(value["evaluator"]["type"],"BenchmarkOBBEvaluator")
        self.assertNotIn("use_07_metric",value["evaluator"])
        self.assertEqual(value["evaluator"]["selection_metric"],"AP50")
        self.assertFalse(value["HGNetv2"]["pretrained"])
        ops=value["train_dataloader"]["dataset"]["transforms"]["ops"]
        self.assertEqual([op["type"] for op in ops],["RotatedResize","RotatedPad","RotatedConvertToTensor"])
        self.assertEqual(value["RotatedPostProcessor"]["nms_method"],"probiou_fast")
        self.assertEqual(value["RotatedPostProcessor"]["max_detections"],300)
        self.assertEqual(value["optimizer"]["lr"],.01)
        self.assertEqual(value["lr_warmup_scheduler"]["warmup_duration"],3435)

        # Sequential indices do not reveal the module type. All normalization
        # weights must avoid decay even when their names contain no norm/bn.
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.LayerNorm(4))
        groups=YAMLConfig.get_optim_params(value["optimizer"],model)
        optimizer=torch.optim.AdamW(groups,lr=.01,weight_decay=.0005)
        settings={id(p):g for g in optimizer.param_groups for p in g["params"]}
        self.assertEqual(settings[id(model[0].weight)]["weight_decay"],.0005)
        self.assertEqual(settings[id(model[1].weight)]["weight_decay"],0.)
        self.assertEqual(settings[id(model[1].bias)]["warmup_start_lr"],.1)

    def test_epoch_decay_and_bias_warmup_match_declared_schedule(self):
        bias=torch.nn.Parameter(torch.zeros(1));weight=torch.nn.Parameter(torch.zeros(1))
        optimizer=torch.optim.AdamW([{"params":[bias],"warmup_start_lr":.1},
                                    {"params":[weight],"warmup_start_lr":0.}],lr=.01)
        scheduler=LinearEpochLR(optimizer,total_epochs=20,final_ratio=.01)
        warmup=GroupLinearWarmup(scheduler,warmup_duration=6)
        self.assertEqual([g["lr"] for g in optimizer.param_groups],[.1,0.])
        optimizer.step();warmup.step()
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"],.01/6)
        optimizer.step();warmup.step();scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0],.009505)
        warmup.prepare_step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.1+(2/6)*(.009505-.1))
        warmup.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],.1+.5*(.009505-.1))
        for _ in range(18):
            optimizer.step();scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0],.000595)
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0],.0001)


if __name__ == "__main__": unittest.main()
