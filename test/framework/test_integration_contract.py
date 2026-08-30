import json
import unittest
from pathlib import Path

from engine.core import GLOBAL_CONFIG, YAMLConfig
from engine.rtv4 import RotatedDFINETransformer
from engine.rtv4.obb.methods.o2.adr import o2_weighting_function


ROOT = Path(__file__).resolve().parents[2]


def _tiny_decoder(**overrides):
    arguments = dict(
        num_classes=3,
        hidden_dim=32,
        num_queries=20,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[2, 2, 2],
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
        reg_max=8,
    )
    arguments.update(overrides)
    return RotatedDFINETransformer(**arguments)


class FrameworkContractTest(unittest.TestCase):
    def test_only_protocol_level_evaluators_are_registered(self):
        self.assertIn("DotaOBBEvaluator", GLOBAL_CONFIG)
        self.assertIn("MergedDotaOBBEvaluator", GLOBAL_CONFIG)
        self.assertNotIn("CODroneEvaluator", GLOBAL_CONFIG)
        self.assertNotIn("CODroneMergedEvaluator", GLOBAL_CONFIG)

    def test_obsolete_module_paths_are_removed(self):
        obsolete = (
            ROOT / "engine/data/dataset/codrone_eval.py",
            ROOT / "engine/rtv4/o2_adr.py",
        )
        self.assertTrue(all(not path.exists() for path in obsolete))
        self.assertTrue(callable(o2_weighting_function))

    def test_refinement_mode_is_the_only_public_selector(self):
        expected = {
            "direct_angle": False,
            "o2_adr": True,
        }
        for mode, use_adr in expected.items():
            model = _tiny_decoder(refinement_mode=mode)
            self.assertEqual(model.refinement_mode, mode)
            self.assertEqual(model.use_adr, use_adr)
        with self.assertRaisesRegex(ValueError, "Unknown refinement_mode"):
            _tiny_decoder(refinement_mode="unsupported_mode")
        with self.assertRaises(TypeError):
            _tiny_decoder(use_adr=True)

    def test_stable_configs_resolve_generic_evaluators_and_named_modes(self):
        expected = {
            "dfine_obb_angle.yml": ("direct_angle", "DotaOBBEvaluator"),
            "dfine_obb_o2.yml": ("o2_adr", "DotaOBBEvaluator"),
            "dfine_obb_angle_tile.yml": (
                "direct_angle", "MergedDotaOBBEvaluator"),
            "dfine_obb_o2_tile.yml": ("o2_adr", "MergedDotaOBBEvaluator"),
        }
        for name, (mode, evaluator) in expected.items():
            resolved = YAMLConfig(str(ROOT / "configs" / "dfine" / name))
            yaml = resolved.yaml_cfg
            decoder = yaml["RotatedDFINETransformer"]
            matcher = yaml["RotatedRTv4Criterion"]["matcher"]
            self.assertEqual(decoder["refinement_mode"], mode)
            self.assertEqual(yaml["evaluator"]["type"], evaluator)
            self.assertNotIn("use_adr", decoder)
            self.assertNotIn("chamfer_squared", matcher)
            self.assertEqual(yaml["diagnostics_detailed_epoch_interval"], 10)
            self.assertEqual(resolved.diagnostics_detailed_epoch_interval, 10)
            self.assertEqual(yaml["diagnostics_layerwise_epoch_interval"], 10)
            self.assertEqual(resolved.diagnostics_layerwise_epoch_interval, 10)

    def test_integration_contract_and_canonical_packages_exist(self):
        self.assertTrue((ROOT / "docs/framework/INTEGRATION_CONTRACT.md").is_file())
        self.assertTrue((ROOT / "docs/framework/WORKTREE_CONTRACT.md").is_file())
        self.assertTrue((ROOT / "docs/research/o2/implementation_audit.md").is_file())
        self.assertTrue((ROOT / "docs/research/IDEA_LIFECYCLE.md").is_file())
        self.assertTrue((ROOT / "docs/research/EXPERIENCE_LOG.md").is_file())
        registry = json.loads(
            (ROOT / "docs/research/registry.json").read_text(encoding="utf-8"))
        active = {
            idea["id"] for idea in registry["ideas"]
            if idea["status"] != "retired"
        }
        for manifest_path in (ROOT / "docs/research").glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["schema_version"], "research-idea-manifest-v1")
            active.add(manifest["idea"]["id"])
        self.assertTrue((ROOT / "engine/rtv4/obb/methods/o2/adr.py").is_file())
        self.assertFalse((ROOT / "configs/research").exists())
        self.assertEqual(
            {path.name for path in (ROOT / "docs/research").iterdir()
             if path.is_dir()},
            {"o2", *active},
        )
        self.assertEqual(
            {path.name for path in (ROOT / "test/research").iterdir()
             if path.is_dir() and path.name != "__pycache__"},
            {"o2", *active},
        )
        self.assertEqual(
            {path.name for path in (ROOT / "tools/research").iterdir()
             if path.is_dir() and path.name != "__pycache__"},
            {"o2", *active},
        )
        self.assertFalse((ROOT / "参考资料").exists())


if __name__ == "__main__":
    unittest.main()
