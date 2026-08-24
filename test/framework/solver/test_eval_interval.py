"""Contract tests for configurable epoch-based validation cadence."""

import unittest
from pathlib import Path

from engine.core import YAMLConfig
from engine.solver._solver import (
    should_evaluate_epoch,
    validate_eval_interval,
)


ROOT = Path(__file__).resolve().parents[3]


class EvaluationIntervalTest(unittest.TestCase):
    def test_runtime_default_and_recipe_override_are_exposed_by_config(self):
        config_path = ROOT / "configs" / "dfine" / "dfine_obb_angle.yml"
        default = YAMLConfig(str(config_path))
        overridden = YAMLConfig(str(config_path), eval_interval=4)
        self.assertEqual(default.eval_interval, 1)
        self.assertEqual(default.yaml_cfg["eval_interval"], 1)
        self.assertEqual(overridden.eval_interval, 4)
        self.assertEqual(overridden.yaml_cfg["eval_interval"], 4)

    def test_default_interval_evaluates_every_epoch(self):
        actual = [
            epoch
            for epoch in range(4)
            if should_evaluate_epoch(epoch, 4, 1)
        ]
        self.assertEqual(actual, [0, 1, 2, 3])

    def test_interval_counts_completed_epochs_and_includes_final_epoch(self):
        actual = [
            epoch
            for epoch in range(8)
            if should_evaluate_epoch(epoch, 8, 3)
        ]
        self.assertEqual(actual, [2, 5, 7])

    def test_interval_must_be_a_positive_integer(self):
        for invalid in (0, -1, 1.5, True, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "eval_interval must be a positive integer"
                ):
                    validate_eval_interval(invalid)


if __name__ == "__main__":
    unittest.main()
