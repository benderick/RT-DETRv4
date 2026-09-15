import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from engine.core import YAMLConfig
from tools.experiments import moda


class MODAEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, action, variant, *extra):
        return moda.parser().parse_args([action, variant, "--runs-dir", str(self.root), *extra])

    def checkpoint(self, variant):
        run = self.root / variant / "seed0_fp32"
        run.mkdir(parents=True)
        cfg = YAMLConfig(str(moda.ROOT / "configs/experiments/moda" / moda.EXPERIMENTS[variant])).yaml_cfg
        (run / "configs.json").write_text(json.dumps({"yaml_cfg": cfg}))
        checkpoint = run / "best_stg1.pth"
        checkpoint.touch()
        return checkpoint

    def test_rgb_diff_is_limited_to_input_and_output(self):
        configs = [YAMLConfig(str(moda.ROOT / "configs/experiments/moda" / filename)).yaml_cfg
                   for filename in moda.EXPERIMENTS.values()]
        baseline, rgb = configs
        for key in ("__include__", "imports", "output_dir"):
            baseline.pop(key, None)
            rgb.pop(key, None)
        for key in ("train_dataloader", "val_dataloader"):
            dataset = rgb[key]["dataset"]
            self.assertEqual(dataset.pop("retained_bands"), [4, 2, 1])
            self.assertEqual(dataset["type"], "MODAInformationControl")
            dataset["type"] = "MODADetection"
        self.assertEqual(baseline, rgb)

    def test_train_launches_directly_and_dry_run_writes_nothing(self):
        for variant in moda.EXPERIMENTS:
            argv = ["train", variant, "--runs-dir", str(self.root)]
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(moda.subprocess, "run") as launch:
                moda.main(argv + ["--dry-run"])
                launch.assert_not_called()
                run = self.root / variant / "seed0_fp32"
                self.assertFalse(run.exists())
                launch.return_value.returncode = 0
                with self.assertRaises(SystemExit) as stopped:
                    moda.main(argv)
                self.assertEqual(stopped.exception.code, 0)
                launch.assert_called_once()
                command = launch.call_args.args[0]
                self.assertIn(str(moda.ROOT / "train.py"), command)
                self.assertNotIn("--test-only", command)
                self.assertEqual(json.loads((run / "run_spec.json").read_text())["variant"], variant)
                with self.assertRaises(ValueError):
                    moda.command(self.args("train", variant))

    def test_eval_checks_input_identity_and_preserves_training_directory(self):
        for variant in moda.EXPERIMENTS:
            checkpoint = self.checkpoint(variant)
            command, _, record = moda.command(self.args("eval", variant))
            self.assertIn("--test-only", command)
            self.assertEqual(command[command.index("-r") + 1], str(checkpoint))
            self.assertEqual(command[command.index("--output-dir") + 1],
                             str(checkpoint.parent / "eval_best_stg1_detr"))
            self.assertIsNone(record)
            wrong = "rgb" if variant == "baseline" else "baseline"
            with self.assertRaisesRegex(ValueError, "input does not match"):
                moda.command(self.args("eval", wrong, "--checkpoint", str(checkpoint)))
            resumed, _, record = moda.command(self.args("train", variant, "--checkpoint", str(checkpoint)))
            self.assertNotIn("--test-only", resumed)
            self.assertEqual(resumed[resumed.index("--output-dir") + 1], str(checkpoint.parent))
            self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
