"""Three-channel input order, source immutability and label preservation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from engine.data.dataset.moda_dataset import MODADetection
from engine.data.dataset.moda_rgb_dataset import MODARGBDetection


class MODARGBTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "labels").mkdir()
        self.path = self.root / "images/object.npy"
        self.source = np.random.default_rng(7).integers(0, 256, (8, 32, 24), dtype=np.uint8)
        np.save(self.path, self.source)
        (self.root / "labels/object.txt").write_text("3 4 13 4 13 12 3 12 car 2\n")

    def test_withheld_pixels_have_no_effect_and_selected_pixels_do(self):
        control = MODARGBDetection(self.root)
        original, target = control[0]
        changed = self.source.copy()
        changed[[0, 3, 5, 6, 7]] = 255 - changed[[0, 3, 5, 6, 7]]
        np.save(self.path, changed)
        torch.testing.assert_close(control[0][0], original, rtol=0, atol=0)
        changed[4] = 255 - changed[4]
        np.save(self.path, changed)
        self.assertFalse(torch.equal(control[0][0], original))
        np.save(self.path, self.source)
        baseline = MODADetection(self.root)
        full, full_target = baseline[0]
        torch.testing.assert_close(original, full[[4, 2, 1]], rtol=0, atol=0)
        self.assertEqual(original.shape, (3, 24, 32))
        self.assertTrue(original.is_contiguous())
        for key in target:
            if torch.is_tensor(target[key]):
                torch.testing.assert_close(target[key], full_target[key], rtol=0, atol=0)
            else:
                self.assertEqual(target[key], full_target[key])
        # Difficulty and geometry still follow the published evaluation contract.
        self.assertEqual(target["source_difficulty"].item(), 2)
        self.assertEqual(baseline.get_ground_truth(0)["labels"].tolist(), [0])
        record = control.get_dataset_provenance()
        self.assertEqual(record["retained_source_bands"], [4, 2, 1])
        self.assertEqual(record["input_channels"], 3)

    def test_reading_rgb_preserves_source_bytes_and_eight_band_baseline(self):
        before = self.path.read_bytes()
        MODARGBDetection(self.root)[0]
        self.assertEqual(self.path.read_bytes(), before)
        full, _ = MODADetection(self.root)[0]
        torch.testing.assert_close(full, torch.from_numpy(self.source.transpose(0, 2, 1).copy()), rtol=0, atol=0)
        np.testing.assert_array_equal(np.load(self.path), self.source)


if __name__ == "__main__":
    unittest.main()
