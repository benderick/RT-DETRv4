"""Scientific control invariants: withheld information cannot reach the model."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from engine.data.dataset.moda_dataset import MODADetection
from engine.data.dataset.moda_information_control import MODAInformationControl


class InformationControlTest(unittest.TestCase):
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
        control = MODAInformationControl(self.root)
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
        torch.testing.assert_close(original[[1, 2, 4]], full[[1, 2, 4]], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(original[[0, 3, 5, 6, 7]]).item(), 0)
        self.assertEqual(original.shape, full.shape)
        for key in target:
            if torch.is_tensor(target[key]):
                torch.testing.assert_close(target[key], full_target[key], rtol=0, atol=0)
            else:
                self.assertEqual(target[key], full_target[key])
        # Difficulty and geometry still follow the published evaluation contract.
        self.assertEqual(target["source_difficulty"].item(), 2)
        self.assertEqual(baseline.get_ground_truth(0)["labels"].tolist(), [0])
        record = control.get_dataset_provenance()
        self.assertEqual(record["retained_source_bands"], [1, 2, 4])
        self.assertEqual(record["input_channels"], 8)

    def test_all_bands_recovers_original_loader_exactly(self):
        full, target = MODADetection(self.root)[0]
        same, _ = MODAInformationControl(self.root, retained_bands=list(range(8)))[0]
        torch.testing.assert_close(same, full, rtol=0, atol=0)
        np.testing.assert_array_equal(np.load(self.path), self.source)

    def test_invalid_band_identity_is_rejected(self):
        for bands in ([], [1, 1], [True, 2, 4], [-1], [8], [1.0], "421"):
            with self.subTest(bands=bands), self.assertRaises(ValueError):
                MODAInformationControl(self.root, retained_bands=bands)


if __name__ == "__main__":
    unittest.main()
