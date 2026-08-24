import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.inference.obb_tile_infer import _jsonable, _write_json, _write_jsonl_gz


class TileInferenceSerializationTest(unittest.TestCase):
    def test_jsonable_handles_evaluator_numpy_statistics(self):
        payload = {
            "stats": np.asarray([0.4, np.nan, np.inf], dtype=np.float32),
            "count": np.int64(3),
            "enabled": np.bool_(True),
            "tensor": torch.tensor([1.0, float("nan")]),
            "path": Path("predictions"),
        }
        converted = _jsonable(payload)
        self.assertAlmostEqual(converted["stats"][0], 0.4, places=6)
        self.assertEqual(converted["stats"][1:], [None, None])
        self.assertEqual(converted["count"], 3)
        self.assertIs(converted["enabled"], True)
        self.assertEqual(converted["tensor"], [1.0, None])
        self.assertEqual(converted["path"], "predictions")

    def test_json_writers_round_trip_numpy_and_publish_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "metrics.json"
            payload = {"stats": np.asarray([0.1, 0.2]), "scalar": np.float64(0.3)}
            _write_json(json_path, payload)
            self.assertFalse((root / "metrics.json.tmp").exists())
            with json_path.open(encoding="utf-8") as handle:
                restored = json.load(handle)
            self.assertEqual(restored["stats"], [0.1, 0.2])
            self.assertAlmostEqual(restored["scalar"], 0.3)

            stream_path = root / "predictions.jsonl.gz"
            _write_jsonl_gz(stream_path, [payload])
            with gzip.open(stream_path, "rt", encoding="utf-8") as handle:
                restored_stream = json.loads(handle.readline())
            self.assertEqual(restored_stream["stats"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
