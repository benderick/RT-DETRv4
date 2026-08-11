#!/usr/bin/env python3
"""Run every CODrone OBB test without third-party test dependencies."""

import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "test"), pattern="test_*.py", top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
