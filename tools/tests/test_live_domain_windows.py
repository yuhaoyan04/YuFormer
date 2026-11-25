import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('live_domain_windows', Path(__file__).resolve().parents[1]/'audit_live_domain_windows.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WindowTests(unittest.TestCase):
    def test_physical_source_offset_and_half_open_consumed_window(self):
        counts = np.array([[8, 0], [0, 8], [3, 5], [6, 2]], dtype=np.int64)
        order = np.array([102, 100, 103, 101], dtype=np.uint32)
        result = MODULE.count_window(counts, order, 1, 3, source_first_block=100)
        np.testing.assert_array_equal(result, [14, 2])

    def test_physical_block_outside_source_window_rejected(self):
        counts = np.ones((2, 2), dtype=np.int64)
        for bad in [99, 102]:
            with self.assertRaises(ValueError):
                MODULE.count_window(counts, np.array([100, bad]), 0, 2, source_first_block=100)

    def test_empty_or_unconsumed_window_rejected(self):
        counts = np.ones((2, 2), dtype=np.int64)
        for first, last in [(-1, 1), (0, 0), (0, 3)]:
            with self.assertRaises(ValueError):
                MODULE.count_window(counts, np.array([100, 101]), first, last, source_first_block=100)
