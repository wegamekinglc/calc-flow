"""Check the registered active-watermark measurement boundaries."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

STUDY = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/join-asof/20261123-anti-watermark/collect.py"
)
spec = importlib.util.spec_from_file_location("active_anti_collect", STUDY)
assert spec is not None and spec.loader is not None
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class ActiveWatermarkTests(unittest.TestCase):
    def test_excludes_both_warmups_and_rejects_incomplete_processes(self):
        rows = [
            {
                "round": group,
                "index": index,
                "elapsed_ns": 99999 if index == 0 else 1000 * index,
            }
            for group in (1, 2)
            for index in range(11)
        ]
        self.assertEqual(collector.summarize(rows)["p50_us"], 5.5)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            collector.summarize(rows[:-1])

    def test_requires_equality_acknowledgment_and_last_output_chunk(self):
        record = {
            "setup": [260, 0, 261],
            "chunks": [32] * 8 + [1],
            "rows": 257,
            "tail": [259],
            "elapsed_ns": 1000,
        }
        collector.verify(record)
        for wrong in ({"setup": [260, 0]}, {"chunks": [32] * 8}, {"elapsed_ns": None}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                collector.verify(dict(record, **wrong))
