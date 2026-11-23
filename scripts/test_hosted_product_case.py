from __future__ import annotations

import unittest

from scripts import hosted_interval_join as evidence


class BenchCase(unittest.TestCase):
    def test_only_registered_single_worker_case(self):
        from scripts.benchmark_suite.catalog import worker_thread_count

        case = evidence.case()
        self.assertEqual(
            case["id"], "engines/1000000/calc-flow-stream-1t/interval_join"
        )
        self.assertEqual(worker_thread_count(case), 1)
        self.assertEqual(case["batch_rows"], 1_000_000)
        self.assertEqual(case["source_bindings"], ["left", "right"])
        self.assertIsNone(case["checkpoint_interval_millis"])
        self.assertEqual(case["scope"], "ready-enqueue-to-arrow/retained-interval-v2")

    def test_deadline_counts_prior_setup_and_keeps_worker_cleanup_time(self):
        self.assertEqual(evidence.remaining_seconds(100, 120), 250 - 20 - 45)
        with self.assertRaises(TimeoutError):
            evidence.remaining_seconds(100, 306)
