from __future__ import annotations

import unittest

from scripts import hosted_duration_sum as evidence


class BenchCase(unittest.TestCase):
    def test_three_window_case_uses_one_native_worker(self):
        from scripts.benchmark_suite.catalog import worker_thread_count

        case = evidence.case()
        self.assertEqual(case["id"], "python/public-duration-sum-30-60-90/fresh-stream")
        self.assertEqual(case["rows"], 100_000)
        self.assertEqual(worker_thread_count(case), 1)

    def test_deadline_counts_prior_setup_and_keeps_worker_cleanup_time(self):
        self.assertEqual(evidence.remaining_seconds(100, 120), 119 - 20 - 45)
        with self.assertRaises(TimeoutError):
            evidence.remaining_seconds(100, 175)
