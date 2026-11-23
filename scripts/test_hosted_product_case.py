from __future__ import annotations

import unittest

from scripts import hosted_interval_join as evidence
from scripts.hosted_join_state import retained_state


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
        self.assertEqual(evidence.remaining_seconds(100, 120), 350 - 20 - 45)
        with self.assertRaises(TimeoutError):
            evidence.remaining_seconds(100, 406)

    def test_verify_both_retained_sides_without_prescribing_improvement(self):
        side = {
            "evicted_rows": 999680,
            "retained_rows": 320,
            "retained_bytes": 80029416,
            "watermark_micros": 1767241224000001,
            "idle": False,
            "ended": False,
            "late_rows": 0,
            "null_key_rows": 0,
            "null_event_time_rows": 0,
        }
        node = {
            "left": side,
            "right": {**side, "retained_bytes": 12345},
            "emitted_match_rows": 10998080,
            "match_limit_failures": 0,
            "state_limit_failures": 0,
        }

        def sample(**overrides):
            return {
                "stream_evidence": {"interval_join": {"test": {**node, **overrides}}},
                "correctness": {
                    "passed": True,
                    "rows": 10998080,
                    "finite_rows": 10998080,
                    "max_abs_error": 0.0,
                },
                "seconds": 1.0,
            }

        evidence_row = retained_state(sample())
        self.assertEqual(evidence_row["left"], side)
        self.assertEqual(evidence_row["right"]["retained_bytes"], 12345)
        for key, value in (
            ("evicted_rows", 999679),
            ("retained_rows", 321),
            ("watermark_micros", 123),
            ("idle", True),
            ("ended", True),
            ("late_rows", 1),
            ("null_key_rows", 1),
            ("retained_bytes", 0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                retained_state(sample(right={**side, key: value}))
        for key, value in (
            ("emitted_match_rows", 10998079),
            ("match_limit_failures", 1),
            ("state_limit_failures", 1),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                retained_state(sample(**{key: value}))

    def test_warmup_and_measured_observations_reject_missing_join_state(self):
        from scripts.benchmark_suite.catalog import STREAM_EVIDENCE_FIELDS
        from scripts.benchmark_suite.measure import validate_stream_sample

        case = evidence.case()
        sample = {
            "stream_evidence": {k: case[k] for k in STREAM_EVIDENCE_FIELDS},
            "correctness": {
                "passed": True,
                "rows": 10998080,
                "finite_rows": 10998080,
                "max_abs_error": 0.0,
            },
        }
        with self.assertRaises(KeyError):
            validate_stream_sample(case, sample)
