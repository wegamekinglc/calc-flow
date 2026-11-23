"""The single public Equi Join case requires full outputs and causal state."""

from __future__ import annotations

import copy
import unittest

from scripts.hosted_static_product_state import CASE_ID, ROWS, validate_static_sample


def complete():
    return {
        "correctness": {
            "passed": True,
            "rows": ROWS,
            "finite_rows": ROWS,
            "max_abs_error": 0.0,
        },
        "stream_evidence": {
            "sealed_dimension_watermark_micros": 45,
            "join_operator": {
                "input_rows": ROWS + 64,
                "fanned_out_rows": ROWS,
                "fanned_out_bytes": 16 * ROWS,
                "errors": 0,
            },
            "lookup_join": {
                "emitted_match_rows": ROWS,
                "state_limit_failures": 0,
                "match_limit_failures": 0,
                "left": {"retained_rows": 0, "evicted_rows": 0},
                "right": {"watermark_micros": 45},
            },
        },
    }


class StaticProductTest(unittest.TestCase):
    def test_observation_requires_exact_outputs_and_causal_state(self):
        self.assertEqual(CASE_ID, "engines/1000000/calc-flow-stream-1t/join")
        proof = complete()
        validate_static_sample(proof)
        for field in ("rows", "finite_rows", "max_abs_error"):
            invalid = copy.deepcopy(proof)
            invalid["correctness"][field] += 1
            with self.assertRaises(ValueError):
                validate_static_sample(invalid)
        for side, field in (("left", "retained_rows"), ("left", "evicted_rows")):
            invalid = copy.deepcopy(proof)
            invalid["stream_evidence"]["lookup_join"][side][field] = 1
            with self.assertRaises(ValueError):
                validate_static_sample(invalid)
        invalid = copy.deepcopy(proof)
        invalid["stream_evidence"]["lookup_join"]["right"]["watermark_micros"] = 44
        with self.assertRaises(ValueError):
            validate_static_sample(invalid)

    def test_rejects_three_operands_instead_of_fused_sequence_and_product(self):
        proof = complete()
        proof["stream_evidence"]["join_operator"]["fanned_out_bytes"] = 24 * ROWS
        with self.assertRaises(ValueError):
            validate_static_sample(proof)

    def test_static_public_entry_observes_normalized_watermark(self):
        from datetime import timedelta
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from benchmarks.engine_comparison import EngineCase
        from benchmarks.engine_stream import BASE, BASE_MICROS, _validated_timed_streams

        with TemporaryDirectory() as root:
            case = EngineCase(
                {"backend": "calc-flow-stream-1t", "scenario": "join", "rows": 128},
                Path(root),
            )
            try:
                self.assertIsNone(case.streams["right"][-1])
                plan = SimpleNamespace(source_binding_ids=("left", "right"))
                timed = _validated_timed_streams(plan, case.streams)
                timestamp = timed["right"][-1].at
                with self.assertRaisesRegex(ValueError, "EOF marker"):
                    _validated_timed_streams(plan, timed)
                row = case.sample()
                self.assertEqual(row["correctness"]["max_abs_error"], 0)
                self.assertEqual(row["correctness"]["rows"], 128)
                evidence = row["stream_evidence"]
                self.assertEqual(evidence["join_operator"]["fanned_out_rows"], 128)
                self.assertEqual(
                    evidence["sealed_dimension_watermark_micros"],
                    BASE_MICROS + (timestamp - BASE) // timedelta(microseconds=1),
                )
            finally:
                case.close()

    def test_empty_public_output_cannot_satisfy_static_workload(self):
        proof = complete()
        proof["correctness"].update(rows=0, finite_rows=0)
        with self.assertRaisesRegex(ValueError, "one million exact"):
            validate_static_sample(proof)
