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
