from __future__ import annotations

import unittest

from scripts.benchmark_suite.rust import allocation_rows


def reports():
    case = {
        "name": "one",
        "valid": True,
        "repetitions": [
            {"normalized": {"calls_per_dispatch": 0, "bytes_per_dispatch": 0}}
        ],
    }
    return {
        side: {"role": side, "valid": True, "cases": [case]}
        for side in ("baseline", "candidate")
    }


class BenchmarkRustTests(unittest.TestCase):
    def test_zero_allocation_counts_remain_valid_metric_rows(self):
        rows = allocation_rows(reports())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["candidate_value"] == 0 for row in rows))

    def test_wrong_role_cannot_be_a_version_comparison(self):
        inputs = reports()
        inputs["candidate"]["role"] = "baseline"
        with self.assertRaises(ValueError):
            allocation_rows(inputs)

    def test_duplicate_allocation_cases_cannot_disappear_in_a_mapping(self):
        inputs = reports()
        inputs["candidate"]["cases"] *= 2
        with self.assertRaises(ValueError):
            allocation_rows(inputs)


if __name__ == "__main__":
    unittest.main()
