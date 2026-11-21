from __future__ import annotations

import argparse
import unittest
from types import SimpleNamespace

from scripts.profile_checkpoint_v4 import (
    DIRECT_CONTRACT,
    paired_statistics,
    positive_int,
    validate_direct_evidence,
)


def evidence(ratios: list[float]) -> dict:
    return {
        "contract": DIRECT_CONTRACT,
        "case": "1x1-dual",
        "history_rows": 4096,
        "recorder": "absent",
        "pairs": [
            {
                "on_median_seconds": ratio,
                "off_median_seconds": 1.0,
                "dispatches_per_arm": 30,
                "output_digest": f"digest-{index}",
            }
            for index, ratio in enumerate(ratios)
        ],
    }


def arguments() -> SimpleNamespace:
    return SimpleNamespace(
        case="1x1-dual",
        history_rows=4096,
        recorder="absent",
        pairs=10,
        dispatches=30,
    )


class CheckpointV4DirectTests(unittest.TestCase):
    def test_paired_statistics_passes_inside_the_three_percent_gate(self):
        result = paired_statistics([evidence([1.01] * 10), evidence([1.02] * 10)], 0.03)
        self.assertEqual(result["verdict"], "passed")
        self.assertEqual(len(result["per_round"]), 2)
        self.assertAlmostEqual(result["pooled"]["on_off_median"], 0.015)

    def test_paired_statistics_fails_wholly_above_the_gate(self):
        result = paired_statistics([evidence([1.10] * 10), evidence([1.08] * 10)], 0.03)
        self.assertEqual(result["verdict"], "failed")

    def test_paired_statistics_marks_gate_crossing_inconclusive(self):
        ratios = [0.90, 1.10] * 5
        result = paired_statistics([evidence(ratios), evidence(ratios)], 0.03)
        self.assertEqual(result["verdict"], "inconclusive")

    def test_direct_validation_accepts_matching_evidence(self):
        validate_direct_evidence(evidence([1.0] * 10), arguments())

    def test_direct_validation_rejects_missing_digest(self):
        value = evidence([1.0] * 10)
        value["pairs"][0]["output_digest"] = None
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_direct_evidence(value, arguments())

    def test_positive_int_rejects_zero(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")


if __name__ == "__main__":
    unittest.main()
