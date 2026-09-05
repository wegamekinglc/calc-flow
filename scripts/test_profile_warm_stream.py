"""Fail-closed checks for paired warm-stream profiling."""

from __future__ import annotations

import unittest

from scripts.profile_warm_stream import matrix_points, paired_summary


class WarmProfileTests(unittest.TestCase):
    def test_matrix_covers_history_and_increment_scales_without_duplicates(
        self,
    ) -> None:
        points = matrix_points()
        self.assertEqual(len(points), 7)
        self.assertIn((10_240_000, 64), points)
        self.assertIn((1_024_000, 64_000), points)
        self.assertEqual(len(set(points)), len(points))

    def test_paired_summary_preserves_speedup_direction_and_tail(self) -> None:
        result = paired_summary([2, 4, 6, 8], [1, 2, 3, 4], rows=64)
        self.assertEqual(result["paired_speedup_median"], 2)
        self.assertEqual(result["paired_speedup_ci95"], [2, 2])
        self.assertAlmostEqual(result["candidate"]["p95_seconds"], 3.85)

    def test_unpaired_or_invalid_samples_cannot_produce_evidence(self) -> None:
        for baseline, candidate in (
            ([1, 2], [1]),
            ([1, 2], [0, 1]),
            ([1, 2], [float("nan"), 1]),
            ([], []),
            ([1], [1]),
        ):
            with self.assertRaises(ValueError):
                paired_summary(baseline, candidate, rows=64)


if __name__ == "__main__":
    unittest.main()
