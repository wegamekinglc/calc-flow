"""Tests for the P3 screening matrix and Pareto selection."""

from __future__ import annotations

import unittest

from scripts.run_sql_datafusion_matrix import matrix_points, pareto_frontier


class TestSqlDataFusionMatrix(unittest.TestCase):
    def test_full_screening_matrix_has_every_declared_point(self) -> None:
        points = matrix_points()

        self.assertEqual(len(points), 3 * 4 * 6 * 4)
        self.assertIn((100_000, 1, 1, 4_096), points)
        self.assertIn((2_100_000, 64, 32, 32_768), points)
        self.assertEqual(points, sorted(points))

    def test_pareto_frontier_rejects_dominated_latency_and_memory(self) -> None:
        candidates = pareto_frontier(
            [
                {"id": "fast-large", "median_ms": 8.0, "peak_rss_bytes": 140},
                {"id": "balanced", "median_ms": 10.0, "peak_rss_bytes": 100},
                {"id": "dominated", "median_ms": 11.0, "peak_rss_bytes": 120},
                {"id": "small-slow", "median_ms": 12.0, "peak_rss_bytes": 80},
            ]
        )

        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            ["fast-large", "balanced", "small-slow"],
        )


if __name__ == "__main__":
    unittest.main()
