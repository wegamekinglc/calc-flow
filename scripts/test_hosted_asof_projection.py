"""The public ASOF experiment requires complete pairs and correct output proof."""

from __future__ import annotations

import unittest


class AsofPairTest(unittest.TestCase):
    def test_fresh_worker_schedule_and_proof(self):
        from scripts.hosted_asof_projection import schedule, verify_worker

        rows = list(schedule())
        self.assertEqual(len(rows), 44)
        for group in (1, 2):
            pairings = [
                [mode for r, p, mode in rows if r == group and p == pair]
                for pair in range(11)
            ]
            self.assertEqual(pairings[0], ["baseline", "candidate"])
            self.assertEqual(
                pairings[1:3], [["baseline", "candidate"], ["candidate", "baseline"]]
            )
            self.assertEqual(pairings[1:], pairings[1:3] * 5)
        obs = {
            "proof": {
                "rows": 1024,
                "matched": 764,
                "unmatched": 260,
                "columns": [
                    "left__key",
                    "left__time",
                    "left__seq",
                    "right__key",
                    "right__time",
                    "right__seq",
                ],
            },
            "stream_ns": 3,
            "environment": {
                "cpu_affinity": [0],
                "native_sha256": "native",
                "tokio_worker_threads": "1",
            },
        }
        verify_worker(obs, 0, "native")
        for field, value in [
            ("stream_ns", 0),
            ("proof", {**obs["proof"], "matched": 763}),
            ("proof", {**obs["proof"], "columns": []}),
        ]:
            with self.assertRaises(ValueError):
                verify_worker({**obs, field: value}, 0, "native")
