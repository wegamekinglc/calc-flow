"""The public sparse experiment requires full paired, correctly sized workers."""

from __future__ import annotations

import unittest


class WarmUpdateTest(unittest.TestCase):
    def test_fresh_worker_schedule_and_proof(self):
        from scripts.hosted_sparse_update import schedule, verify_worker

        rows = list(schedule())
        self.assertEqual(len(rows), 44)
        for group in (1, 2):
            pairings = [
                [mode for r, p, mode in rows if r == group and p == pair]
                for pair in range(11)
            ]
            self.assertEqual(pairings[0], ["snapshot", "changelog"])
            self.assertEqual(
                pairings[1:3], [["snapshot", "changelog"], ["changelog", "snapshot"]]
            )
            self.assertEqual(pairings[1:], pairings[1:3] * 5)
        obs = {
            "mode": "changelog",
            "update_ns": [1] * 20,
            "output_rows": [10000] + [160] * 20,
            "environment": {
                "cpu_affinity": [0],
                "native_sha256": "native",
                "tokio_worker_threads": "1",
            },
        }
        verify_worker(obs, "changelog", 0, "native")
        for field, value in [
            ("update_ns", [1] * 19),
            ("output_rows", [10000] * 21),
            ("mode", "snapshot"),
        ]:
            with self.assertRaises(ValueError):
                verify_worker({**obs, field: value}, "changelog", 0, "native")
