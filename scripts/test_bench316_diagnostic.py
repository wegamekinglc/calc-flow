from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.side_bias_diagnostic import (
    ObservedWorker,
    condition_matrix,
)


class SideBiasDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def test_condition_matrix_covers_identity_and_reversed_physical_slots(self):
        conditions = {condition[0]: condition[1:] for condition in condition_matrix()}

        self.assertEqual(conditions["aa_same_site"], ("A0", "A0", "baseline"))
        self.assertEqual(conditions["aa_separate"], ("A0", "A1", "baseline"))
        self.assertEqual(conditions["aa_swapped"], ("A1", "A0", "baseline"))
        self.assertEqual(conditions["ab"], ("A0", "B1", "baseline"))
        self.assertEqual(conditions["ba"], ("B1", "A0", "baseline"))
        self.assertEqual(conditions["aa_reversed_start"], ("A0", "A1", "candidate"))

    async def test_observed_worker_records_start_warmup_and_sample_order(self):
        class FakeWorker:
            process = SimpleNamespace(pid=123)

            async def request(self, **message):
                if message["operation"] == "prepare":
                    return {"warmup": {"seconds": 0.2}}
                return {"seconds": 0.1, "start_row": 10}

            async def close(self):
                return None

        events = []
        with patch(
            "scripts.benchmark_suite.side_bias_diagnostic.Worker.start",
            return_value=FakeWorker(),
        ):
            worker = await ObservedWorker.start(
                Path("/physical/A0/site"), Path("/trial/round-0/baseline"), events
            )
            await worker.request(operation="prepare", case={"id": "case"})
            await worker.request(operation="sample")
            await worker.close()

        self.assertEqual(
            [event["event"] for event in events],
            ["start", "prepare", "sample", "close"],
        )
        self.assertEqual(events[0]["site"], "/physical/A0/site")
        self.assertEqual(events[0]["side"], "baseline")
        self.assertEqual(events[0]["pid"], 123)
        self.assertEqual(events[1]["warmup_seconds"], 0.2)
        self.assertEqual(events[2]["sample_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
