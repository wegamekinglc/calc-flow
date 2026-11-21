from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


class JoinStageProfileTests(unittest.TestCase):
    def test_summary_orders_settled_stages_and_preserves_markers(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks/join-asof/p5-stage-profile-20261120/profile.py"
        )
        spec = importlib.util.spec_from_file_location(
            "join_stage_profile_fixture", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trace = {
            "schema": "calc-flow.join-cpu-trace.v1",
            "capacity": 100,
            "overflowed": False,
            "max_active_workers": 1,
            "events": [
                {
                    "operator": "join",
                    "stage": "probe",
                    "rows": 4,
                    "start_ns": 10,
                    "end_ns": 20,
                    "thread_cpu_ns": 5,
                },
                {
                    "operator": "join",
                    "stage": "probe",
                    "rows": 6,
                    "start_ns": 20,
                    "end_ns": 40,
                    "thread_cpu_ns": 7,
                },
                {
                    "operator": "join",
                    "stage": "admit",
                    "rows": 10,
                    "start_ns": 0,
                    "end_ns": 5,
                    "thread_cpu_ns": None,
                },
                {
                    "operator": "runtime",
                    "stage": "ready_timer_start",
                    "start_ns": 1,
                    "end_ns": 1,
                },
            ],
        }
        result = module.summarize_trace(trace)
        self.assertEqual(result["stages"][0]["stage"], "probe")
        self.assertEqual(result["stages"][0]["wall_ns"], 30)
        self.assertEqual(result["stages"][0]["cpu_ns"], 12)
        self.assertEqual(result["stages"][0]["rows"], 10)
        self.assertEqual(result["stages"][0]["count"], 2)
        self.assertEqual(result["stages"][1]["stage"], "admit")
        self.assertEqual(result["markers"], [trace["events"][-1]])
        with self.assertRaisesRegex(ValueError, "unsettled"):
            module.summarize_trace(
                {**trace, "events": [{**trace["events"][0], "end_ns": None}]}
            )


if __name__ == "__main__":
    unittest.main()
