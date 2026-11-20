from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


def load_measurement():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/join-asof/p5-morsel-state-20261120/measure.py"
    )
    spec = importlib.util.spec_from_file_location("join_p5_measurement_fixture", path)
    module = importlib.util.module_from_spec(spec)
    engine = ModuleType("benchmarks.engine_comparison")
    engine.EngineCase = object
    calc_flow = ModuleType("calc_flow")
    calc_flow._native = ModuleType("calc_flow._native")
    with patch.dict(
        sys.modules,
        {
            "benchmarks.engine_comparison": engine,
            "calc_flow": calc_flow,
            "calc_flow._native": calc_flow._native,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class JoinP5MeasurementTests(unittest.TestCase):
    def test_observation_passes_remaining_deadline_to_real_execution(self):
        module = load_measurement()
        stream = object()
        wait = object()
        engine = SimpleNamespace(
            loop=SimpleNamespace(run_until_complete=Mock(return_value=("table", 1.25))),
            _stream=Mock(return_value=stream),
            validate=Mock(return_value={"passed": True, "rows": 3}),
            finish=Mock(return_value={"state": "completed"}),
            close=Mock(),
        )
        with (
            patch.object(module, "EngineCase", return_value=engine),
            patch.object(
                module.asyncio, "wait_for", Mock(return_value=wait)
            ) as wait_for,
        ):
            result = module.observe({"rows": 3}, Path("sample"), 12.5)
        wait_for.assert_called_once_with(stream, timeout=12.5)
        engine.loop.run_until_complete.assert_called_once_with(wait)
        engine.validate.assert_called_once_with("table")
        engine.close.assert_called_once_with()
        self.assertEqual(result["seconds"], 1.25)
        self.assertEqual(result["correctness"]["rows"], 3)

    def test_case_does_not_start_an_observation_after_the_deadline(self):
        module = load_measurement()
        plan = {"execution": {"warmups_per_case": 0, "measured_samples_per_case": 2}}
        calls = []

        def observe(_case, root, timeout):
            calls.append((root.name, timeout))
            if timeout <= 0:
                raise TimeoutError("deadline")
            return {"seconds": 1.0, "correctness": {"passed": True, "rows": 3}}

        with (
            patch.object(module, "observe", side_effect=observe),
            patch.object(module.time, "perf_counter", side_effect=[10.0, 571.0]),
            self.assertRaisesRegex(TimeoutError, "deadline"),
            patch("pathlib.Path.mkdir"),
        ):
            module.measure_case(
                plan,
                {"scenario": "join", "rows": 3, "threshold_p50_ms": 1_000},
                Path("case"),
                570.0,
            )
        self.assertEqual(calls, [("sample-1", 560.0)])

    def test_nonpositive_observation_budget_does_not_construct_engine(self):
        module = load_measurement()
        with (
            patch.object(module, "EngineCase") as constructor,
            self.assertRaisesRegex(TimeoutError, "deadline"),
        ):
            module.observe({}, Path("expired"), 0.0)
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
