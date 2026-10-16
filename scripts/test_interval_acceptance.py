from __future__ import annotations

import asyncio
import copy
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pyarrow as pa

from benchmarks.interval_acceptance import (
    export_interval_fixture,
    validate_interval_output,
)


def expected_output():
    return pa.table(
        {
            "sequence": pa.array([0, 0, 1], type=pa.uint64()),
            "right_sequence": pa.array([0, 1, 1], type=pa.uint64()),
            "value": [4.0, 6.0, 9.0],
        }
    )


def engine_fixture():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Data:
        batch: object

    @dataclass(frozen=True)
    class Watermark:
        at: datetime

    engine = ModuleType("benchmarks.engine_stream")
    engine.Data, engine.Watermark = Data, Watermark
    engine.BASE = datetime(2026, 1, 1, tzinfo=UTC)
    engine.BASE_MICROS = 1_767_225_600_000_000
    engine.Runtime = lambda: "runtime"
    engine._require_ready_sources = Mock()
    batch = pa.table({"sequence": pa.array([0, 1], type=pa.uint64())})
    envelope = SimpleNamespace(to_pyarrow=lambda: batch)
    events = (
        Data(SimpleNamespace(num_rows=2, to_pyarrow=envelope.to_pyarrow)),
        Watermark(engine.BASE + timedelta(microseconds=1)),
        None,
    )
    return engine, events, batch


class IntervalOracleTests(unittest.TestCase):
    def test_all_nine_new_configurations_reach_the_maintained_runtime_fixture(self):
        from dataclasses import asdict

        from benchmarks.join_asof_acceptance_runtime import AcceptanceEngineCase
        from scripts.benchmark_suite.issue406_milestone import inventory

        engine, events, batch = engine_fixture()
        engine.stream_events = Mock(return_value=events)
        engine.stream_program = Mock(return_value="program")
        comparison = ModuleType("benchmarks.engine_comparison")
        comparison.workload = Mock(
            return_value=SimpleNamespace(table=batch, dimension=None, entities=64)
        )
        comparison.expected_output = Mock(return_value=expected_output())
        cases = [case for case in inventory() if case.row in (9, 10)]
        self.assertEqual(len(cases), 9)
        with (
            TemporaryDirectory() as raw,
            patch.dict(
                "sys.modules",
                {engine.__name__: engine, comparison.__name__: comparison},
            ),
            patch("benchmarks.engine_stream", engine, create=True),
            patch(
                "benchmarks.interval_acceptance.export_interval_fixture",
                return_value={"interval": True},
            ) as export,
        ):
            for case in cases:
                active = AcceptanceEngineCase(asdict(case), Path(raw))
                try:
                    scenario = "asof_join" if case.row == 9 else "interval_join"
                    comparison.workload.assert_called_with(case.rows, scenario=scenario)
                    self.assertEqual(
                        set(active.streams), {"reference.input", "quotes.input"}
                    )
                    engine.stream_program.assert_called_with(
                        scenario, batch, None, batch_rows=64_000, retained_rows=None
                    )
                finally:
                    active.close()
            self.assertEqual(export.call_count, 6)

    def test_oracle_checks_both_identities_multiplicity_values_and_types(self):
        expected = expected_output()
        original = expected.combine_chunks()
        actual = expected.take(pa.array([2, 0, 1]))
        self.assertEqual(validate_interval_output(actual, expected, 2)["rows"], 3)
        invalid = (
            expected.take(pa.array([0, 0, 2])),
            expected.set_column(
                1, "right_sequence", pa.array([0, 0, 1], type=pa.uint64())
            ),
            expected.set_column(2, "value", pa.array([4.0, 7.0, 9.0])),
            expected.set_column(0, "sequence", pa.array([0, 0, 1], type=pa.int64())),
            expected.slice(0, 2),
        )
        for table in invalid:
            with self.subTest(table=table), self.assertRaises(ValueError):
                validate_interval_output(table, expected, 2)
        self.assertTrue(expected.equals(original))

    def test_fixture_seals_both_inputs_and_full_output_without_asof_barriers(self):
        engine, events, batch = engine_fixture()
        program = SimpleNamespace(
            to_project=lambda: SimpleNamespace(
                model_dump=lambda **_: {"graph": "interval"}
            ),
            compile_stream=lambda _: SimpleNamespace(fingerprint="f" * 64),
        )
        streams = {"reference.input": events, "quotes.input": events}
        original = copy.copy(streams)
        with (
            TemporaryDirectory() as raw,
            patch.dict("sys.modules", {engine.__name__: engine}),
            patch("benchmarks.engine_stream", engine, create=True),
        ):
            root = Path(raw)
            manifest = export_interval_fixture(
                root,
                program,
                streams,
                expected_output(),
                {"rows": 2, "batch_rows": 64, "mode": "off"},
            )
            self.assertEqual(manifest["rows_per_side"], 2)
            self.assertEqual(manifest["expected_output"]["rows"], 3)
            self.assertFalse(manifest["checkpoint_enabled"])
            self.assertEqual(set(manifest["inputs"]), set(streams))
            records = json.loads((root / "events.json").read_text())
            self.assertEqual(
                [item["kind"] for item in records],
                ["data", "data", "watermark", "watermark"],
            )
            self.assertEqual([item["input"] for item in records[:2]], list(streams))
        self.assertEqual(streams, original)
        self.assertEqual(batch.num_rows, 2)

    def test_acceptance_fixture_routes_interval_without_loading_native(self):
        from benchmarks.join_asof_acceptance import prepare_acceptance_fixture

        case = {"workload": "interval_join", "fixture_root": "interval-fixture"}
        with patch(
            "benchmarks.interval_acceptance.prepare_interval_fixture",
            return_value={"fixture": "interval"},
        ) as prepare:
            self.assertEqual(
                prepare_acceptance_fixture(case, Path("root")), {"fixture": "interval"}
            )
        prepare.assert_called_once_with(case, Path("interval-fixture"))


class IntervalClockTests(unittest.IsolatedAsyncioTestCase):
    async def test_interval_clock_feeds_both_streams_without_one_to_one_asof_gate(self):
        from benchmarks.join_asof_acceptance_runtime import _measure_interval_clock
        from scripts.benchmark_suite.issue406_measurement import PhaseClock

        engine, events, _ = engine_fixture()
        ready, complete = asyncio.Event(), asyncio.Event()
        ready.set()
        complete.set()
        sources = {
            name: SimpleNamespace(ready=ready, push=AsyncMock())
            for name in ("reference.input", "quotes.input")
        }
        sink = SimpleNamespace(
            complete=complete, rows=3, expected_rows=3, tables=[expected_output()]
        )
        bindings = SimpleNamespace(
            sources=sources, sink=sink, streams={name: events[:-1] for name in sources}
        )
        with (
            patch.dict("sys.modules", {engine.__name__: engine}),
            patch("benchmarks.engine_stream", engine, create=True),
        ):
            actual, seconds, evidence = await _measure_interval_clock(
                bindings, phase_clock=PhaseClock
            )
        self.assertTrue(actual.equals(expected_output()))
        self.assertGreater(seconds, 0)
        self.assertEqual(evidence["cpu_evidence"]["scope"], "processing")
        for source in sources.values():
            self.assertEqual(source.push.await_count, 2)
