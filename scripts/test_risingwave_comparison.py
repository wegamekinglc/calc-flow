"""Focused input-boundary and evidence checks for the Linux comparison script."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

if sys.platform != "linux" or any(
    importlib.util.find_spec(name) is None
    for name in ("calc_flow", "numpy", "psutil", "psycopg", "pyarrow")
):
    raise unittest.SkipTest("requires Linux and the isolated comparison dependencies")

from scripts import risingwave_comparison as comparison  # noqa: E402


class ComparisonInputTests(unittest.TestCase):
    def test_insert_rejects_relations_outside_the_fixed_catalog(self):
        table = comparison.fixture(0, 2, progress=False)
        for name in ("unknown", "bench_grouped_640_0; DROP TABLE mutations"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(RuntimeError, "unknown benchmark relation"),
            ):
                comparison.insert_sql(table, name)

    def test_prepared_insert_preserves_negative_null_and_timestamp_literals(self):
        table = comparison.fixture(0, 2, progress=False)
        self.assertEqual(
            comparison.insert_sql(table, "bench_grouped_640_0"),
            "INSERT INTO bench_grouped_640_0 VALUES "
            "(0,0,NULL,CAST('2026-01-01T00:00:00+00:00' AS TIMESTAMPTZ)),"
            "(1,1,-127,CAST('2026-01-01T00:00:05+00:00' AS TIMESTAMPTZ))",
        )

    def test_saved_samples_produce_the_original_summary(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks/risingwave/results"
        records = [
            json.loads(path.read_text()) for path in sorted(root.glob("bench_*.json"))
        ]
        self.assertEqual(
            comparison.summaries(records),
            json.loads((root / "summary.json").read_text()),
        )

    def test_numeric_adaptation_rejects_strings_and_preserves_integer_extremes(self):
        for value in ("0); DROP TABLE mutations", True, 1.5):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "expected an integer or NULL"),
            ):
                comparison._integer_literal(value)
        for value in (-(2**63), -128, -1, 0, 2**63 - 1):
            self.assertEqual(comparison._integer_literal(value).as_string(), str(value))

    def test_oracle_keeps_null_groups_and_window_boundaries(self):
        table = comparison.pa.table(
            {
                "id": [0, 1, 2, -1],
                "k": [0, 0, 1, 0],
                "v": [2, None, None, None],
                "ts": [comparison.BASE + timedelta(seconds=s) for s in (9, 10, 10, 20)],
            },
            schema=comparison.SCHEMA,
        )
        self.assertEqual(comparison.expected("projection", [table]), [[0, 5]])
        self.assertEqual(
            comparison.expected("grouped", [table]), [[0, 2, 1], [1, None, 0]]
        )
        start = comparison.BASE_US
        self.assertEqual(
            comparison.expected("window", [table]),
            [
                [start, start + 10_000_000, 0, 2, 1],
                [start + 10_000_000, start + 20_000_000, 0, None, 0],
                [start + 10_000_000, start + 20_000_000, 1, None, 0],
            ],
        )


class CollectionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def _failed_collection(self, directory: Path, error: BaseException):
        async def records(*_):
            yield {"case": "projection_640", "samples": []}, False
            raise error

        rw = SimpleNamespace(process=None, stop=AsyncMock())
        args = SimpleNamespace(output=directory / "run", budget_seconds=5, reuse=None)
        with (
            patch.object(comparison, "provenance", AsyncMock(return_value={})),
            patch.object(comparison, "RisingWave", return_value=rw),
            patch.object(comparison, "_start_collection", AsyncMock()),
            patch.object(comparison, "_collect_cases", records),
            self.assertRaises(type(error)),
        ):
            await comparison.run(args)
        rw.stop.assert_awaited_once()
        completion = json.loads((args.output / "completion.json").read_text())
        self.assertEqual(completion["completed_cases"], 1)
        failure = json.loads((args.output / "failure.json").read_text())
        self.assertEqual(failure["type"], type(error).__name__)

    async def test_failure_preserves_completed_case_count_and_closes_service(self):
        with TemporaryDirectory() as directory:
            await self._failed_collection(
                Path(directory), RuntimeError("smoke failure")
            )

    async def test_cancellation_preserves_completed_case_count_and_closes_service(self):
        with TemporaryDirectory() as directory:
            await self._failed_collection(Path(directory), asyncio.CancelledError())

    async def test_cpu_probe_uses_absolute_executable_without_path_lookup(self):
        with patch.dict("os.environ", {"PATH": "/does-not-exist"}):
            description = await comparison._cpu_description()
        self.assertIn("Architecture:", description)


if __name__ == "__main__":
    unittest.main()
