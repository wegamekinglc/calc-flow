from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.engine_lifecycle import run_with_completion
from scripts.benchmark_suite.diagnostics import (
    mark_phase,
    persist,
    stream_failure,
    trace_request,
)


class BenchmarkDiagnosticsTests(unittest.TestCase):
    def test_trace_flushes_phases_and_response_after_the_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = {
                "side": "baseline",
                "round": 1,
                "sample": 5,
                "phase": "measurement",
                "release": {"git_sha": "release"},
            }
            with trace_request(root, {"request_id": 7, "context": context}) as trace:
                mark_phase("restored")
                mark_phase("eof-enqueued")
                self.assertEqual(list(root.iterdir()), [])
                trace["response"] = {"correctness": {"passed": True}}
            record = json.loads((root / "worker-request-0007.json").read_text())
            self.assertEqual(record["context"], context)
            self.assertEqual(
                [row["phase"] for row in record["phases"]], ["restored", "eof-enqueued"]
            )
            self.assertEqual(record["response"], {"correctness": {"passed": True}})

    def test_diagnostic_write_failure_preserves_the_original_exception(self):
        original = ValueError("original failure")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "scripts.benchmark_suite.diagnostics.write_json",
                side_effect=OSError("disk unavailable"),
            ),
            self.assertRaises(ValueError) as raised,
            trace_request(Path(directory), {"request_id": 1}),
        ):
            raise original
        self.assertIs(raised.exception, original)
        self.assertIn("disk unavailable", original.__notes__[0])

    def test_snapshot_retains_other_components_when_job_status_fails(self):
        def broken():
            raise OSError("status unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persist(
                root / "manifests/manifest-0001.json", {"epoch": 1, "terminal": False}
            )
            source = SimpleNamespace(
                position=3,
                _pushed=4,
                events=(1, 2, 3, None),
                _events=asyncio.Queue(),
                opened=asyncio.Event(),
                ready=asyncio.Event(),
            )
            stream_failure(
                root,
                SimpleNamespace(status=broken),
                {"left": source},
                SimpleNamespace(rows=12, expected_rows=12),
                TimeoutError("original"),
            )
            record = json.loads((root / "failure-before-cancel.json").read_text())
            self.assertIn("status unavailable", record["job"]["diagnostic_error"])
            self.assertTrue(record["sources"]["left"]["eof_push_started"])
            self.assertFalse(record["sources"]["left"]["eof_consumed"])
            self.assertEqual(record["sink"]["rows"], 12)
            self.assertEqual(
                record["checkpoint"][0]["manifest"], {"epoch": 1, "terminal": False}
            )


class CompletionEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_snapshot_precedes_cancellation_of_the_completion_waiter(
        self,
    ):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        observed = []

        async def operation():
            await entered.wait()
            return "rows collected, both EOF queued"

        async def completion():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            patch("benchmarks.engine_lifecycle.COMPLETION_TIMEOUT", 0.001),
            self.assertRaises(TimeoutError),
        ):
            await run_with_completion(
                operation(),
                completion(),
                on_failure=lambda error: observed.append(
                    (type(error), cancelled.is_set())
                ),
            )
        self.assertEqual(observed, [(TimeoutError, False)])
        self.assertTrue(cancelled.is_set())

    async def test_failed_snapshot_does_not_replace_the_operation_error(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        original = ValueError("input failure")

        async def operation():
            await entered.wait()
            raise original

        async def completion():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        def broken_snapshot(error):
            raise OSError("snapshot unavailable")

        with self.assertRaises(ValueError) as raised:
            await run_with_completion(
                operation(), completion(), on_failure=broken_snapshot
            )
        self.assertIs(raised.exception, original)
        self.assertIn("snapshot unavailable", original.__notes__[0])
        self.assertTrue(cancelled.is_set())
