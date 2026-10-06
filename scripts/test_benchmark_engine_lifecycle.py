from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from benchmarks.engine_lifecycle import interleaved_events, run_with_completion


class EngineInputTests(unittest.TestCase):
    def test_inputs_advance_together_without_changing_source_order(self):
        streams = {"right": ("r0", "rw0", "r1", "rw1"), "left": ("l0", "lw0")}
        self.assertEqual(
            list(interleaved_events(streams)),
            [
                ("right", "r0"),
                ("left", "l0"),
                ("right", "rw0"),
                ("left", "lw0"),
                ("right", "r1"),
                ("right", "rw1"),
            ],
        )
        self.assertEqual(streams["left"], ("l0", "lw0"))

    def test_interleaving_preserves_none_and_empty_inputs(self):
        self.assertEqual(
            list(interleaved_events({"empty": (), "input": (None,)})),
            [("input", None)],
        )


class EngineLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_job_interrupts_blocked_input(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def enqueue():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def completion():
            await entered.wait()
            return SimpleNamespace(state="failed", errors=["state limit exceeded"])

        with self.assertRaisesRegex(RuntimeError, "state limit exceeded"):
            await asyncio.wait_for(run_with_completion(enqueue(), completion()), 1)
        self.assertTrue(cancelled.is_set())

    async def test_operation_failure_cleans_up_completion_waiter(self):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def operation():
            await entered.wait()
            raise ValueError("input failure")

        async def completion():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with self.assertRaisesRegex(ValueError, "input failure"):
            await run_with_completion(operation(), completion())
        self.assertTrue(cancelled.is_set())

    async def test_success_waits_for_job_completion(self):
        ended = asyncio.Event()

        async def operation():
            ended.set()
            return ("table", 0.25)

        async def completion():
            await ended.wait()
            return SimpleNamespace(state="completed", errors=[])

        self.assertEqual(
            await run_with_completion(operation(), completion()), ("table", 0.25)
        )

    async def test_cancellation_drains_both_owned_tasks(self):
        started = (asyncio.Event(), asyncio.Event())
        cancelled = (asyncio.Event(), asyncio.Event())

        async def waiting(index):
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled[index].set()

        task = asyncio.create_task(run_with_completion(waiting(0), waiting(1)))
        await asyncio.gather(*(event.wait() for event in started))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(all(event.is_set() for event in cancelled))
