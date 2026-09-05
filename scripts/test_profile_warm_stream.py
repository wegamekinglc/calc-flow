"""Fail-closed checks for paired warm-stream profiling."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import psutil

from scripts import profile_warm_stream as profile
from scripts.profile_warm_stream import matrix_points, paired_summary


class WarmProfileTests(unittest.TestCase):
    def test_affinity_unavailable_does_not_abort_worker_handshake(self) -> None:
        for error in (psutil.AccessDenied(pid=1), NotImplementedError()):
            process = Mock(cpu_affinity=Mock(side_effect=error))
            with patch.object(psutil, "Process", return_value=process):
                self.assertIsNone(profile._cpu_affinity())

    def test_matrix_covers_history_and_increment_scales_without_duplicates(
        self,
    ) -> None:
        points = matrix_points()
        self.assertEqual(len(points), 7)
        self.assertIn((10_240_000, 64), points)
        self.assertIn((1_024_000, 64_000), points)
        self.assertEqual(len(set(points)), len(points))

    def test_paired_summary_preserves_speedup_direction_and_tail(self) -> None:
        result = paired_summary([2, 4, 6, 8], [1, 2, 3, 4], rows=64)
        self.assertEqual(result["paired_speedup_median"], 2)
        self.assertEqual(result["paired_speedup_ci95"], [2, 2])
        self.assertAlmostEqual(result["candidate"]["p95_seconds"], 3.85)

    def test_unpaired_or_invalid_samples_cannot_produce_evidence(self) -> None:
        for baseline, candidate in (
            ([1, 2], [1]),
            ([1, 2], [0, 1]),
            ([1, 2], [float("nan"), 1]),
            ([], []),
            ([1], [1]),
        ):
            with self.assertRaises(ValueError):
                paired_summary(baseline, candidate, rows=64)


class WarmProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_fingerprint_rejects_a_different_loaded_native_module(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(request=AsyncMock(return_value={"native_sha256": value}))
            for value in ("baseline", "unexpected")
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        with self.assertRaisesRegex(ValueError, "different native module"):
            await profile._worker_environments(workers, manifests)

    async def test_worker_fingerprint_requires_identical_runtime_dependencies(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(
                request=AsyncMock(
                    return_value={
                        "native_sha256": name,
                        "numpy": version,
                    }
                )
            )
            for name, version in (("baseline", "1"), ("candidate", "2"))
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            await profile._worker_environments(workers, manifests)

    async def test_matching_worker_fingerprints_keep_optional_affinity(self) -> None:
        environments = [
            {"native_sha256": name, "numpy": "1", "cpu_affinity": None}
            for name in ("baseline", "candidate")
        ]
        workers = [
            SimpleNamespace(request=AsyncMock(return_value=environment))
            for environment in environments
        ]
        manifests = [{"native_sha256": value} for value in ("baseline", "candidate")]
        self.assertEqual(
            await profile._worker_environments(workers, manifests), environments
        )

    async def test_worker_session_requires_start_and_cancels_unfinished_scenario(
        self,
    ) -> None:
        session = profile.WorkerSession(Path("."))
        for operation in ("sample", "finish"):
            with self.assertRaisesRegex(RuntimeError, "not started"):
                await session.dispatch({"operation": operation, "collect_gc": False})
        with self.assertRaisesRegex(ValueError, "unknown"):
            await session.dispatch({"operation": "invalid"})
        scenario = SimpleNamespace(job=SimpleNamespace(cancel_async=AsyncMock()))
        session.scenario = scenario
        await session.close()
        scenario.job.cancel_async.assert_awaited_once()
        await session.close()
        scenario.job.cancel_async.assert_awaited_once()

    async def test_metadata_process_failure_cannot_produce_a_fingerprint(self) -> None:
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"partial", b"metadata failed")),
            returncode=2,
        )
        with self.assertRaisesRegex(RuntimeError, "metadata failed"):
            await profile._command_output(process)

    async def test_git_metadata_rejects_non_metadata_commands_before_spawn(
        self,
    ) -> None:
        with (
            patch.object(
                asyncio, "create_subprocess_exec", new_callable=AsyncMock
            ) as spawn,
            self.assertRaisesRegex(ValueError, "metadata command"),
        ):
            await profile._git(Path("."), "checkout", "main")
        spawn.assert_not_awaited()

    async def test_git_metadata_uses_a_fixed_executable_and_argument_vector(
        self,
    ) -> None:
        process = SimpleNamespace(
            communicate=AsyncMock(return_value=(b"abc\n", b"")),
            returncode=0,
        )
        with patch.object(
            asyncio,
            "create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as spawn:
            self.assertEqual(await profile._git(Path("."), "rev-parse", "HEAD"), "abc")
        self.assertEqual(spawn.call_args.args, ("git", "rev-parse", "HEAD"))
        self.assertNotIn("shell", spawn.call_args.kwargs)

    async def test_missing_worker_pipes_raise_an_explicit_runtime_error(self) -> None:
        worker = profile.Worker(SimpleNamespace(stdin=None, stdout=None))
        with self.assertRaisesRegex(RuntimeError, "pipes"):
            await worker.request(operation="hello")

    async def test_worker_rejects_a_non_object_response(self) -> None:
        process = SimpleNamespace(
            stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
            stdout=SimpleNamespace(readline=AsyncMock(return_value=b"[]\n")),
        )
        with self.assertRaisesRegex(ValueError, "object"):
            await profile.Worker(process).request(operation="hello")


if __name__ == "__main__":
    unittest.main()
