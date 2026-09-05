"""Fail-closed checks for paired warm-stream profiling."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import psutil

from scripts import profile_warm_stream as profile
from scripts.profile_warm_stream import matrix_points, paired_summary


class WarmProfileTests(unittest.TestCase):
    def test_worker_thread_override_is_explicit_validated_and_immutable(self) -> None:
        original = {"TOKIO_WORKER_THREADS": "8", "PATH": "unchanged"}
        configured = profile._configured_worker_environment(original, 1)
        self.assertEqual(configured["TOKIO_WORKER_THREADS"], "1")
        self.assertEqual(original["TOKIO_WORKER_THREADS"], "8")
        self.assertEqual(
            profile._configured_worker_environment(original, None), original
        )
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                profile._configured_worker_environment(original, value)

    def test_sparse_matrix_is_seeded_and_keeps_layout_explicit(self) -> None:
        args = SimpleNamespace(
            history_rows=[128, 256],
            append_rows=[1, 4],
            append_entities=[1, 4],
            indicators=["rolling_mean"],
            gc_modes=["forced", "normal"],
            seed=7,
        )
        cases = profile.matrix_cases(args)
        self.assertEqual(len(cases), 16)
        self.assertEqual(cases, profile.matrix_cases(args))
        args.seed = 8
        self.assertNotEqual(cases, profile.matrix_cases(args))
        self.assertEqual({config["append_entities"] for config, _ in cases}, {1, 4})
        self.assertEqual({config["history_rows"] for config, _ in cases}, {128, 256})

    def test_original_matrix_never_silently_uses_sparse_timestamps(self) -> None:
        args = SimpleNamespace(
            history_rows=profile.HISTORY_ROWS,
            append_rows=profile.APPEND_ROWS,
            append_entities=None,
            indicators=["rolling_mean", "dual_sma_spread"],
            gc_modes=["forced", "normal"],
            seed=7,
        )
        cases = profile.matrix_cases(args)
        self.assertEqual(len(cases), 28)
        self.assertTrue(all("append_entities" not in config for config, _ in cases))

    def test_phase_summary_preserves_conversion_as_a_subset(self) -> None:
        first = {
            "seconds": 0.010,
            "phases_seconds": {
                "enqueue_to_source_data": 0.001,
                "source_data_to_source_watermark": 0.002,
                "source_watermark_to_sink": 0.003,
                "sink_to_receive": 0.004,
                "to_pyarrow": 0.0005,
            },
        }
        second = {
            "seconds": 0.020,
            "phases_seconds": {
                name: value * 2 for name, value in first["phases_seconds"].items()
            },
        }
        result = profile._phase_summary([first, second])
        self.assertAlmostEqual(result["to_pyarrow"]["p50_seconds"], 0.00075)
        self.assertAlmostEqual(result["sink_to_receive"]["p50_seconds"], 0.006)
        self.assertAlmostEqual(result["sink_to_receive"]["p95_seconds"], 0.0078)
        self.assertEqual(first["phases_seconds"]["to_pyarrow"], 0.0005)

    def test_phase_summary_rejects_invalid_or_double_counted_boundaries(self) -> None:
        phases = {
            "enqueue_to_source_data": 1.0,
            "source_data_to_source_watermark": 2.0,
            "source_watermark_to_sink": 3.0,
            "sink_to_receive": 4.0,
            "to_pyarrow": 0.5,
        }
        for name, value in (
            ("enqueue_to_source_data", -1),
            ("source_data_to_source_watermark", float("nan")),
            ("source_watermark_to_sink", 9),
            ("to_pyarrow", 5),
        ):
            with self.assertRaises(ValueError):
                profile._phase_summary(
                    [
                        {
                            "seconds": 10,
                            "phases_seconds": {**phases, name: value},
                        }
                    ]
                )
        with self.assertRaises(ValueError):
            profile._phase_summary([{"seconds": 10, "phases_seconds": {}}])
        with self.assertRaises(ValueError):
            profile._phase_summary([])

    def test_python_launch_environment_selects_the_current_interpreter(self) -> None:
        supplied = {"PATH": "caller-path", "PYTHONPATH": "caller-modules"}
        with patch.object(
            profile.shutil, "which", return_value=sys.executable
        ) as which:
            environment = profile._python_environment(supplied)
        self.assertEqual(environment["PYTHONPATH"], "caller-modules")
        self.assertEqual(supplied["PATH"], "caller-path")
        self.assertEqual(
            environment["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent)
        )
        which.assert_called_once_with("python", path=environment["PATH"])

    def test_python_launch_environment_rejects_a_different_interpreter(self) -> None:
        for selected in (None, __file__):
            with (
                patch.object(profile.shutil, "which", return_value=selected),
                self.assertRaisesRegex(ValueError, "current Python interpreter"),
            ):
                profile._python_environment({"PATH": "caller-path"})

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
    async def test_scheduler_comparison_rejects_different_python_wheels(self) -> None:
        manifests = [
            {"native_sha256": "same", "wheel_sha256": side}
            for side in ("left", "right")
        ]
        args = SimpleNamespace(
            baseline_build=Path("left"),
            candidate_build=Path("right"),
            worker_threads=(32, 1),
        )
        with (
            patch.object(
                profile, "_compatible_builds", AsyncMock(return_value=manifests)
            ),
            self.assertRaisesRegex(ValueError, "same native wheel"),
        ):
            await profile.compare(args)

    async def test_scheduler_comparison_requires_the_exact_declared_thread_override(
        self,
    ) -> None:
        workers = [
            SimpleNamespace(
                request=AsyncMock(
                    return_value={
                        "native_sha256": "same",
                        "tokio_worker_threads": str(count),
                    }
                )
            )
            for count in (32, 1)
        ]
        manifests = [{"native_sha256": "same"}] * 2
        result = await profile._worker_environments(
            workers, manifests, thread_counts=(32, 1)
        )
        self.assertEqual(result[1]["tokio_worker_threads"], "1")
        with self.assertRaisesRegex(ValueError, "thread override"):
            await profile._worker_environments(
                workers, manifests, thread_counts=(32, 2)
            )
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            await profile._worker_environments(workers, manifests)

    async def test_fresh_case_closes_both_workers_on_measurement_failure(self) -> None:
        workers = [SimpleNamespace(close=AsyncMock()) for _ in range(2)]
        with (
            patch.object(profile.Worker, "start", AsyncMock(side_effect=workers)),
            patch.object(profile, "_worker_environments", AsyncMock(return_value=[])),
            patch.object(
                profile,
                "_measure_case",
                AsyncMock(side_effect=ValueError("invalid sample")),
            ),
            self.assertRaisesRegex(ValueError, "invalid sample"),
        ):
            await profile._fresh_case(
                [{}, {}], {}, 2, False, profile.WorkerPairOptions(Path("target"))
            )
        for worker in workers:
            worker.close.assert_awaited_once()

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
